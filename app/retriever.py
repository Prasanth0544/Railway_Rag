"""
retriever.py — ChromaDB Unified Retriever

Connects to the persistent ChromaDB instance and searches across
all collections: railway_rules, trains, stations, train_routes, references.

Supports both Gemini embeddings (cloud) and local sentence-transformers
(for fully offline operation with LM Studio).
"""

from __future__ import annotations

import os
import re
import sys

import chromadb  # type: ignore[import-untyped]
from langchain_core.documents import Document  # type: ignore[import-untyped]

# Force UTF-8 for Windows console (prevents emoji UnicodeEncodeError)
from app.logger import get_logger
logger = get_logger("app.retriever")

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass


# Force HuggingFace to use local cache only (no network calls on startup)
# The model is already downloaded; this prevents httpx errors when offline
import os as _os
if _os.getenv("HF_HUB_OFFLINE", "0") == "1":
    _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# CONFIG

CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")

# All known collections — only ones that actually exist will be loaded
ALL_COLLECTIONS = ["railway_rules", "trains", "stations", "train_routes", "references"]

# Results to pull per collection before merging
PER_COLLECTION_K = 10  # results per collection before merging (increased from 5 for better recall)

# Max docs returned from a single keyword $contains scan
# Must be large enough to cover all routes for any station code.
# NRT appears in 40 routes, BVRT in 60 — so 30 was cutting off results
# before the Python intersection could find trains with BOTH stations.
# 300 safely covers all routes per station (max observed: ~120 routes).
KEYWORD_SCAN_LIMIT = 300


# ─────────────────────────────────────────────
# EMBEDDINGS — Gemini (cloud) or local
# ─────────────────────────────────────────────

def get_embeddings():
    """
    Return the embedding model based on .env settings.

    USE_LOCAL_EMBEDDINGS=true  → sentence-transformers (100% offline)
    USE_LOCAL_EMBEDDINGS=false → Gemini embedding API (cloud)
    """
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
    api_key = os.getenv("GOOGLE_API_KEY", "")

    if not use_local and api_key and api_key not in ("your-gemini-api-key-here", ""):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore[import-untyped]
        logger.info("[CLOUD] Using Gemini embeddings (models/gemini-embedding-001)")
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )

    # Local offline embeddings
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore[import-untyped]
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[import-untyped]

    logger.info("[LOCAL] Using sentence-transformers/all-MiniLM-L6-v2 (offline, no rate limits)")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu", "local_files_only": False},
    )


def get_chroma_client() -> chromadb.ClientAPI:
    """Get persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_DB_DIR)


# ─────────────────────────────────────────────
# RAILWAY SYNONYM MAP
# ─────────────────────────────────────────────

# Maps short/colloquial railway terms → official terminology used in ChromaDB documents.
# Improves recall for jargon-heavy queries without retraining the embedding model.
_RAILWAY_SYNONYMS: dict[str, str] = {
    r"\bsl\b":      "Sleeper class SL",
    r"\b2a\b":      "AC 2 Tier 2A",
    r"\b3a\b":      "AC 3 Tier 3A",
    r"\bcc\b":      "AC Chair Car CC",
    r"\b1a\b":      "AC First Class 1A",
    r"\bec\b":      "Executive Chair Car EC",
    r"\b2s\b":      "Second Sitting 2S",
    r"\brac\b":     "RAC Reservation Against Cancellation",
    r"\bwl\b":      "Waiting List WL",
    r"\btte\b":     "Travelling Ticket Examiner TTE",
    r"\btatkal\b":  "Tatkal quota premium booking",
    r"\bpnr\b":     "PNR Passenger Name Record",
    r"\bntes\b":    "National Train Enquiry System NTES",
    r"\birctc\b":   "IRCTC Indian Railway Catering Tourism Corporation",
    r"\bsf\b":      "Superfast SF",
    r"\bexp\b":     "Express",
    r"\bpax\b":     "passenger",
    r"\bluggage\b": "luggage baggage allowance",
    r"\bbaggage\b": "luggage baggage allowance",
    r"\bfine\b":    "penalty fine charge",
    r"\bvizag\b":   "Visakhapatnam VSKP Vizag",
    r"\bhyd\b":     "Hyderabad HYD HYB SC Secunderabad",
    r"\bhyderabad\b": "Hyderabad HYD HYB SC Secunderabad Nampally",
    r"\bsecunderabad\b": "Secunderabad SC Hyderabad HYB",
    r"\bbzp\b":     "Vijayawada BZA",
    r"\bmas\b":     "Chennai Central MAS",
    r"\bndls?\b":   "New Delhi NDLS",
    r"\bcst\b":     "Chhatrapati Shivaji Terminus Mumbai CSTM",
    r"\bsbc\b":     "Bengaluru KSR SBC",
    r"\bbpl\b":     "Bhopal BPL",
}


def _expand_railway_synonyms(query: str) -> str:
    """
    Expand railway jargon and abbreviations in a query string.

    E.g. "SL class RAC quota" → "Sleeper class SL RAC Reservation Against Cancellation quota"
    """
    import re as _re
    result = query
    for pattern, expansion in _RAILWAY_SYNONYMS.items():
        result = _re.sub(pattern, expansion, result, flags=_re.IGNORECASE)
    return result


# ─────────────────────────────────────────────
# RECIPROCAL RANK FUSION (RRF)
# ─────────────────────────────────────────────

def _reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], k: int = 60
) -> list[Document]:
    """
    Merge multiple ranked document lists using Reciprocal Rank Fusion.
    Each doc's score = sum(1 / (k + rank)) across all lists it appears in.
    Docs found by multiple retrieval methods (vector + keyword + exact)
    get boosted to the top.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            doc_id = doc.page_content[:100]  # use content prefix as dedup key
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            # Keep the doc with the highest individual relevance_score
            existing = doc_map.get(doc_id)
            if existing is None or doc.metadata.get("relevance_score", 0) > existing.metadata.get("relevance_score", 0):
                doc_map[doc_id] = doc

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    result = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id]
        # Inject the RRF score into metadata for transparency
        doc.metadata["rrf_score"] = round(scores[doc_id], 6)
        result.append(doc)
    return result


# ─────────────────────────────────────────────
# UNIFIED RETRIEVER
# ─────────────────────────────────────────────

class UnifiedRetriever:
    """
    Searches across all ChromaDB collections (railway_rules, trains,
    stations, train_routes, references) and merges results by relevance score.

    Only loads collections that actually exist in ChromaDB — so partial
    data (e.g. rules-only) works fine.
    """

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        self.client = get_chroma_client()
        self.embeddings = get_embeddings()
        # Lazy store registry: collection name → Chroma object (or None = known but not yet opened)
        self.vector_stores: dict = {}
        self._discover_collections()
        self._init_station_resolver()

    def _discover_collections(self) -> None:
        """Discover which collections exist — but do NOT load their HNSW indices yet.
        
        Chroma vector stores are opened on-demand in _get_store() to avoid
        loading all HNSW indices into RAM at startup (critical for Render 512MB).
        """
        existing = {col.name for col in self.client.list_collections()}
        for name in ALL_COLLECTIONS:
            if name in existing:
                self.vector_stores[name] = None   # sentinel = known, not yet loaded

        if self.vector_stores:
            logger.info(f"[OK] Retriever ready — collections discovered (lazy): {list(self.vector_stores.keys())}")
        else:
            logger.warning("[WARN] No ChromaDB collections found! Run: python scripts/create_embeddings.py")

    def _get_store(self, name: str):
        """Return the Chroma vector store for `name`, opening it on first access.
        
        HNSW index files are memory-mapped by the OS — calling this only
        allocates the Python wrapper + triggers mmap, not a full RAM copy.
        Stores that are never needed during a query cycle are never loaded.
        """
        if name not in self.vector_stores:
            return None
        if self.vector_stores[name] is None:
            from langchain_chroma import Chroma  # type: ignore[import-untyped]
            try:
                self.vector_stores[name] = Chroma(
                    collection_name=name,
                    embedding_function=self.embeddings,
                    client=self.client,
                )
                logger.debug(f"[LAZY] Opened collection '{name}'")
            except Exception as exc:
                logger.warning(f"[WARN] Could not open collection '{name}': {exc}")
                del self.vector_stores[name]  # remove bad entry
                return None
        return self.vector_stores[name]

    def _init_station_resolver(self) -> None:
        """Initialize station alias search maps using build_station_lookup."""
        try:
            import difflib
            from scripts.preprocess import build_station_lookup
            
            lookup = build_station_lookup()
            if not lookup and "stations" in self.vector_stores:
                try:
                    col = self.client.get_collection("stations")
                    data = col.get(limit=15000, include=["metadatas"])
                    if data and data.get("metadatas"):
                        for meta in data["metadatas"]:
                            code = meta.get("station_code")
                            name = meta.get("station_name")
                            if code and name:
                                lookup[code] = {"name": name, "aka": []}
                except Exception as exc:
                    logger.warning(f"[WARN] Failed to load stations from ChromaDB fallback: {exc}")

            self.station_names_to_code = {}
            self.all_station_names = []
            
            for code, info in lookup.items():
                code_lower = code.lower()
                self.station_names_to_code[code_lower] = (info.get("name", code), code)
                
                name = info.get("name", "")
                if name:
                    name_lower = name.lower()
                    self.station_names_to_code[name_lower] = (name, code)
                    self.all_station_names.append(name)
                
                for aka in info.get("aka", []):
                    aka_lower = aka.lower()
                    # Use the OFFICIAL station name (not the alias) as canonical name.
                    # Route documents use official names like "Visakhapatnam",
                    # not AKAs like "Vizag", so $contains must search the official name.
                    official_name = name if name else aka
                    self.station_names_to_code[aka_lower] = (official_name, code)
                    self.all_station_names.append(aka)
                    
            # Also index the first significant word of each multi-word station name
            # so queries like "Vijayawada" match "Vijayawada Junction" (BZA)
            # and "Hyderabad" matches "Hyderabad Deccan Nampally" (HYB).
            _noise = {"new", "old", "north", "south", "east", "west", "central",
                      "junction", "road", "halt", "cabin", "town", "city"}
            for code, info in lookup.items():
                name = info.get("name", "")
                if not name:
                    continue
                parts = name.split()
                if len(parts) > 1:
                    first = parts[0].lower()
                    if len(first) >= 4 and first not in _noise and first not in self.station_names_to_code:
                        self.station_names_to_code[first] = (name, code)
                        self.all_station_names.append(first)

            # Pre-sort station names by length descending ONCE (avoids re-sorting on every query)
            self._sorted_station_names = sorted(self.station_names_to_code.keys(), key=len, reverse=True)

            logger.info(f"[Resolver] Loaded {len(self.station_names_to_code)} names/AKAs for fuzzy resolution")
        except Exception as e:
            logger.warning(f"[WARN] Failed to load station lookup for resolver: {e}")
            self.station_names_to_code = {}
            self.all_station_names = []
            self._sorted_station_names = []

    def _resolve_station(self, query: str) -> tuple[str, str] | None:
        """
        Fuzzy match query text against station names or codes.
        Returns: tuple of (canonical_name, station_code) or None
        """
        if not self.station_names_to_code:
            return None

        import difflib
        query_lower = query.lower()

        # 1. Exact phrase/name/code scan with word boundaries (longest first to avoid noise)
        # Use pre-sorted list built at init time — avoids O(n log n) sort on every query
        sorted_names = self._sorted_station_names
        stop_words = {
            "the", "and", "for", "are", "get", "new", "old", "can", "our", "out",
            "all", "its", "any", "not", "but", "who", "you", "she", "his", "her",
            "him", "has", "had", "was", "web", "doc", "app", "now", "day", "runs"
        }
        for name in sorted_names:
            if len(name) > 2:  # Ignore 1-2 char noise
                if name in stop_words:
                    continue
                if re.search(rf"\b{re.escape(name)}\b", query_lower):
                    return self.station_names_to_code[name]

        # 2. Fuzzy match single word tokens (cutoff 0.8)
        # We split the query into words (length >= 4)
        words = re.findall(r"\b[a-zA-Z]{4,}\b", query_lower)
        ignore_words = {
            "train", "trains", "station", "stations", "route", "routes", "daily", "weekly",
            "what", "where", "from", "stop", "stops", "here", "arrive", "departure",
            "cancellation", "cancel", "charge", "charges", "rule", "rules", "luggage", "class"
        }
        for word in words:
            if word in ignore_words:
                continue
            # Fuzzy match against canonical and AKA names
            close_matches = difflib.get_close_matches(word, self.all_station_names, n=1, cutoff=0.8)
            if close_matches:
                canonical_match = close_matches[0]
                return self.station_names_to_code[canonical_match.lower()]

        return None

    def _resolve_all_stations(self, query: str) -> list[tuple[str, str]]:
        """
        Find ALL station mentions in the query (e.g. 'between Vijayawada and Hyderabad'
        returns both). Returns list of (canonical_name, station_code) tuples.
        """
        if not self.station_names_to_code:
            return []

        import difflib
        query_lower = query.lower()
        found_with_pos: list[tuple[int, tuple[str, str]]] = []
        matched_spans: list[tuple[int, int]] = []

        _stop = {
            "the", "and", "for", "are", "get", "new", "old", "can", "our", "out",
            "all", "its", "any", "not", "but", "who", "you", "she", "his", "her",
            "him", "has", "had", "was", "web", "doc", "app", "now", "day", "runs"
        }

        # Pass 1: scan pre-sorted names for exact boundary matches
        for name in self._sorted_station_names:
            if len(name) <= 2 or name in _stop:
                continue
            m = re.search(rf"\b{re.escape(name)}\b", query_lower)
            if m:
                s, e = m.start(), m.end()
                # Skip if this span overlaps an already-matched region
                if any(s < me and e > ms for ms, me in matched_spans):
                    continue
                matched_spans.append((s, e))
                entry = self.station_names_to_code[name]
                if not any(e == entry for _, e in found_with_pos):
                    found_with_pos.append((s, entry))

        # Pass 2: fuzzy match remaining unmatched long words
        ignore_words = {
            "train", "trains", "station", "stations", "route", "routes", "daily", "weekly",
            "what", "where", "from", "stop", "stops", "here", "arrive", "departure",
            "cancellation", "cancel", "charge", "charges", "rule", "rules", "luggage", "class",
            "between", "which", "running", "express", "superfast", "mail"
        }
        words = list(re.finditer(r"\b[a-zA-Z]{4,}\b", query_lower))
        for match in words:
            word = match.group()
            s = match.start()
            if word in ignore_words or word in _stop:
                continue
            close_matches = difflib.get_close_matches(word, self.all_station_names, n=1, cutoff=0.82)
            if close_matches:
                entry = self.station_names_to_code.get(close_matches[0].lower())
                if entry and not any(e == entry for _, e in found_with_pos):
                    found_with_pos.append((s, entry))

        # Sort by appearance position s in the query (earliest first)
        found_with_pos.sort(key=lambda x: x[0])
        return [entry for _, entry in found_with_pos]

    def _lookup_by_train_number(self, query: str) -> tuple[list[Document], list[str]]:
        """
        If the query contains a 5-digit train number (e.g. 12727),
        do a direct ChromaDB metadata lookup in both 'trains' and
        'train_routes' collections and return exact matches.
        This bypasses semantic search for precise train number queries.

        Returns: (list of matched Documents, list of matched train numbers)
        """
        numbers = re.findall(r"\b(\d{5})\b", query)
        if not numbers:
            return [], []

        exact_docs: list[Document] = []
        matched_numbers: list[str] = []

        # Collections to search for exact train number matches
        lookup_collections = [
            ("trains",       "trains",       1.0),
            ("train_routes", "train_routes", 1.0),
        ]

        for col_name, label, score in lookup_collections:
            if col_name not in self.vector_stores:
                continue
            try:
                col = self.client.get_collection(col_name)
            except Exception:
                continue

            for num in numbers:
                try:
                    result = col.get(
                        where={"train_no": num},
                        limit=5,
                        include=["documents", "metadatas"],
                    )
                    if result["documents"]:
                        for i in range(len(result["documents"])):
                            doc = Document(
                                page_content=result["documents"][i],
                                metadata={**result["metadatas"][i], "collection": label, "relevance_score": score},
                            )
                            exact_docs.append(doc)
                        train_name = result["metadatas"][0].get("train_name", "")
                        if num not in matched_numbers:
                            matched_numbers.append(num)
                        logger.debug(f"[EXACT] {label}: train {num} — {train_name} ({len(result['documents'])} docs)")
                except Exception as exc:
                    logger.warning(f"[WARN] Lookup failed in '{col_name}' for train {num}: {exc}")

        return exact_docs, matched_numbers

    # ── Train Validation ─────────────────────────────────────────────────────

    def _validate_route_candidates(
        self,
        docs: list[Document],
        station_terms: list[tuple[str, str]],
    ) -> tuple[list[Document], list[Document], list[str]]:
        """
        Validate candidate docs for a 2-station route query.

        Checks (for train_route docs):
          1. Both queried stations present in doc content
          2. Correct travel direction — FROM station before TO station in stops
          3. Wrong-direction train metadata (trains collection) dropped from other_docs

        Returns:
          valid_routes  — route docs that pass all checks (both_match equivalent)
          other_docs    — non-route docs + correctly filtered pass-throughs
          both_trains   — list of matched train_no strings
        """
        valid_routes: list[Document] = []
        other_docs: list[Document] = []

        from_canon, from_code = station_terms[0]

        # All stations beyond [0] are treated as alternative TO stations
        # e.g. "Hyderabad" resolves to both HYB and SC — trains to either qualify
        to_terms = station_terms[1:]  # list of (canon, code) for destination alternatives

        for doc in docs:
            src_type = doc.metadata.get("source_type", "")

            # Non-route docs go to other_docs by default
            if src_type != "train_route":
                other_docs.append(doc)
                continue

            content_lower = doc.page_content.lower()

            # Check 1: FROM station present
            has_from = from_canon in content_lower or from_code in content_lower

            # Check 1b: ANY of the TO alternatives present (handles HYB/SC both = Hyderabad)
            has_to = any(
                canon in content_lower or code in content_lower
                for canon, code in to_terms
            )

            if not (has_from and has_to):
                continue  # drop silently — doesn't serve this route

            # Check 2: Correct direction (FROM before nearest TO in stop sequence)
            # Direction check — handles both doc formats:
            #   New (time-based): "Header\nVSKP dep 17:20 | BZA arr 23:15 | HYB arr 06:15 [last]."
            #   Old (stop-codes): "Stops (N): VSKP > DVD > BZA > HYB"
            try:
                _content = doc.page_content
                _stops: list[str] = []
                if "\n" in _content and "|" in _content:
                    # New format: extract first token (station code) from each pipe segment
                    _sched = _content.split("\n", 1)[1]
                    _stops = [seg.strip().split()[0].lower()
                              for seg in _sched.split("|")
                              if seg.strip() and seg.strip().split()]
                elif "Stops" in _content:
                    # Old format: split stop sequence by >
                    _stops = [s.strip().lower()
                              for s in _content.split("Stops")[1].split(">")]

                if _stops:
                    fi = next((i for i, s in enumerate(_stops) if from_canon in s or from_code in s), -1)
                    # Find the earliest TO station position among all alternatives
                    ti = -1
                    for to_canon, to_code in to_terms:
                        ti_alt = next((i for i, s in enumerate(_stops) if to_canon in s or to_code in s), -1)
                        if ti_alt >= 0 and (ti < 0 or ti_alt < ti):
                            ti = ti_alt
                    if fi >= 0 and ti >= 0 and fi >= ti:
                        logger.debug(
                            f"[DIR] Dropped wrong-direction {doc.metadata.get('train_no','?')}: "
                            f"{from_code}@{fi} >= TO@{ti}"
                        )
                        continue  # wrong direction
            except Exception:
                pass  # parsing failed — keep doc to be safe

            valid_routes.append(doc)

        both_trains = [d.metadata.get("train_no", "?") for d in valid_routes]
        logger.debug(f"[VALIDATE] valid_routes={len(valid_routes)}, trains={both_trains}")

        # Check 3: Drop trains collection docs for trains NOT in valid set
        if both_trains:
            filtered_other: list[Document] = []
            for doc in other_docs:
                if (doc.metadata.get("collection") == "trains"
                        and doc.metadata.get("train_no") not in both_trains):
                    logger.debug(f"[VALIDATE] Dropped wrong train metadata: {doc.metadata.get('train_no','?')}")
                    continue
                filtered_other.append(doc)
            other_docs = filtered_other

        return valid_routes, other_docs, both_trains

    # ── Train Name & Schedule Enrichment ─────────────────────────────────────

    def _enrich_with_train_names(
        self,
        valid_routes: list[Document],
        other_docs: list[Document],
        both_trains: list[str],
    ) -> tuple[list[Document], list[Document]]:
        """
        For each matched train, fetch its metadata doc from the trains collection
        and inject it at the top of valid_routes. Also stamps 'Runs on: X' onto
        the matching route doc so Gemini can distinguish daily vs seasonal trains.
        """
        if "trains" not in self.vector_stores:
            return valid_routes, other_docs
        try:
            trains_col = self.client.get_collection("trains")
            for train_no in both_trains:
                res = trains_col.get(
                    where={"train_no": train_no},
                    limit=1,
                    include=["documents", "metadatas"],
                )
                if not res["documents"]:
                    continue

                train_info = res["documents"][0]

                # Inject name doc at front so Gemini sees name before route
                name_doc = Document(
                    page_content=train_info,
                    metadata={**res["metadatas"][0], "relevance_score": 0.95},
                )
                valid_routes.insert(0, name_doc)

                # Stamp running days onto matching route doc
                runs_on = ""
                if "Runs on:" in train_info:
                    runs_on = train_info.split("Runs on:")[1].split(".")[0].strip()
                if runs_on:
                    for rd in valid_routes:
                        if (rd.metadata.get("source_type") == "train_route"
                                and rd.metadata.get("train_no") == train_no):
                            rd.page_content += f" | Runs on: {runs_on}"

                logger.debug(f"[ENRICH] Injected name+schedule for {train_no} (runs: {runs_on})")
        except Exception as exc:
            logger.debug(f"[ENRICH] Train name lookup failed: {exc}")

        return valid_routes, other_docs

    def retrieve(self, query: str, intent_category: str = "") -> list[Document]:

        """
        Search relevant collections and return the top-k most relevant
        documents merged, sorted by score.

        Uses fuzzy station query rewriting, keyword substring search (hybrid search),
        intent-based collection filtering, and threshold filtering.

        Args:
            query:           The user's question string.
            intent_category: Fine-grained intent from intent.py (e.g. BETWEEN_STATIONS,
                             CANCELLATION_RULES). When provided, overrides keyword-based
                             collection routing in Step 3.
        """
        if not self.vector_stores:
            return []

        # --- Step 1: Exact train number lookup ---
        exact_docs, matched_train_numbers = self._lookup_by_train_number(query)
        train_number_detected = len(matched_train_numbers) > 0

        # --- Step 1b: Railway synonym expansion ---
        query = _expand_railway_synonyms(query)

        # --- Step 2: Resolve ALL station names in the query ---
        # _resolve_all_stations returns stations sorted by appearance order (s) in the query,
        # ensuring station[0] is always the FROM station (e.g. BZA before HYB/SC).
        all_stations = self._resolve_all_stations(query)
        # Expose last resolved stations to RAGChain for confidence check
        self._last_all_stations = all_stations
        # Keep backward-compatible single station_info for route trimming logic
        station_info = all_stations[0] if all_stations else None


        # --- Step 2b: Direct station code extraction (handles NRT, BVRT, bza, sc etc.) ---
        # Extracts 2-5 letter tokens from the query and looks them up directly in
        # ChromaDB station metadata. This handles:
        #   - Lowercase: "trains from nrt to bvrt"
        #   - Uppercase: "trains from NRT to BVRT"
        # The fuzzy resolver misses short codes (< 4 chars) and fails when map is empty.
        raw_codes = re.findall(r'\b([a-zA-Z]{2,5})\b', query)
        _code_noise = {
            "AC", "DC", "ID", "OK", "OR", "IS", "AT", "IN", "OF",
            "TO", "BY", "UP", "ON", "AS", "AN", "MY", "WE", "THE",
            "AND", "FOR", "FROM", "VIA", "ARE", "GET", "NOW", "DAY",
            "TRAIN", "TRAINS", "ROUTE", "STOP", "STOPS",
        }
        for raw in raw_codes:
            code = raw.upper()          # normalize to uppercase (NRT, BVRT, BZA)
            if code in _code_noise:
                continue
            already_found = any(c == code for _, c in all_stations)
            if already_found:
                continue
            if "stations" in self.vector_stores:
                try:
                    col = self.client.get_collection("stations")
                    res = col.get(
                        where={"station_code": code},
                        limit=1,
                        include=["metadatas"],
                    )
                    if res["metadatas"]:
                        sname = res["metadatas"][0].get("station_name", code)
                        all_stations.append((sname, code))
                        if station_info is None:
                            station_info = (sname, code)
                        logger.debug(f"[CODE_LOOKUP] '{raw}' → '{code}' ({sname})")
                except Exception as exc:
                    logger.debug(f"[CODE_LOOKUP] Failed for '{code}': {exc}")

        # Build enriched search query with all resolved names + codes
        search_query = query
        if all_stations:
            extras = []
            for canon, code in all_stations:
                if code.lower() not in query.lower():
                    extras.append(code)
                if canon.lower() not in query.lower():
                    extras.append(canon)
            if extras:
                search_query = f"{query} {' '.join(extras)}"
                logger.debug(f"[REWRITE] Multi-station resolved: '{query}' -> '{search_query}")

        # --- Step 3: Intent-Based Collection Filtering (Metadata Routing) ---
        query_lower = query.lower()
        active_collections = list(self.vector_stores.keys())

        # Priority 1: Use fine-grained intent_category from intent.py (smarter)
        _CATEGORY_TO_COLS: dict[str, list[str]] = {
            "BETWEEN_STATIONS":   ["train_routes"],
            "SCHEDULE_QUERY":     ["train_routes", "trains"],
            "STATION_INFO":       ["stations"],
            "COACH_QUERY":        ["trains", "railway_rules"],
            "CANCELLATION_RULES": ["railway_rules", "references"],
            "GENERAL_INFO":       ["railway_rules", "references", "trains"],
        }
        if intent_category and intent_category in _CATEGORY_TO_COLS:
            wanted = _CATEGORY_TO_COLS[intent_category]
            active_collections = [c for c in wanted if c in self.vector_stores]
            logger.debug(f"[INTENT-CAT] {intent_category} → collections: {active_collections}")

        # Priority 2: Keyword-based fallback (when intent_category not provided)
        elif not intent_category:
            transit_keywords = ["stop", "stops", "route", "timings", "timetable", "departure", "arrive", "arrival", "halt", "pass through", "runs from"]
            rules_keywords   = ["cancel", "cancellation", "refund", "luggage", "penalty", "fine", "tte", "rule", "duty", "duties", "allowance", "charge", "charges", "fee"]
            if any(kw in query_lower for kw in transit_keywords):
                active_collections = [c for c in active_collections if c in ("train_routes", "stations", "trains")]
                logger.debug(f"[INTENT-KW] Transit → collections: {active_collections}")
            elif any(kw in query_lower for kw in rules_keywords):
                active_collections = [c for c in active_collections if c in ("railway_rules", "references")]
                logger.debug(f"[INTENT-KW] Rules → collections: {active_collections}")

        # --- Step 4: Hybrid Keyword Contains Matching for ALL resolved stations ---
        keyword_docs: list[Document] = []
        if all_stations:
            for canonical_name, station_code in all_stations:
                # Try both the canonical name AND the station code for broader matches
                search_terms = list({canonical_name, station_code})  # deduplicated
                for name in active_collections:
                    if name in ("train_routes", "stations"):
                        for term in search_terms:
                            if not term or len(term) < 2:
                                continue
                            try:
                                col = self.client.get_collection(name)
                                res = col.get(
                                    where_document={"$contains": term},
                                    limit=KEYWORD_SCAN_LIMIT,  # cap to avoid OOM on large collections
                                )
                                if res["documents"]:
                                    logger.debug(f"[KEYWORD] Found {len(res['documents'])} matches in '{name}' containing '{term}'")
                                    for i in range(len(res["documents"])):
                                        content_raw = res["documents"][i]
                                        content_lower = content_raw.lower()
                                        
                                        # 1. Term frequency factor (normalized, max +0.20)
                                        freq = content_lower.count(term.lower())
                                        freq_score = min(freq / 5.0, 1.0)
                                        
                                        # 2. Multi-term bonus (if both name and code are in text, +0.10)
                                        all_terms_present = all(t.lower() in content_lower for t in search_terms if t)
                                        multi_bonus = 0.10 if all_terms_present else 0.0
                                        
                                        # 3. Collection weight
                                        col_weight = 1.0 if name == "train_routes" else 0.85
                                        
                                        base_score = 0.70 + (0.20 * freq_score) + multi_bonus
                                        final_score = round(base_score * col_weight, 4)
                                        
                                        # Cutoff threshold — ignore weak keyword matches (e.g. single off-hand mention in rules)
                                        # Threshold must be ≤ 0.74 for train_route docs where a station code
                                        # appears exactly once (freq=1 → score=0.74). The old 0.75 cut them off.
                                        cutoff = 0.65 if name == "train_routes" else 0.75
                                        if final_score < cutoff:
                                            continue

                                            
                                        doc = Document(
                                            page_content=content_raw,
                                            metadata={**res["metadatas"][i], "collection": name, "relevance_score": final_score}
                                        )
                                        keyword_docs.append(doc)
                            except Exception as exc:
                                logger.warning(f"[WARN] Keyword contains query failed for '{name}' (term='{term}'): {exc}")

        # --- Step 5: Semantic search across active collections ---
        all_results: list[tuple[Document, float]] = []

        for name in active_collections:
            store = self._get_store(name)
            if not store:
                continue
            try:
                results = store.similarity_search_with_relevance_scores(
                    search_query, k=PER_COLLECTION_K
                )
                for doc, score in results:
                    if score < 0.30:
                        continue  # tightened threshold (was 0.10) — reduces noise docs
                    doc.metadata["collection"] = name
                    doc.metadata["relevance_score"] = round(score, 4)
                    all_results.append((doc, score))
            except Exception as exc:
                logger.warning(f"[WARN] Error searching '{name}': {exc}")

        all_results.sort(key=lambda x: x[1], reverse=True)
        semantic_docs = [doc for doc, _ in all_results[: self.top_k]]

        # --- Step 6: Merge via Reciprocal Rank Fusion & Deduplicate ---
        fused_docs = _reciprocal_rank_fusion([exact_docs, keyword_docs, semantic_docs])
        deduped_docs = []
        seen_content = set()
        for doc in fused_docs:
            snippet = doc.page_content[:100]
            if snippet not in seen_content:
                seen_content.add(snippet)
                deduped_docs.append(doc)

        # --- Step 6b + 7 + 9: Train Validation (all checks in one stage) ---
        # Check 1: Both queried stations present in doc
        # Check 2: Correct travel direction (FROM before TO in stop sequence)
        # Check 3: Drop wrong-direction train metadata docs from semantic results
        if len(all_stations) >= 2:
            station_terms = [(canon.lower(), code.lower()) for canon, code in all_stations]
            valid_routes, other_docs, both_trains = self._validate_route_candidates(
                deduped_docs, station_terms
            )

            if valid_routes:
                # ── Metadata injection REMOVED (2026-08-05) ──────────────────
                # All 12,341 train_routes docs are now in the enriched format:
                #   "Train 12727 — Godavari SF Express (Daily). From VSKP to HYB. Stops..."
                # Name + running days are already embedded in the route doc text.
                # Injecting the trains-collection doc would DUPLICATE this info,
                # wasting ~230 chars × N trains = ~8,000+ chars for broad queries.
                # _enrich_with_train_names() is kept below for reference/rollback only.
                logger.debug(f"[ENRICH] Injection skipped — all route docs are enriched format ({len(valid_routes)} docs)")
            else:
                route_codes = [d.metadata.get('train_no', '?') for d in deduped_docs
                               if d.metadata.get('source_type') == 'train_route']
                logger.debug(f"[INTERSECT] NO match. station_terms={station_terms}, routes present={route_codes}")

            deduped_docs = valid_routes + other_docs




        # --- Step 6c: Filter out wrong station docs ---
        # When we have resolved specific station codes from the query (e.g. NRT, BVRT),
        # remove any station-type docs whose code doesn't match.
        # This stops semantic search from returning NRSP (Narasimhapura) when user
        # queries "nrt" and confusing Gemini into wrong station name mappings.
        if all_stations:
            resolved_codes = {code.upper() for _, code in all_stations}
            filtered: list[Document] = []
            for doc in deduped_docs:
                if doc.metadata.get("source_type") == "station":
                    doc_code = doc.metadata.get("station_code", "").upper()
                    if doc_code and doc_code not in resolved_codes:
                        logger.debug(f"[FILTER] Dropped wrong station doc: {doc_code} (not in {resolved_codes})")
                        continue   # skip this wrong station doc
                filtered.append(doc)
            deduped_docs = filtered

        # --- Step 7: Trim Route Schedules to key stops only ---
        # Handles BOTH doc formats:
        #   New (time-based): "Header\nVSKP dep 17:20 | BZA arr 23:15 dep 23:30 (15min) | HYB arr 06:15 [last]."
        #   Old (stop-codes): "Train X. Stops (N): VSKP > DVD > BZA > HYB. Distance: Y km."
        # Keeps: first segment (origin) + segments matching queried stations + last segment (destination)
        if all_stations and not train_number_detected:
            all_station_terms = set()
            for canon, code in all_stations:
                all_station_terms.add(canon.lower())
                all_station_terms.add(code.lower())

            for doc in deduped_docs:
                if doc.metadata.get("source_type") != "train_route":
                    continue
                content = doc.page_content

                if "\n" in content and "|" in content:
                    # ── New time-based format ────────────────────────────────
                    # "Train 12727 — SF Express (Daily). From VSKP to HYB. 21 stops, 707 km.\n"
                    # "VSKP dep 17:20 | DVD arr 17:45 dep 17:47 (2min) | BZA arr 23:15 dep 23:30 (15min) | HYB arr 06:15 [last]."
                    parts = content.split("\n", 1)
                    header = parts[0]            # "Train 12727 — SF Express..."
                    sched  = parts[1] if len(parts) > 1 else ""

                    segments = [seg.strip() for seg in sched.rstrip(".").split("|") if seg.strip()]
                    if len(segments) <= 4:
                        continue  # already short

                    trimmed_segs = []
                    for idx, seg in enumerate(segments):
                        is_first  = (idx == 0)
                        is_last   = (idx == len(segments) - 1)
                        # First word of segment = station code (e.g. "BZA" from "BZA arr 23:15")
                        seg_code  = seg.split()[0].lower() if seg.split() else ""
                        is_target = any(term in seg.lower() for term in all_station_terms) \
                                    or seg_code in all_station_terms
                        if is_first or is_last or is_target:
                            trimmed_segs.append(seg)

                    if not trimmed_segs:
                        continue

                    trimmed_content = header + "\n" + " | ".join(trimmed_segs) + "."
                    logger.debug(
                        f"[TRIM-NEW] {doc.metadata.get('train_no','?')}: "
                        f"{len(segments)} segs → {len(trimmed_segs)} "
                        f"({len(content)} → {len(trimmed_content)} chars)"
                    )
                    doc.page_content = trimmed_content

                elif "Stops" in content:
                    # ── Old stop-codes format ────────────────────────────────
                    # "Train X. Stops (N): VSKP > DVD > BZA > HYB. Distance: Y km."
                    stops_idx    = content.index("Stops")
                    header       = content[:stops_idx]
                    stops_section = content[stops_idx:]
                    colon_idx    = stops_section.index(":")
                    stops_raw    = stops_section[colon_idx + 1:].strip()
                    all_stops    = [s.strip() for s in stops_raw.split(">") if s.strip()]

                    if len(all_stops) <= 4:
                        continue

                    trimmed = []
                    for idx, stop in enumerate(all_stops):
                        is_first  = (idx == 0)
                        is_last   = (idx == len(all_stops) - 1)
                        is_target = any(term in stop.lower() for term in all_station_terms)
                        if is_first or is_last or is_target:
                            trimmed.append(stop)

                    if not trimmed:
                        continue

                    trimmed_content = (
                        header
                        + f"Stops ({len(all_stops)} total, key stops shown): "
                        + " > ".join(trimmed)
                    )
                    logger.debug(
                        f"[TRIM-OLD] {doc.metadata.get('train_no','?')}: "
                        f"{len(all_stops)} stops → {len(trimmed)} "
                        f"({len(content)} → {len(trimmed_content)} chars)"
                    )
                    doc.page_content = trimmed_content

        # Expand limit when station or train number detected to avoid truncation
        limit = 25 if (station_info or train_number_detected) else self.top_k
        return deduped_docs[:limit]

    def __call__(self, query: str) -> list[Document]:
        """Make retriever callable (for use in LCEL chains)."""
        return self.retrieve(query)


# ─────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────

def get_unified_retriever(top_k: int = 5) -> UnifiedRetriever:
    """Factory function to create a UnifiedRetriever instance."""
    return UnifiedRetriever(top_k=top_k)
