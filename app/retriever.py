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
PER_COLLECTION_K = 8


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
        self.vector_stores: dict = {}
        self._load_collections()
        self._init_station_resolver()

    def _load_collections(self) -> None:
        """Load all existing ChromaDB collections."""
        from langchain_chroma import Chroma  # type: ignore[import-untyped]

        existing = {col.name for col in self.client.list_collections()}

        for name in ALL_COLLECTIONS:
            if name not in existing:
                continue
            try:
                self.vector_stores[name] = Chroma(
                    collection_name=name,
                    embedding_function=self.embeddings,
                    client=self.client,
                )
            except Exception as exc:
                logger.warning(f"[WARN] Could not load collection '{name}': {exc}")

        if self.vector_stores:
            logger.info(f"[OK] Retriever ready — collections: {list(self.vector_stores.keys())}")
        else:
            logger.warning("[WARN] No ChromaDB collections found! Run: python scripts/create_embeddings.py")

    def _init_station_resolver(self) -> None:
        """Initialize station alias search maps using build_station_lookup."""
        try:
            import difflib
            from scripts.preprocess import build_station_lookup
            
            lookup = build_station_lookup()
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
        found: list[tuple[str, str]] = []
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
                if entry not in found:
                    found.append(entry)

        # Pass 2: fuzzy match remaining unmatched long words
        ignore_words = {
            "train", "trains", "station", "stations", "route", "routes", "daily", "weekly",
            "what", "where", "from", "stop", "stops", "here", "arrive", "departure",
            "cancellation", "cancel", "charge", "charges", "rule", "rules", "luggage", "class",
            "between", "which", "running", "express", "superfast", "mail"
        }
        words = re.findall(r"\b[a-zA-Z]{4,}\b", query_lower)
        for word in words:
            if word in ignore_words or word in _stop:
                continue
            close_matches = difflib.get_close_matches(word, self.all_station_names, n=1, cutoff=0.82)
            if close_matches:
                entry = self.station_names_to_code.get(close_matches[0].lower())
                if entry and entry not in found:
                    found.append(entry)

        return found

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

    def retrieve(self, query: str) -> list[Document]:
        """
        Search relevant collections and return the top-k most relevant
        documents merged, sorted by score.

        Uses fuzzy station query rewriting, keyword substring search (hybrid search),
        intent-based collection filtering, and threshold filtering.
        """
        if not self.vector_stores:
            return []

        # --- Step 1: Exact train number lookup ---
        exact_docs, matched_train_numbers = self._lookup_by_train_number(query)
        train_number_detected = len(matched_train_numbers) > 0

        # --- Step 1b: Railway synonym expansion ---
        query = _expand_railway_synonyms(query)

        # --- Step 2: Resolve ALL station names in the query ---
        all_stations = self._resolve_all_stations(query)
        # Keep backward-compatible single station_info for route trimming logic
        station_info = all_stations[0] if all_stations else None

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
        
        transit_keywords = ["stop", "stops", "route", "timings", "timetable", "departure", "arrive", "arrival", "halt", "pass through", "runs from"]
        rules_keywords = ["cancel", "cancellation", "refund", "luggage", "penalty", "fine", "tte", "rule", "duty", "duties", "allowance", "charge", "charges", "fee"]
        
        if any(kw in query_lower for kw in transit_keywords):
            active_collections = [c for c in active_collections if c in ("train_routes", "stations", "trains")]
            logger.debug(f"[INTENT] Routing to transit collections: {active_collections}")
        elif any(kw in query_lower for kw in rules_keywords):
            active_collections = [c for c in active_collections if c in ("railway_rules", "references")]
            logger.debug(f"[INTENT] Routing to rules/references collections: {active_collections}")

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
                                res = col.get(where_document={"$contains": term})
                                if res["documents"]:
                                    logger.debug(f"[KEYWORD] Found {len(res['documents'])} matches in '{name}' containing '{term}'")
                                    for i in range(len(res["documents"])):
                                        doc = Document(
                                            page_content=res["documents"][i],
                                            metadata={**res["metadatas"][i], "collection": name, "relevance_score": 0.95}
                                        )
                                        keyword_docs.append(doc)
                            except Exception as exc:
                                logger.warning(f"[WARN] Keyword contains query failed for '{name}' (term='{term}'): {exc}")

        # --- Step 5: Semantic search across active collections ---
        all_results: list[tuple[Document, float]] = []

        for name in active_collections:
            store = self.vector_stores.get(name)
            if not store:
                continue
            try:
                results = store.similarity_search_with_relevance_scores(
                    search_query, k=PER_COLLECTION_K
                )
                for doc, score in results:
                    if score < 0.10:
                        continue  # ignore very low relevance scores (cut-off threshold)
                    doc.metadata["collection"] = name
                    doc.metadata["relevance_score"] = round(score, 4)
                    all_results.append((doc, score))
            except Exception as exc:
                logger.warning(f"[WARN] Error searching '{name}': {exc}")

        all_results.sort(key=lambda x: x[1], reverse=True)
        semantic_docs = [doc for doc, _ in all_results[: self.top_k]]

        # --- Step 6: Merge & Deduplicate ---
        all_candidate_docs = exact_docs + keyword_docs + semantic_docs
        deduped_docs = []
        seen_content = set()
        for doc in all_candidate_docs:
            snippet = doc.page_content[:80]
            if snippet not in seen_content:
                seen_content.add(snippet)
                deduped_docs.append(doc)

        # --- Step 6b: Two-Station Intersection Boost ---
        # When query has 2+ stations, boost docs containing ALL station terms to top.
        # This fixes BZA→SC queries where docs only matching one station dominate.
        if len(all_stations) >= 2:
            station_terms = []
            for canon, code in all_stations:
                station_terms.append((canon.lower(), code.lower()))

            def _doc_matches_all_stations(doc: Document) -> bool:
                content_lower = doc.page_content.lower()
                for canon, code in station_terms:
                    if canon not in content_lower and code not in content_lower:
                        return False
                return True

            both_match = [d for d in deduped_docs if _doc_matches_all_stations(d)]
            one_match  = [d for d in deduped_docs if not _doc_matches_all_stations(d)]
            if both_match:
                logger.debug(f"[INTERSECT] {len(both_match)} docs match ALL stations, {len(one_match)} match only one")
            deduped_docs = both_match + one_match

        # --- Step 7: Trim Route Schedules to avoid LLM context length overflow ---
        # Only trim when query is about a station (not a specific train number).
        # For train number queries, the user wants the full schedule.
        if all_stations and not train_number_detected:
            # Collect all station codes + canonical names across ALL resolved stations
            all_station_terms = set()
            for canon, code in all_stations:
                all_station_terms.add(canon.lower())
                all_station_terms.add(code.lower())

            for doc in deduped_docs:
                if doc.metadata.get("source_type") == "train_route":
                    content = doc.page_content
                    if "Schedule:" in content:
                        parts = content.split("Schedule:")
                        route_meta = parts[0]
                        schedule_part = parts[1]

                        stops_list = schedule_part.split("|")
                        trimmed_stops = []

                        for idx, stop in enumerate(stops_list):
                            stop_lower = stop.lower()
                            is_target = any(term in stop_lower for term in all_station_terms)
                            is_first = (idx == 0)
                            is_last = (idx == len(stops_list) - 1)

                            if is_target or is_first or is_last:
                                trimmed_stops.append(stop.strip())

                        doc.page_content = route_meta + "Schedule: " + " | ".join(trimmed_stops)

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
