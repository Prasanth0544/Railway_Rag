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
PER_COLLECTION_K = 5


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
        print("[CLOUD] Using Gemini embeddings (models/gemini-embedding-001)")
        return GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )

    # Local offline embeddings
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore[import-untyped]
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[import-untyped]

    print("[LOCAL] Using sentence-transformers/all-MiniLM-L6-v2 (offline, no rate limits)")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu", "local_files_only": False},
    )


def get_chroma_client() -> chromadb.ClientAPI:
    """Get persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_DB_DIR)


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
                print(f"[WARN] Could not load collection '{name}': {exc}")

        if self.vector_stores:
            print(f"[OK] Retriever ready — collections: {list(self.vector_stores.keys())}")
        else:
            print("[WARN] No ChromaDB collections found! Run: python scripts/create_embeddings.py")

    def _lookup_by_train_number(self, query: str) -> list[Document]:
        """
        If the query contains a 5-digit train number (e.g. 12727),
        do a direct ChromaDB metadata lookup and return the exact match.
        This bypasses semantic search for precise train number queries.
        """
        if "trains" not in self.vector_stores:
            return []

        numbers = re.findall(r"\b(\d{5})\b", query)
        if not numbers:
            return []

        exact_docs: list[Document] = []
        trains_col = self.client.get_collection("trains")

        for num in numbers:
            try:
                result = trains_col.get(
                    where={"train_no": num},
                    limit=1,
                    include=["documents", "metadatas"],
                )
                if result["documents"]:
                    doc = Document(
                        page_content=result["documents"][0],
                        metadata={**result["metadatas"][0], "collection": "trains", "relevance_score": 1.0},
                    )
                    exact_docs.append(doc)
                    print(f"[EXACT] Found train {num}: {result['metadatas'][0].get('train_name', '')}")
            except Exception as exc:
                print(f"[WARN] Train number lookup failed for {num}: {exc}")

        return exact_docs

    def retrieve(self, query: str) -> list[Document]:
        """
        Search all collections and return the top-k most relevant
        documents merged across all of them, sorted by score.

        Also does a direct metadata lookup for any 5-digit train numbers
        found in the query, prepending exact matches at the top.
        """
        if not self.vector_stores:
            return []

        # --- Step 1: Exact train number lookup (bypasses semantic search) ---
        exact_docs = self._lookup_by_train_number(query)

        # --- Step 2: Semantic search across all collections ---
        all_results: list[tuple[Document, float]] = []

        for name, store in self.vector_stores.items():
            try:
                results = store.similarity_search_with_relevance_scores(
                    query, k=PER_COLLECTION_K
                )
                for doc, score in results:
                    if score < 0.1:
                        continue  # skip irrelevant cross-collection noise
                    doc.metadata["collection"] = name
                    doc.metadata["relevance_score"] = round(score, 4)
                    all_results.append((doc, score))
            except Exception as exc:
                print(f"[WARN] Error searching '{name}': {exc}")

        all_results.sort(key=lambda x: x[1], reverse=True)
        semantic_docs = [doc for doc, _ in all_results[: self.top_k]]

        # --- Step 3: Merge — exact matches first, then semantic ---
        # Deduplicate: skip semantic results that duplicate exact matches
        exact_ids = {d.page_content[:80] for d in exact_docs}
        deduped_semantic = [d for d in semantic_docs if d.page_content[:80] not in exact_ids]

        return (exact_docs + deduped_semantic)[: self.top_k]

    def __call__(self, query: str) -> list[Document]:
        """Make retriever callable (for use in LCEL chains)."""
        return self.retrieve(query)


# ─────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────

def get_unified_retriever(top_k: int = 5) -> UnifiedRetriever:
    """Factory function to create a UnifiedRetriever instance."""
    return UnifiedRetriever(top_k=top_k)
