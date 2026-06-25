"""
create_embeddings.py — Generate embeddings and store in ChromaDB

Reads all data sources, generates embeddings in batches,
and stores them in a persistent ChromaDB vector store.

Collections created:
  - railway_rules  — 183 railway rules
  - trains         — 12,813 train info documents
  - stations       — 11,354 station documents
  - references     — ticket classes, service tax
  - train_routes   — route/schedule documents (skipped by default)

Usage:
  python scripts/create_embeddings.py               # all collections
  python scripts/create_embeddings.py --skip-routes  # skip routes (faster)
  python scripts/create_embeddings.py --rules-only   # only rules
  python scripts/create_embeddings.py --trains-only  # trains + stations only
"""

from __future__ import annotations

import os
import sys
import time
import argparse

# Force UTF-8 encoding for Windows console (prevents UnicodeEncodeError with emojis)
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

# Add project root to path so scripts.preprocess can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv  # type: ignore[import-untyped]
import chromadb  # type: ignore[import-untyped]
from langchain_chroma import Chroma  # type: ignore[import-untyped]

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# ChromaDB storage path
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")

# Batch size:
#   - sentence-transformers (local): 256 — no rate limit, fast
#   - Gemini cloud API:              50  — 100 req/min limit
BATCH_SIZE = 256


# ─────────────────────────────────────────────
# EMBEDDINGS SETUP
# ─────────────────────────────────────────────

def get_embeddings():
    """
    Return the embedding model based on USE_LOCAL_EMBEDDINGS in .env.

    USE_LOCAL_EMBEDDINGS=true  → sentence-transformers (offline, no limits)
    USE_LOCAL_EMBEDDINGS=false → Gemini embedding API (cloud, free tier)
    """
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"

    if use_local:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore[import-untyped]
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[import-untyped]

        print("   [LOCAL] Using sentence-transformers/all-MiniLM-L6-v2 — 100% offline, no rate limits")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": 256},
        )

    # Cloud: Gemini
    from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore[import-untyped]
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key or api_key == "your-gemini-api-key-here":
        print("[ERROR] GOOGLE_API_KEY not set. Either set it in .env or use USE_LOCAL_EMBEDDINGS=true")
        sys.exit(1)
    print("   [CLOUD] Using Gemini gemini-embedding-001")
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )


# ─────────────────────────────────────────────
# COLLECTION BUILDER (with batching + retry)
# ─────────────────────────────────────────────

def create_collection(
    client: chromadb.ClientAPI,
    embeddings: object,
    collection_name: str,
    documents: list,
    overwrite: bool = True,
) -> None:
    """
    Create (or overwrite) a ChromaDB collection from a list of Documents.
    Uses batched inserts with auto-retry for rate limit errors.
    """
    if not documents:
        print(f"   [SKIP] No documents for '{collection_name}'")
        return

    print(f"\n[EMBED] Collection: '{collection_name}' ({len(documents):,} docs)")

    if overwrite:
        try:
            client.delete_collection(collection_name)
            print(f"   [DEL]  Deleted existing '{collection_name}'")
        except Exception:
            pass

    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
    total = len(documents)
    stored = 0
    vector_store = None

    for batch_start in range(0, total, BATCH_SIZE):
        batch = documents[batch_start: batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, total)

        max_retries = 5
        retry_delay = 5

        for attempt in range(1, max_retries + 1):
            try:
                if vector_store is None:
                    vector_store = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        collection_name=collection_name,
                        client=client,
                    )
                else:
                    vector_store.add_documents(batch)

                stored += len(batch)
                pct = stored / total * 100
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"\r   [{bar}] {stored:,}/{total:,} ({pct:.0f}%)", end="", flush=True)

                # Delay only for Gemini cloud (rate limits)
                if not use_local and batch_end < total:
                    time.sleep(1.0)
                break

            except Exception as exc:
                err = str(exc)
                is_rate_limit = "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower()
                sleep_time = 60 if is_rate_limit else retry_delay

                print(f"\n   [WARN] Attempt {attempt}/{max_retries} failed for batch {batch_start}-{batch_end}: {exc}")
                if attempt < max_retries:
                    print(f"   [WAIT] Retrying after {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    print(f"   [FAIL] All {max_retries} attempts failed — skipping batch.")

    count = client.get_collection(collection_name).count() if vector_store else 0
    print(f"\n   [DONE] '{collection_name}': {count:,} documents stored")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Railway RAG — Embedding Pipeline")
    parser.add_argument("--skip-routes",  action="store_true", help="Skip train route documents (faster)")
    parser.add_argument("--rules-only",   action="store_true", help="Only embed railway_rules collection")
    parser.add_argument("--trains-only",  action="store_true", help="Only embed trains + stations")
    parser.add_argument("--routes-only",  action="store_true", help="Only embed train_routes collection")
    args = parser.parse_args()

    print("Railway RAG Assistant — Embedding Pipeline")
    print("=" * 55)

    from scripts.preprocess import (  # type: ignore[import-untyped]
        load_rules_documents,
        load_train_documents,
        load_station_documents,
        load_train_route_documents,
        load_reference_documents,
    )

    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
    mode = "Offline (sentence-transformers)" if use_local else "Cloud (Gemini)"
    print(f"\n[INIT] Embeddings: {mode}")
    embeddings = get_embeddings()

    print(f"\n[DB]   ChromaDB: {CHROMA_DB_DIR}")
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    start_time = time.time()

    if args.rules_only:
        create_collection(client, embeddings, "railway_rules", load_rules_documents())

    elif args.trains_only:
        create_collection(client, embeddings, "trains",   load_train_documents())
        create_collection(client, embeddings, "stations", load_station_documents())

    elif args.routes_only:
        create_collection(client, embeddings, "train_routes", load_train_route_documents(max_trains=None))

    else:
        create_collection(client, embeddings, "railway_rules", load_rules_documents())
        create_collection(client, embeddings, "trains",        load_train_documents())
        create_collection(client, embeddings, "stations",      load_station_documents())
        create_collection(client, embeddings, "references",    load_reference_documents())

        if not args.skip_routes:
            create_collection(client, embeddings, "train_routes", load_train_route_documents(max_trains=None))
        else:
            print("\n[SKIP] Skipping train routes (--skip-routes)")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 55}")
    print(f"[OK]   Embedding pipeline complete! ({elapsed:.1f}s)")
    print(f"[DB]   Stored in: {CHROMA_DB_DIR}")
    print(f"\n[INFO] Collections in ChromaDB:")
    for col in client.list_collections():
        count = client.get_collection(col.name).count()
        print(f"   - {col.name}: {count:,} documents")
    print(f"\n[NEXT] Start the API server:")
    print(f"   .venv\\Scripts\\python -m uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
