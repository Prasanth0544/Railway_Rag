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
import json
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

# Checkpoint file — tracks progress for pause/resume
CHECKPOINT_FILE = os.path.join(CHROMA_DB_DIR, ".embed_checkpoint.json")


def _save_checkpoint(collection: str, stored: int) -> None:
    """Save current progress so we can resume after a pause."""
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    data = {}
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE) as f:
                data = json.load(f)
        except Exception:
            pass
    data[collection] = stored
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _load_checkpoint(collection: str) -> int:
    """Return number of docs already stored for this collection (0 if no checkpoint)."""
    if not os.path.exists(CHECKPOINT_FILE):
        return 0
    try:
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        return data.get(collection, 0)
    except Exception:
        return 0


def _clear_checkpoint(collection: str) -> None:
    """Remove checkpoint entry for a finished collection."""
    if not os.path.exists(CHECKPOINT_FILE):
        return
    try:
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
        data.pop(collection, None)
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


# Batch size — auto-selected based on embedding provider
_use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
BATCH_SIZE = 256 if _use_local else 50   # Gemini: 100 req/min → 50 docs/batch with sleep


# ─────────────────────────────────────────────
# EMBEDDINGS SETUP  (multi-key rotation)
# ─────────────────────────────────────────────

def _load_api_keys() -> list[str]:
    """
    Load all Gemini API keys from .env.
    Reads GOOGLE_API_KEY, GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, … GOOGLE_API_KEY_N.
    Returns a deduplicated ordered list.
    """
    keys = []
    # Primary key
    k = os.getenv("GOOGLE_API_KEY", "").strip()
    if k and k != "your-gemini-api-key-here":
        keys.append(k)
    # Numbered extras
    for i in range(1, 20):
        k = os.getenv(f"GOOGLE_API_KEY_{i}", "").strip()
        if k and k != "your-gemini-api-key-here":
            if k not in keys:
                keys.append(k)
    return keys


# Module-level mutable key state (shared across all create_collection calls)
_api_keys: list[str] = []
_key_index: int = 0
_exhausted_keys: set[int] = set()   # indices of keys that hit daily quota today


def _make_gemini_embeddings(api_key: str):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore[import-untyped]
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )


def get_embeddings():
    """
    Return the embedding model based on USE_LOCAL_EMBEDDINGS in .env.

    USE_LOCAL_EMBEDDINGS=true  → sentence-transformers (offline, no limits)
    USE_LOCAL_EMBEDDINGS=false → Gemini embedding API with multi-key rotation
    """
    global _api_keys, _key_index
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

    # Cloud: Gemini with key rotation
    _api_keys = _load_api_keys()
    if not _api_keys:
        print("[ERROR] No GOOGLE_API_KEY found. Set it in .env or use USE_LOCAL_EMBEDDINGS=true")
        sys.exit(1)
    _key_index = 0
    print(f"   [CLOUD] Using Gemini gemini-embedding-001 — {len(_api_keys)} API key(s) loaded")
    return _make_gemini_embeddings(_api_keys[0])


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

    # ── Pause/Resume: check how many docs already stored ──────────────────
    checkpoint_stored = _load_checkpoint(collection_name)
    try:
        existing_count = client.get_collection(collection_name).count()
    except Exception:
        existing_count = 0

    resume_from = 0
    if existing_count > 0 and checkpoint_stored > 0:
        resume_from = min(existing_count, checkpoint_stored)
        print(f"   [RESUME] Found {existing_count:,} docs already stored — resuming from doc #{resume_from + 1}")
        overwrite = False   # don't delete existing progress!
    elif overwrite:
        _clear_checkpoint(collection_name)

    if overwrite and resume_from == 0:
        try:
            client.delete_collection(collection_name)
            print(f"   [DEL]  Deleted existing '{collection_name}'")
        except Exception:
            pass

    global _key_index, _exhausted_keys
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
    total = len(documents)
    stored = resume_from
    vector_store = None

    # Skip already-embedded documents
    documents = documents[resume_from:]

    consecutive_rate_hits = 0   # track rate-limit hits without progress on same key

    for batch_start in range(0, len(documents), BATCH_SIZE):
        batch = documents[batch_start: batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, len(documents))

        max_retries = 20   # more retries to allow key rotation
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

                # Save checkpoint after every successful batch
                _save_checkpoint(collection_name, stored)
                consecutive_rate_hits = 0   # reset on success

                # Small delay for Gemini cloud (per-minute rate limits)
                # 50 docs/batch → ~40 batches/min at 1.5s delay → well under 100 req/min
                if not use_local and batch_end < total:
                    time.sleep(1.5)
                break

            except Exception as exc:
                err = str(exc)

                # ── Classify the error precisely ──────────────────────────────
                # Per-DAY  quota: quotaId contains "PerDay", limit is 1000
                # Per-MIN  quota: quotaId contains "PerMinute", limit is 100
                # Generic 429: no quotaId field — treat as per-minute (rate limit)
                # We check PerDay first; anything else with 429/RESOURCE_EXHAUSTED
                # is treated as a per-minute rate limit to avoid wasting retries.
                is_daily_quota = "PerDay" in err
                is_per_minute  = (not is_daily_quota) and ("PerMinute" in err
                                  or "429" in err
                                  or "RESOURCE_EXHAUSTED" in err)
                is_rate_limit  = is_daily_quota or is_per_minute

                # ── Case 1: Per-MINUTE / generic rate limit ───────────────────
                # Stay on same key and wait — BUT if we hit rate limit 3+ times
                # in a row with no progress, the key is likely truly exhausted
                # (Google just isn't returning PerDay in the error). Rotate it.
                if is_per_minute:
                    consecutive_rate_hits += 1
                    if consecutive_rate_hits >= 3 and not use_local and len(_api_keys) > 1:
                        # Force-rotate: this key is stuck, likely daily-exhausted
                        _exhausted_keys.add(_key_index)
                        found_fresh = False
                        for _ in range(len(_api_keys)):
                            _key_index = (_key_index + 1) % len(_api_keys)
                            if _key_index not in _exhausted_keys:
                                found_fresh = True
                                break
                        if found_fresh:
                            new_key = _api_keys[_key_index]
                            print(f"\n   [KEY] Key stuck on rate limit — force-rotating to key #{_key_index + 1} "
                                  f"({len(_exhausted_keys)} exhausted, {len(_api_keys) - len(_exhausted_keys)} fresh)")
                            embeddings = _make_gemini_embeddings(new_key)  # type: ignore[assignment]
                            consecutive_rate_hits = 0
                            time.sleep(65)
                            continue
                        else:
                            _save_checkpoint(collection_name, stored)
                            print(f"\n   [STOP] All {len(_api_keys)} API keys exhausted for today.")
                            print(f"          Progress saved: {stored:,}/{total:,} docs.")
                            print(f"          Run this script again tomorrow to resume.")
                            sys.exit(1)
                    print(f"\n   [RATE] Rate limit hit #{consecutive_rate_hits} (key #{_key_index + 1}) — waiting 65s...")
                    time.sleep(65)
                    continue   # retry same key, don't count as failed attempt

                # ── Case 2: Per-DAY quota exhausted (1000 req/day) ────────────
                # Mark this key as truly exhausted and rotate to a fresh one.
                if is_daily_quota and not use_local and len(_api_keys) > 1:
                    _exhausted_keys.add(_key_index)   # mark current key as exhausted

                    # Find next non-exhausted key
                    found_fresh = False
                    for _ in range(len(_api_keys)):
                        _key_index = (_key_index + 1) % len(_api_keys)
                        if _key_index not in _exhausted_keys:
                            found_fresh = True
                            break

                    if not found_fresh:
                        _save_checkpoint(collection_name, stored)
                        print(f"\n   [STOP] All {len(_api_keys)} API keys exhausted for today.")
                        print(f"          Progress saved: {stored:,}/{total:,} docs.")
                        print(f"          Run this script again tomorrow to resume.")
                        sys.exit(1)

                    new_key = _api_keys[_key_index]
                    print(f"\n   [KEY] Daily quota exhausted — switching to key #{_key_index + 1} "
                          f"({len(_exhausted_keys)} exhausted, {len(_api_keys) - len(_exhausted_keys)} fresh)")
                    embeddings = _make_gemini_embeddings(new_key)  # type: ignore[assignment]
                    # Wait 65s so the new key's per-minute window is clean
                    print(f"   [KEY] Cooling down 65s before using new key...")
                    time.sleep(65)
                    continue   # retry with fresh key (don't count as failed attempt)

                # ── Case 3: Other errors (network, 503, etc.) ────────────────
                sleep_time = 62 if is_rate_limit else retry_delay
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
