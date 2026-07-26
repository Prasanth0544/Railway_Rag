"""
embed_routes.py — Dedicated script to embed ONLY train_routes collection.

Reads train_routes.csv and builds compact documents from stop_codes.
Document format:
  Train 22848 from LTT to VSKP. Stops (24): LTT > KYN > NK > ... > VSKP. Distance: 1651 km.

Avg document size: ~173 chars — well within Gemini's 2048 token limit.

Key rotation:
  Loads ALL GOOGLE_API_KEY* keys from .env.
  Automatically rotates to the next key when daily quota (PerDay) is exhausted.
  Stops only when ALL keys are exhausted.

Usage:
    .venv\Scripts\python scripts/embed_routes.py
"""

from __future__ import annotations

import os
import sys
import time
import json

import pandas as pd
from langchain_core.documents import Document

# Force UTF-8 on Windows console
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv  # type: ignore[import-untyped]
import chromadb  # type: ignore[import-untyped]
from langchain_chroma import Chroma  # type: ignore[import-untyped]
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore[import-untyped]

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# ── Config ────────────────────────────────────────────────────────────────────
COLLECTION   = "train_routes"
BATCH_SIZE   = 50
CHROMA_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
CHECKPOINT_F = os.path.join(CHROMA_DIR, ".embed_checkpoint.json")

# Source CSV — train_routes.csv (compact stop_codes, not decoded)
DATA_DIR = os.getenv(
    "DATA_COLLECTIONS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
)
CSV_PATH = os.path.join(DATA_DIR, "train_routes.csv")


# ── Load ALL API keys from .env ───────────────────────────────────────────────

def _load_api_keys() -> list[str]:
    """
    Load all GOOGLE_API_KEY* keys from .env in order:
      GOOGLE_API_KEY, GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, ...
    Skips commented-out or empty keys automatically.
    """
    keys: list[str] = []

    # Primary key (no suffix)
    v = os.getenv("GOOGLE_API_KEY", "").strip()
    if v:
        keys.append(v)

    # Numbered keys _1 … _99
    for i in range(1, 20):
        v = os.getenv(f"GOOGLE_API_KEY_{i}", "").strip()
        if v:
            keys.append(v)
        else:
            # Stop at first gap (or commented-out key)
            break

    return keys


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint() -> int:
    try:
        with open(CHECKPOINT_F) as f:
            return json.load(f).get(COLLECTION, 0)
    except Exception:
        return 0


def save_checkpoint(stored: int) -> None:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    data: dict = {}
    try:
        with open(CHECKPOINT_F) as f:
            data = json.load(f)
    except Exception:
        pass
    data[COLLECTION] = stored
    with open(CHECKPOINT_F, "w") as f:
        json.dump(data, f, indent=2)


# ── Document builder ──────────────────────────────────────────────────────────

def load_route_documents() -> list[Document]:
    """
    Read train_routes.csv and build one compact Document per train.
    Uses stop_codes (avg 136 chars) — NOT the full decoded stops JSON.
    """
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] File not found: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"[OK] Loaded train_routes.csv — {len(df):,} rows")

    documents = []
    skipped   = 0

    for _, row in df.iterrows():
        train_no    = str(row.get("train_number", "")).strip()
        src         = str(row.get("source_station", "")).strip()
        dst         = str(row.get("dest_station", "")).strip()
        total_dist  = str(row.get("total_distance_km", "")).strip()
        total_stops = str(row.get("total_stops", "")).strip()
        stop_codes  = str(row.get("stop_codes", "")).strip()

        if not train_no or train_no in ("nan", ""):
            skipped += 1
            continue

        # Parse stop_codes JSON array
        try:
            codes: list[str] = json.loads(stop_codes)
        except (json.JSONDecodeError, ValueError):
            codes = [c.strip().strip('"') for c in stop_codes.strip("[]").split(",") if c.strip()]

        if not codes:
            skipped += 1
            continue

        if not src or src == "nan": src = codes[0]
        if not dst or dst == "nan": dst = codes[-1]

        stop_seq = " > ".join(codes)
        text = f"Train {train_no} from {src} to {dst}."
        if total_stops and total_stops not in ("nan", "N/A"):
            text += f" Stops ({total_stops}): {stop_seq}."
        else:
            text += f" Stops: {stop_seq}."
        if total_dist and total_dist not in ("nan", "N/A"):
            text += f" Distance: {total_dist} km."

        documents.append(Document(
            page_content=text,
            metadata={
                "source_type"        : "train_route",
                "train_no"           : train_no,
                "source_station"     : src,
                "destination_station": dst,
                "total_stops"        : int(total_stops) if total_stops.isdigit() else len(codes),
                "total_distance_km"  : total_dist,
            }
        ))

    print(f"[OK] Built {len(documents):,} route documents (skipped {skipped})")
    return documents


# ── Embedding factory ─────────────────────────────────────────────────────────

def _make_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )


# ── Main embedding logic ──────────────────────────────────────────────────────

def main() -> None:
    api_keys = _load_api_keys()
    if not api_keys:
        print("[ERROR] No GOOGLE_API_KEY found in .env")
        sys.exit(1)

    print("=" * 60)
    print("train_routes Embedder (compact stop_codes + key rotation)")
    print("=" * 60)
    print(f"  Source : {CSV_PATH}")
    print(f"  Keys   : {len(api_keys)} API key(s) loaded")
    print(f"  Model  : gemini-embedding-001 (3072-dim)")
    print(f"  DB     : {CHROMA_DIR}")
    print()

    documents = load_route_documents()
    total     = len(documents)

    if documents:
        print(f"[SAMPLE] {documents[0].page_content[:200]}")
        print(f"[INFO]   Avg doc length: {sum(len(d.page_content) for d in documents) // len(documents)} chars\n")

    client      = chromadb.PersistentClient(path=CHROMA_DIR)
    resume_from = 0

    # ── Resume detection ──────────────────────────────────────────────────────
    try:
        existing = client.get_collection(COLLECTION).count()
    except Exception:
        existing = 0

    chk = load_checkpoint()
    if existing > 0 and chk > 0:
        resume_from = min(existing, chk)
        print(f"[RESUME] {existing:,} docs already stored — resuming from doc #{resume_from + 1}")
    else:
        try:
            client.delete_collection(COLLECTION)
            print(f"[DEL] Deleted old '{COLLECTION}' collection")
        except Exception:
            pass

    documents    = documents[resume_from:]
    stored       = resume_from
    vector_store = None

    # ── Key rotation state ────────────────────────────────────────────────────
    key_index      = 0
    exhausted_keys: set[int] = set()
    embeddings     = _make_embeddings(api_keys[key_index])

    print(f"[START] Embedding {len(documents):,} remaining docs ({total:,} total)")
    print(f"[KEY]   Using key #{key_index + 1} of {len(api_keys)}\n")

    for batch_start in range(0, len(documents), BATCH_SIZE):
        batch     = documents[batch_start: batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, len(documents))

        while True:   # keep retrying until success or all keys exhausted
            try:
                if vector_store is None:
                    vector_store = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        collection_name=COLLECTION,
                        client=client,
                    )
                else:
                    vector_store.add_documents(batch)

                stored += len(batch)
                pct = stored / total * 100
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"\r   [{bar}] {stored:,}/{total:,} ({pct:.0f}%)", end="", flush=True)
                save_checkpoint(stored)

                # Small inter-batch delay — stay under 100 req/min
                if batch_end < len(documents):
                    time.sleep(1.5)
                break  # success — move to next batch

            except Exception as exc:
                err = str(exc)

                # ── Daily quota exhausted → rotate to next key ────────────────
                if "PerDay" in err:
                    exhausted_keys.add(key_index)
                    save_checkpoint(stored)
                    print(f"\n   [QUOTA] Key #{key_index + 1} daily quota exhausted "
                          f"({stored:,}/{total:,} done).")

                    # Find next fresh key
                    found = False
                    for _ in range(len(api_keys)):
                        key_index = (key_index + 1) % len(api_keys)
                        if key_index not in exhausted_keys:
                            found = True
                            break

                    if not found:
                        print(f"\n[STOP] All {len(api_keys)} API keys exhausted for today.")
                        print(f"       Progress saved: {stored:,}/{total:,} docs.")
                        print(f"       Run again tomorrow to continue.")
                        sys.exit(0)

                    embeddings = _make_embeddings(api_keys[key_index])
                    remaining  = len(api_keys) - len(exhausted_keys)
                    print(f"   [KEY]  Rotated to key #{key_index + 1} "
                          f"({remaining} fresh key(s) remaining)")
                    print(f"   [KEY]  Cooling down 65s before using new key...")
                    time.sleep(65)
                    continue  # retry with fresh key

                # ── Per-minute / generic 429 → wait and retry same key ────────
                if "429" in err or "RESOURCE_EXHAUSTED" in err or "PerMinute" in err:
                    print(f"\n   [RATE] Rate limit (key #{key_index + 1}) — waiting 65s...")
                    time.sleep(65)
                    continue

                # ── Network error → short retry ───────────────────────────────
                if "getaddrinfo" in err or "10065" in err or "Server disconnected" in err:
                    print(f"\n   [NET]  Network error — retrying in 10s...")
                    time.sleep(10)
                    continue

                # ── Unknown error → skip batch ────────────────────────────────
                print(f"\n   [SKIP] Unhandled error for batch {batch_start}-{batch_end}: {err[:100]}")
                break

    count = client.get_collection(COLLECTION).count() if vector_store else existing
    print(f"\n\n{'=' * 60}")
    print(f"[DONE] '{COLLECTION}': {count:,} / {total:,} documents stored")
    if count >= total:
        print(f"[✓] All route documents embedded successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
