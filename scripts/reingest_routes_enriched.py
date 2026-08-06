"""
reingest_routes_enriched.py — Re-ingests train_routes with enriched content.

Reads train_routes.csv + train names from 'trains' ChromaDB collection,
builds enriched documents (name + running days in header), then re-indexes
the train_routes collection.

New format:
  "Train 17226 — Narasapur Amaravati Express (Daily). From UBL to NS. Stops (28): ..."

Follows embed_routes.py's proven pattern exactly:
  - PerDay  quota exhausted  → rotate to next key
  - PerMinute / 429          → wait 65s and retry same key
  - Network error            → wait 10s and retry
  - Count-based checkpoint   → safe resume across runs

Usage:
    .venv\\Scripts\\python scripts/reingest_routes_enriched.py
"""

from __future__ import annotations

import os
import sys
import json
import time
import re

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
import chromadb
import pandas as pd
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# ── Config ────────────────────────────────────────────────────────────────────

COLLECTION   = "train_routes"
BATCH_SIZE   = 50
CHROMA_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
CHECKPOINT_F = os.path.join(CHROMA_DIR, ".reingest_enriched_checkpoint.json")

DATA_DIR = os.getenv(
    "DATA_COLLECTIONS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
)
CSV_PATH = os.path.join(DATA_DIR, "train_routes.csv")


# ── Load ALL API keys — no break on gaps ─────────────────────────────────────

def _load_api_keys() -> list[str]:
    """
    Load all GOOGLE_API_KEY* keys from .env.
    Does NOT break on gaps — checks all _1 through _99.
    So GOOGLE_API_KEY + GOOGLE_API_KEY_2 + GOOGLE_API_KEY_3 all load correctly
    even if _1 is missing.
    """
    keys: list[str] = []

    v = os.getenv("GOOGLE_API_KEY", "").strip()
    if v:
        keys.append(v)

    for i in range(1, 100):   # scan _1 through _99, no break on gaps
        v = os.getenv(f"GOOGLE_API_KEY_{i}", "").strip()
        if v:
            keys.append(v)

    return keys


def _make_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )


# ── Checkpoint (count-based, same style as embed_routes.py) ──────────────────

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


# ── Train Name Lookup ─────────────────────────────────────────────────────────

def build_train_name_map(client: chromadb.PersistentClient) -> dict[str, tuple[str, str]]:
    """Returns dict: train_no -> (train_name, runs_on)"""
    print("[INFO] Loading train names from 'trains' collection...")
    try:
        col = client.get_collection("trains")
    except Exception as e:
        print(f"[WARN] trains collection not found: {e}")
        return {}

    result = col.get(limit=20000, include=["documents", "metadatas"])
    name_map: dict[str, tuple[str, str]] = {}

    for doc, meta in zip(result["documents"], result["metadatas"]):
        train_no = str(meta.get("train_no", "")).strip()
        if not train_no:
            continue
        name = ""
        name_m = re.search(r"\(([^)]+)\)", doc)
        if name_m:
            name = name_m.group(1).strip()
        runs_on = ""
        runs_m = re.search(r"Runs on:\s*([^.]+)", doc)
        if runs_m:
            runs_on = runs_m.group(1).strip()
        name_map[train_no] = (name or "", runs_on or "")

    print(f"[INFO] Loaded {len(name_map):,} train name entries")
    return name_map


# ── Build Enriched Docs from CSV ──────────────────────────────────────────────

def build_enriched_docs(name_map: dict[str, tuple[str, str]]) -> list[Document]:
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV not found: {CSV_PATH}")
        sys.exit(1)

    print(f"[INFO] Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"[INFO] Loaded {len(df):,} rows")

    documents: list[Document] = []
    skipped       = 0
    enriched_count = 0

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
        train_name, runs_on = name_map.get(train_no, ("", ""))

        name_part = f" — {train_name}" if train_name else ""
        days_part = f" ({runs_on})"    if runs_on   else ""

        text = f"Train {train_no}{name_part}{days_part}. From {src} to {dst}."
        if total_stops and total_stops not in ("nan", "N/A"):
            text += f" Stops ({total_stops}): {stop_seq}."
        else:
            text += f" Stops: {stop_seq}."
        if total_dist and total_dist not in ("nan", "N/A"):
            text += f" Distance: {total_dist} km."

        if train_name:
            enriched_count += 1

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

    print(f"[INFO] Built {len(documents):,} docs ({enriched_count:,} with names, {skipped} skipped)")
    return documents


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_keys = _load_api_keys()
    if not api_keys:
        print("[ERROR] No GOOGLE_API_KEY found in .env"); sys.exit(1)

    print("=" * 65)
    print("train_routes Enriched Re-ingestion (from CSV)")
    print("=" * 65)
    print(f"  CSV      : {CSV_PATH}")
    print(f"  ChromaDB : {CHROMA_DIR}")
    print(f"  Keys     : {len(api_keys)} API key(s)")
    print(f"  Batch    : {BATCH_SIZE}")
    print()

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Step 1: Build name lookup
    name_map = build_train_name_map(client)

    # Step 2: Build enriched docs from CSV
    documents = build_enriched_docs(name_map)
    total     = len(documents)

    if documents:
        print(f"[SAMPLE] {documents[0].page_content[:180]}")
        print(f"[INFO]   Avg doc: {sum(len(d.page_content) for d in documents) // len(documents)} chars\n")

    # Step 3: Resume detection — use live ChromaDB count as ground truth.
    # NEVER delete the collection if it already has docs.
    # (The old checkpoint file may be missing after a completed run, but
    #  the actual stored count in ChromaDB is always reliable.)
    resume_from = 0
    try:
        existing = client.get_collection(COLLECTION).count()
    except Exception:
        existing = 0

    if existing > 0:
        resume_from = existing
        print(f"[RESUME] {existing:,} docs already stored — resuming from doc #{resume_from + 1}")
    else:
        # Collection is empty or doesn't exist — safe to recreate
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
        print(f"[NEW] Starting fresh — collection is empty")

    documents    = documents[resume_from:]
    stored       = resume_from
    vector_store = None

    # Step 4: Key rotation state
    key_index      = 0
    exhausted_keys: set[int] = set()
    embeddings     = _make_embeddings(api_keys[key_index])

    print(f"[START] Embedding {len(documents):,} remaining docs ({total:,} total)")
    print(f"[KEY]   Using key #{key_index + 1} of {len(api_keys)}\n")

    for batch_start in range(0, len(documents), BATCH_SIZE):
        batch     = documents[batch_start: batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, len(documents))

        while True:   # retry until success or all keys exhausted
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

                # Stay under 100 req/min — 1.5s inter-batch delay
                if batch_end < len(documents):
                    time.sleep(1.5)
                break  # success — next batch

            except Exception as exc:
                err = str(exc)

                # ── Daily quota exhausted → rotate key ────────────────────────
                if "PerDay" in err:
                    exhausted_keys.add(key_index)
                    save_checkpoint(stored)
                    print(f"\n   [QUOTA] Key #{key_index + 1} daily quota exhausted "
                          f"({stored:,}/{total:,} done).")

                    found = False
                    for _ in range(len(api_keys)):
                        key_index = (key_index + 1) % len(api_keys)
                        if key_index not in exhausted_keys:
                            found = True
                            break

                    if not found:
                        print(f"\n[STOP] All {len(api_keys)} API keys exhausted for today.")
                        print(f"       Progress saved: {stored:,}/{total:,} docs.")
                        print(f"       Run again tomorrow to resume.")
                        sys.exit(0)

                    embeddings = _make_embeddings(api_keys[key_index])
                    remaining  = len(api_keys) - len(exhausted_keys)
                    print(f"   [KEY]  Rotated to key #{key_index + 1} "
                          f"({remaining} fresh key(s) remaining)")
                    print(f"   [KEY]  Cooling down 65s before using new key...")
                    time.sleep(65)
                    continue

                # ── Invalid / expired key (401) → rotate permanently ──────────
                if "UNAUTHENTICATED" in err or "401" in err or "API_KEY_INVALID" in err:
                    exhausted_keys.add(key_index)
                    print(f"\n   [INVALID] Key #{key_index + 1} is invalid/expired — rotating.")
                    found = False
                    for _ in range(len(api_keys)):
                        key_index = (key_index + 1) % len(api_keys)
                        if key_index not in exhausted_keys:
                            found = True
                            break
                    if not found:
                        print(f"\n[STOP] No valid API keys remaining.")
                        sys.exit(1)
                    embeddings = _make_embeddings(api_keys[key_index])
                    print(f"   [KEY]  Rotated to key #{key_index + 1}")
                    continue

                # ── Per-minute / generic 429 → wait 65s, retry same key ───────
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
                print(f"\n   [SKIP] Unhandled error batch {batch_start}: {err[:100]}")
                break

    try:
        count = client.get_collection(COLLECTION).count()
    except Exception:
        count = stored

    print(f"\n\n{'=' * 65}")
    print(f"[DONE] '{COLLECTION}': {count:,} / {total:,} documents stored")
    if count >= total:
        print(f"[✓] All route documents embedded successfully!")
        if os.path.exists(CHECKPOINT_F):
            os.remove(CHECKPOINT_F)
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
