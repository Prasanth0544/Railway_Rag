"""
embed_coaches.py — Embed train coach composition/layout data.

Source: coach_positions.csv
Strategy: One Document per train

Document format:
  Train 1664 — LHB Rake, 23 coaches: L > EOG > A1 > A2 > A3 > A4 > B1 > B2 >
  B3 > M1 > M2 > M3 > M4 > M5 > M6 > M7 > S1 > S2 > S3 > S4 > GS > GS > EOG.
  Reversal at: ET.

Key rotation: loads all GOOGLE_API_KEY* from .env, rotates on PerDay exhaustion.

Usage:
    .venv\Scripts\python scripts/embed_coaches.py
"""

from __future__ import annotations

import os
import sys
import time
import json

import pandas as pd
from langchain_core.documents import Document

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
COLLECTION   = "coach_positions"
BATCH_SIZE   = 50
CHROMA_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
CHECKPOINT_F = os.path.join(CHROMA_DIR, ".embed_checkpoint.json")

DATA_DIR = os.getenv(
    "DATA_COLLECTIONS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
)
CSV_PATH = os.path.join(DATA_DIR, "coach_positions.csv")


# ── Load all API keys ─────────────────────────────────────────────────────────

def _load_api_keys() -> list[str]:
    keys: list[str] = []
    v = os.getenv("GOOGLE_API_KEY", "").strip()
    if v:
        keys.append(v)
    for i in range(1, 20):
        v = os.getenv(f"GOOGLE_API_KEY_{i}", "").strip()
        if v:
            keys.append(v)
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

def load_coach_documents() -> list[Document]:
    """
    Build one Document per train from coach_positions.csv.
    Captures: train number, rake type, coach sequence, reversal stations.
    """
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] File not found: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"[OK] Loaded coach_positions.csv — {len(df):,} rows")

    documents = []
    skipped   = 0

    for _, row in df.iterrows():
        train_no      = str(row.get("train_number", "")).strip()
        rake_type     = str(row.get("rake_type", "")).strip()
        total_coaches = str(row.get("total_coaches", "")).strip()
        has_layout    = str(row.get("has_layout", "")).strip()
        coaches_raw   = str(row.get("coaches", "")).strip()
        reversal_raw  = str(row.get("reversal_stations", "")).strip()

        if not train_no or train_no in ("nan", ""):
            skipped += 1
            continue

        # Parse coaches array
        try:
            coaches: list[str] = json.loads(coaches_raw)
        except (json.JSONDecodeError, ValueError):
            coaches = [c.strip().strip('"') for c in coaches_raw.strip("[]").split(",") if c.strip()]

        if not coaches:
            skipped += 1
            continue

        # Parse reversal stations
        try:
            reversal: list[str] = json.loads(reversal_raw)
        except (json.JSONDecodeError, ValueError):
            reversal = []

        # Build text
        coach_seq = " > ".join(coaches)
        text = f"Train {train_no}"
        if rake_type and rake_type not in ("nan", ""):
            text += f" — {rake_type}"
        if total_coaches and total_coaches not in ("nan", ""):
            text += f", {total_coaches} coaches"
        text += f": {coach_seq}."

        if reversal:
            text += f" Reversal station(s): {', '.join(reversal)}."

        if has_layout.lower() == "true":
            text += " Coach layout available."

        documents.append(Document(
            page_content=text,
            metadata={
                "source_type"   : "coach_position",
                "train_no"      : train_no,
                "rake_type"     : rake_type if rake_type not in ("nan", "") else "",
                "total_coaches" : int(float(total_coaches)) if total_coaches.replace(".", "").isdigit() else 0,
                "has_layout"    : has_layout.lower() == "true",
                "reversal_stations": ",".join(reversal) if reversal else "",
            }
        ))

    print(f"[OK] Built {len(documents):,} coach documents (skipped {skipped})")
    return documents


# ── Embedding factory ─────────────────────────────────────────────────────────

def _make_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    api_keys = _load_api_keys()
    if not api_keys:
        print("[ERROR] No GOOGLE_API_KEY found in .env")
        sys.exit(1)

    print("=" * 60)
    print("coach_positions Embedder")
    print("=" * 60)
    print(f"  Source : {CSV_PATH}")
    print(f"  Keys   : {len(api_keys)} API key(s) loaded")
    print(f"  Model  : gemini-embedding-001 (3072-dim)")
    print(f"  DB     : {CHROMA_DIR}")
    print()

    documents = load_coach_documents()
    total     = len(documents)

    if not documents:
        print("[ERROR] No documents built.")
        sys.exit(1)

    print(f"\n[SAMPLE 1] {documents[0].page_content}")
    print(f"[SAMPLE 2] {documents[100].page_content}")
    print(f"[INFO]   Avg doc length: {sum(len(d.page_content) for d in documents)//len(documents)} chars")
    print()

    client      = chromadb.PersistentClient(path=CHROMA_DIR)
    resume_from = 0

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
        except Exception:
            pass

    documents    = documents[resume_from:]
    stored       = resume_from
    vector_store = None

    key_index      = 0
    exhausted_keys: set[int] = set()
    embeddings     = _make_embeddings(api_keys[key_index])

    print(f"[START] Embedding {len(documents):,} remaining docs ({total:,} total)")
    print(f"[KEY]   Using key #{key_index + 1} of {len(api_keys)}\n")

    for batch_start in range(0, len(documents), BATCH_SIZE):
        batch     = documents[batch_start: batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, len(documents))

        while True:
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
                if batch_end < len(documents):
                    time.sleep(1.5)
                break

            except Exception as exc:
                err = str(exc)

                if "PerDay" in err:
                    exhausted_keys.add(key_index)
                    save_checkpoint(stored)
                    print(f"\n   [QUOTA] Key #{key_index + 1} exhausted ({stored:,}/{total:,} done).")
                    found = False
                    for _ in range(len(api_keys)):
                        key_index = (key_index + 1) % len(api_keys)
                        if key_index not in exhausted_keys:
                            found = True
                            break
                    if not found:
                        print(f"\n[STOP] All {len(api_keys)} keys exhausted. Progress saved.")
                        sys.exit(0)
                    embeddings = _make_embeddings(api_keys[key_index])
                    remaining = len(api_keys) - len(exhausted_keys)
                    print(f"   [KEY]  Rotated to key #{key_index + 1} ({remaining} fresh remaining)")
                    time.sleep(65)
                    continue

                if "429" in err or "RESOURCE_EXHAUSTED" in err or "PerMinute" in err:
                    print(f"\n   [RATE] Rate limit — waiting 65s...")
                    time.sleep(65)
                    continue

                if "getaddrinfo" in err or "10065" in err or "Server disconnected" in err:
                    print(f"\n   [NET]  Network error — retrying in 10s...")
                    time.sleep(10)
                    continue

                print(f"\n   [SKIP] Error for batch {batch_start}-{batch_end}: {err[:100]}")
                break

    count = client.get_collection(COLLECTION).count() if vector_store else existing
    print(f"\n\n{'=' * 60}")
    print(f"[DONE] '{COLLECTION}': {count:,} / {total:,} documents stored")
    if count >= total:
        print(f"[OK] All coach documents embedded successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
