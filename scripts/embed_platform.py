"""
embed_platform.py — Embed platform direction data per station.

Source: platform_direction.csv
Strategy: Group all platforms per station -> one Document per station

Document format:
  Station ABH platform info: Platform 1 (Both directions) between ULNR and BUD.
  Platform 2 (Right) between BUD and ULNR. Platform 3 (Left) between BUD and ULNR.

Usage:
    .venv\Scripts\python scripts/embed_platform.py
"""

from __future__ import annotations

import os, sys, time, json
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

COLLECTION   = "platform_info"
BATCH_SIZE   = 50
CHROMA_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
CHECKPOINT_F = os.path.join(CHROMA_DIR, ".embed_checkpoint.json")
DATA_DIR     = os.getenv("DATA_COLLECTIONS_DIR",
                         os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
CSV_PATH     = os.path.join(DATA_DIR, "platform_direction.csv")


def _load_api_keys() -> list[str]:
    keys: list[str] = []
    v = os.getenv("GOOGLE_API_KEY", "").strip()
    if v: keys.append(v)
    for i in range(1, 20):
        v = os.getenv(f"GOOGLE_API_KEY_{i}", "").strip()
        if v: keys.append(v)
    return keys


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
        with open(CHECKPOINT_F) as f: data = json.load(f)
    except Exception: pass
    data[COLLECTION] = stored
    with open(CHECKPOINT_F, "w") as f: json.dump(data, f, indent=2)


def load_platform_documents() -> list[Document]:
    """Group all platforms per station -> one Document per station."""
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] File not found: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"[OK] Loaded platform_direction.csv — {len(df):,} rows")

    # Group by station_code
    grouped = df.groupby("station_code")
    documents = []

    for stn_code, grp in grouped:
        platforms = []
        for _, row in grp.iterrows():
            pf_num   = str(row.get("platform_number", "")).strip()
            direction= str(row.get("direction", "")).strip()
            prev_stn = str(row.get("prev_station", "")).strip()
            next_stn = str(row.get("next_station", "")).strip()

            pf_text = f"Platform {pf_num} ({direction} direction)"
            if prev_stn and prev_stn not in ("nan", ""):
                pf_text += f" coming from {prev_stn}"
            if next_stn and next_stn not in ("nan", ""):
                pf_text += f" going towards {next_stn}"
            platforms.append(pf_text)

        text = f"Station {stn_code} platform information: " + ". ".join(platforms) + "."

        documents.append(Document(
            page_content=text,
            metadata={
                "source_type" : "platform_info",
                "station_code": str(stn_code),
                "num_platforms": len(platforms),
            }
        ))

    print(f"[OK] Built {len(documents):,} platform documents ({len(df)} platform rows)")
    return documents


def _make_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)


def main() -> None:
    api_keys = _load_api_keys()
    if not api_keys:
        print("[ERROR] No GOOGLE_API_KEY found in .env")
        sys.exit(1)

    print("=" * 60)
    print("platform_info Embedder")
    print("=" * 60)
    print(f"  Source : {CSV_PATH}")
    print(f"  Keys   : {len(api_keys)} API key(s) loaded")
    print()

    documents = load_platform_documents()
    total     = len(documents)

    print(f"\n[SAMPLE 1] {documents[0].page_content}")
    print(f"[SAMPLE 2] {documents[5].page_content}")
    print(f"[INFO]   Avg length: {sum(len(d.page_content) for d in documents)//len(documents)} chars\n")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    resume_from = 0
    try:
        existing = client.get_collection(COLLECTION).count()
    except Exception:
        existing = 0
    chk = load_checkpoint()
    if existing > 0 and chk > 0:
        resume_from = min(existing, chk)
        print(f"[RESUME] {existing:,} docs — resuming from #{resume_from + 1}")
    else:
        try: client.delete_collection(COLLECTION)
        except Exception: pass

    documents    = documents[resume_from:]
    stored       = resume_from
    vector_store = None
    key_index    = 0
    exhausted_keys: set[int] = set()
    embeddings   = _make_embeddings(api_keys[key_index])

    print(f"[START] Embedding {len(documents):,} remaining docs ({total:,} total)")
    print(f"[KEY]   Using key #{key_index + 1} of {len(api_keys)}\n")

    for batch_start in range(0, len(documents), BATCH_SIZE):
        batch     = documents[batch_start: batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, len(documents))

        while True:
            try:
                if vector_store is None:
                    vector_store = Chroma.from_documents(documents=batch, embedding=embeddings,
                                                         collection_name=COLLECTION, client=client)
                else:
                    vector_store.add_documents(batch)

                stored += len(batch)
                pct = stored / total * 100
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"\r   [{bar}] {stored:,}/{total:,} ({pct:.0f}%)", end="", flush=True)
                save_checkpoint(stored)
                if batch_end < len(documents): time.sleep(1.5)
                break

            except Exception as exc:
                err = str(exc)
                if "PerDay" in err:
                    exhausted_keys.add(key_index)
                    save_checkpoint(stored)
                    print(f"\n   [QUOTA] Key #{key_index + 1} exhausted.")
                    found = False
                    for _ in range(len(api_keys)):
                        key_index = (key_index + 1) % len(api_keys)
                        if key_index not in exhausted_keys: found = True; break
                    if not found:
                        print(f"\n[STOP] All keys exhausted. Progress saved."); sys.exit(0)
                    embeddings = _make_embeddings(api_keys[key_index])
                    print(f"   [KEY]  Rotated to key #{key_index + 1}")
                    time.sleep(65); continue
                if "429" in err or "RESOURCE_EXHAUSTED" in err or "PerMinute" in err:
                    print(f"\n   [RATE] Rate limit — waiting 65s..."); time.sleep(65); continue
                if "getaddrinfo" in err or "10065" in err or "Server disconnected" in err:
                    print(f"\n   [NET]  Network error — retrying in 10s..."); time.sleep(10); continue
                print(f"\n   [SKIP] Error: {err[:100]}"); break

    count = client.get_collection(COLLECTION).count() if vector_store else existing
    print(f"\n\n{'='*60}\n[DONE] '{COLLECTION}': {count:,}/{total:,} docs stored\n{'='*60}")


if __name__ == "__main__":
    main()
