"""
embed_schedules.py — Embed train stop-level schedule data.

Source: train_routes.csv -> 'stops' column
Strategy: One Document per (train, stop) — atomic, searchable, small

Document format:
  Train 22848 stop 13/24 at Mahasamund (MSMD): arrives 19:33, departs 19:35,
  halt 2 min, 1177 km from origin, day 1. Zone: ECOR, Division: SBP.

Estimated docs: ~12,341 trains x avg 24 stops = ~296,000 documents
Avg doc size: ~120 chars — well within Gemini's 2048 token limit

Key rotation: loads all GOOGLE_API_KEY* from .env, rotates on PerDay exhaustion.

Usage:
    .venv\Scripts\python scripts/embed_schedules.py
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
COLLECTION   = "train_schedules"
BATCH_SIZE   = 50
CHROMA_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
CHECKPOINT_F = os.path.join(CHROMA_DIR, ".embed_checkpoint.json")

DATA_DIR = os.getenv(
    "DATA_COLLECTIONS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
)
CSV_PATH = os.path.join(DATA_DIR, "train_routes.csv")


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

def _fmt_time(t: str) -> str:
    """Convert '19.33' -> '19:33', pass through 'First'/'Last'."""
    if not t or str(t) in ("", "nan", "None", "First", "Last"):
        return str(t) if t else ""
    return str(t).replace(".", ":")


def load_schedule_documents() -> list[Document]:
    """
    Read train_routes.csv stops column and build one Document per (train, stop).
    Each document captures: train number, stop sequence, station name+code,
    arrival, departure, halt, distance, day, zone, division, platform.
    """
    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] File not found: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"[OK] Loaded train_routes.csv — {len(df):,} rows")

    documents = []
    skipped_trains = 0
    skipped_stops  = 0

    for _, row in df.iterrows():
        train_no    = str(row.get("train_number", "")).strip()
        total_stops = str(row.get("total_stops", "")).strip()
        stops_raw   = row.get("stops")

        if not train_no or train_no in ("nan", ""):
            skipped_trains += 1
            continue
        if pd.isna(stops_raw) or not stops_raw:
            skipped_trains += 1
            continue

        try:
            stops = json.loads(str(stops_raw))
        except (json.JSONDecodeError, ValueError):
            skipped_trains += 1
            continue

        total = len(stops)

        for stop in stops:
            seq         = stop.get("seq", "?")
            code        = stop.get("station_code", "?")
            name        = stop.get("station_name", "")
            arrival     = _fmt_time(stop.get("arrival",   ""))
            departure   = _fmt_time(stop.get("departure", ""))
            halt        = str(stop.get("halt_min",    "")).strip()
            platform    = str(stop.get("platform",    "")).strip()
            distance    = str(stop.get("distance_km", "")).strip()
            day         = str(stop.get("day",         "")).strip()
            zone        = str(stop.get("zone",        "")).strip()
            division    = str(stop.get("division",    "")).strip()

            # Build natural language text
            station_label = f"{name} ({code})" if name and name != code else code
            text = f"Train {train_no} stop {seq}/{total} at {station_label}"

            # Arrival / departure
            arr_str = arrival   if arrival   not in ("", "First", "Last", "nan") else ""
            dep_str = departure if departure not in ("", "First", "Last", "nan") else ""

            if arrival == "First":
                text += ": origin/source station"
            elif departure == "Last":
                text += ": destination/terminus station"
            else:
                timing_parts = []
                if arr_str: timing_parts.append(f"arrives {arr_str}")
                if dep_str: timing_parts.append(f"departs {dep_str}")
                if timing_parts:
                    text += ": " + ", ".join(timing_parts)

            if halt and halt not in ("0", "nan", ""):
                text += f", halt {halt} min"
            if platform and platform not in ("nan", ""):
                text += f", platform {platform}"
            if distance and distance not in ("0", "nan", ""):
                text += f", {distance} km from origin"
            if day and day not in ("nan", "1", ""):
                text += f", day {day}"
            if zone and zone not in ("nan", ""):
                text += f". Zone: {zone}"
                if division and division not in ("nan", ""):
                    text += f", Division: {division}"
            text += "."

            if not code or code == "?":
                skipped_stops += 1
                continue

            documents.append(Document(
                page_content=text,
                metadata={
                    "source_type"   : "train_schedule",
                    "train_no"      : train_no,
                    "station_code"  : code,
                    "station_name"  : name,
                    "seq"           : int(seq) if str(seq).isdigit() else 0,
                    "arrival"       : arrival,
                    "departure"     : departure,
                    "halt_min"      : halt,
                    "platform"      : platform,
                    "distance_km"   : distance,
                    "day"           : day,
                    "zone"          : zone,
                    "division"      : division,
                }
            ))

    total_docs = len(documents)
    print(f"[OK] Built {total_docs:,} schedule documents "
          f"(skipped {skipped_trains} trains, {skipped_stops} stops)")
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
    print("train_schedules Embedder (per-stop documents)")
    print("=" * 60)
    print(f"  Source : {CSV_PATH}")
    print(f"  Keys   : {len(api_keys)} API key(s) loaded")
    print(f"  Model  : gemini-embedding-001 (3072-dim)")
    print(f"  DB     : {CHROMA_DIR}")
    print()

    documents = load_schedule_documents()
    total     = len(documents)

    if not documents:
        print("[ERROR] No documents built. Check CSV path.")
        sys.exit(1)

    # Sample output
    print(f"\n[SAMPLE 1] {documents[0].page_content}")
    print(f"[SAMPLE 2] {documents[100].page_content}")
    print(f"[SAMPLE 3] {documents[500].page_content}")
    print(f"[INFO]   Avg doc length : {sum(len(d.page_content) for d in documents) // len(documents)} chars")
    print(f"[INFO]   Max doc length : {max(len(d.page_content) for d in documents)} chars")
    print()

    client      = chromadb.PersistentClient(path=CHROMA_DIR)
    resume_from = 0

    # Resume detection
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

    # Key rotation state
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

                # Daily quota -> rotate key
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
                        print(f"\n[STOP] All {len(api_keys)} keys exhausted for today.")
                        print(f"       Progress saved: {stored:,}/{total:,} docs.")
                        print(f"       Run again tomorrow to continue.")
                        sys.exit(0)

                    embeddings = _make_embeddings(api_keys[key_index])
                    remaining  = len(api_keys) - len(exhausted_keys)
                    print(f"   [KEY]  Rotated to key #{key_index + 1} ({remaining} fresh remaining)")
                    print(f"   [KEY]  Cooling down 65s...")
                    time.sleep(65)
                    continue

                # Per-minute / generic 429 -> wait same key
                if "429" in err or "RESOURCE_EXHAUSTED" in err or "PerMinute" in err:
                    print(f"\n   [RATE] Rate limit (key #{key_index + 1}) — waiting 65s...")
                    time.sleep(65)
                    continue

                # Network error -> short retry
                if "getaddrinfo" in err or "10065" in err or "Server disconnected" in err:
                    print(f"\n   [NET]  Network error — retrying in 10s...")
                    time.sleep(10)
                    continue

                # Unknown -> skip batch
                print(f"\n   [SKIP] Error for batch {batch_start}-{batch_end}: {err[:100]}")
                break

    count = client.get_collection(COLLECTION).count() if vector_store else existing
    print(f"\n\n{'=' * 60}")
    print(f"[DONE] '{COLLECTION}': {count:,} / {total:,} documents stored")
    if count >= total:
        print(f"[OK] All schedule documents embedded successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
