"""
embed_routes.py — Embed train_routes with FULL SCHEDULE TIMES.

Reads train_route_decoded.csv (arr/dep/halt per stop) + train_routes.csv (fallback).
Document format:
  Train 12727 — Hyderabad Godavari SF Express (Daily). From VSKP to HYB. 21 stops, 707.0 km.
  VSKP dep 17:20 | DVD arr 17:45 dep 17:47 (2min) | BZA arr 23:15 dep 23:30 (15min) | HYB arr 06:15 [last].

Key rotation:
  Loads ALL GOOGLE_API_KEY* keys from .env.
  Automatically rotates to the next key when daily quota (PerDay) is exhausted.
  Stops only when ALL keys are exhausted.

Usage:
    .venv\\Scripts\\python scripts/embed_routes.py
"""

from __future__ import annotations

import os
import sys
import time
import json
import re

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
BATCH_SIZE   = 10          # 10 docs per batch = 10 API calls, then sleep
INTER_SLEEP  = 7.0     # 7s between batches → ~85 docs/min (under 100/min limit)
CHROMA_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
CHECKPOINT_F = os.path.join(CHROMA_DIR, ".embed_checkpoint.json")

DATA_DIR = os.getenv(
    "DATA_COLLECTIONS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
)
# Primary: has full arr/dep/halt times per stop
CSV_DECODED = os.path.join(DATA_DIR, "train_route_decoded.csv")
# Fallback: has stop codes only (no times) — used for trains not in decoded CSV
CSV_SIMPLE  = os.path.join(DATA_DIR, "train_routes.csv")


# ── Load ALL API keys from .env ───────────────────────────────────────────────

def _load_api_keys() -> list[str]:
    """
    Load all GOOGLE_API_KEY* keys from .env in order:
      GOOGLE_API_KEY, GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, ...
    """
    keys: list[str] = []

    # Primary key (no suffix)
    v = os.getenv("GOOGLE_API_KEY", "").strip()
    if v:
        keys.append(v)

    # Numbered keys _1 … _99
    for i in range(1, 100):
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
        tno = str(meta.get("train_no", "")).strip()
        if not tno:
            continue
        nm = re.search(r"\(([^)]+)\)", doc)
        ro = re.search(r"Runs on:\s*([^.]+)", doc)
        name_map[tno] = (nm.group(1).strip() if nm else "", ro.group(1).strip() if ro else "")
    print(f"[INFO] Loaded {len(name_map):,} train name entries")
    return name_map


# ── Stop Formatter ────────────────────────────────────────────────────────────

def _fmt_stop(stop: dict, prev_day: int) -> tuple[str, int]:
    """Format one stop into a readable segment. Returns (text, current_day)."""
    code = stop.get("station_code", "???")
    arr  = stop.get("arrival",   "")
    dep  = stop.get("departure", "")
    halt = stop.get("halt_minutes", 0)
    day  = stop.get("day", 1)
    dm   = f"[Day {day}] " if day > prev_day else ""

    if arr in ("First", "", None) and dep not in ("Last", "", None):
        seg = f"{dm}{code} dep {dep}"
    elif dep in ("Last", "", None) and arr not in ("First", "", None):
        seg = f"{dm}{code} arr {arr} [last]"
    elif arr in ("First", "", None) and dep in ("Last", "", None):
        seg = f"{dm}{code}"
    else:
        hs  = f" ({halt}min)" if halt and halt > 0 else ""
        seg = f"{dm}{code} arr {arr} dep {dep}{hs}"
    return seg, day


# ── Document builder ──────────────────────────────────────────────────────────

def build_docs(name_map: dict[str, tuple[str, str]]) -> list[Document]:
    documents:   list[Document] = []
    seen_trains: set[str]       = set()
    with_times    = 0
    without_times = 0
    skipped       = 0

    # ── Pass 1: train_route_decoded.csv  (has full schedule times) ────────────
    if not os.path.exists(CSV_DECODED):
        print(f"[WARN] {CSV_DECODED} not found — will use simple CSV only")
    else:
        print(f"[INFO] Pass 1 — reading {CSV_DECODED} ...")
        df1 = pd.read_csv(CSV_DECODED, low_memory=False)
        print(f"[INFO] {len(df1):,} rows loaded")

        for _, row in df1.iterrows():
            tno        = str(row.get("train_number",      "")).strip()
            src        = str(row.get("source_station",    "")).strip()
            dst        = str(row.get("dest_station",      "")).strip()
            total_dist = str(row.get("total_distance_km", "")).strip()
            total_stp  = str(row.get("total_stops",       "")).strip()
            stops_json = str(row.get("stops",             "[]")).strip()

            if not tno or tno == "nan":
                skipped += 1
                continue

            try:
                stops: list[dict] = json.loads(stops_json)
            except Exception:
                stops = []

            stopping = [s for s in stops if s.get("is_stopping", True)]
            if not stopping:
                skipped += 1
                continue

            if not src or src == "nan": src = stopping[0].get("station_code", "?")
            if not dst or dst == "nan": dst = stopping[-1].get("station_code", "?")

            name, runs = name_map.get(tno, ("", ""))
            n    = int(total_stp) if total_stp.replace(".", "").isdigit() else len(stopping)
            dist = total_dist if total_dist and total_dist not in ("nan", "N/A") else ""

            header = (f"Train {tno}"
                      + (f" — {name}" if name else "")
                      + (f" ({runs})"  if runs  else "")
                      + f". From {src} to {dst}."
                      + (f" {n} stops" if n else "")
                      + (f", {dist} km" if dist else "")
                      + ".")

            parts: list[str] = []
            prev_day = 1
            has_t    = False
            for s in stopping:
                seg, prev_day = _fmt_stop(s, prev_day)
                parts.append(seg)
                if s.get("arrival") not in ("First", "", None) or s.get("departure") not in ("Last", "", None):
                    has_t = True

            text = f"{header}\n{' | '.join(parts)}."

            if has_t: with_times += 1
            else:     without_times += 1

            seen_trains.add(tno)
            documents.append(Document(
                page_content=text,
                metadata={
                    "source_type"        : "train_route",
                    "train_no"           : tno,
                    "source_station"     : src,
                    "destination_station": dst,
                    "total_stops"        : n,
                    "total_distance_km"  : dist,
                    "has_times"          : has_t,
                }
            ))

        print(f"[INFO] Pass 1 done: {len(documents):,} docs  "
              f"({with_times:,} with times ✅, {without_times} stop-codes only)")

    # ── Pass 2: train_routes.csv  (stop codes only, fills missing trains) ─────
    if os.path.exists(CSV_SIMPLE):
        print(f"[INFO] Pass 2 — reading {CSV_SIMPLE} (filling remaining trains) ...")
        df2   = pd.read_csv(CSV_SIMPLE, low_memory=False)
        extra = 0

        for _, row in df2.iterrows():
            tno = str(row.get("train_number", "")).strip()
            if not tno or tno == "nan" or tno in seen_trains:
                continue

            src        = str(row.get("source_station",    "")).strip()
            dst        = str(row.get("dest_station",      "")).strip()
            total_dist = str(row.get("total_distance_km", "")).strip()
            total_stp  = str(row.get("total_stops",       "")).strip()
            stop_codes = str(row.get("stop_codes",        "[]")).strip()

            try:
                codes = json.loads(stop_codes)
            except Exception:
                codes = [c.strip().strip('"') for c in stop_codes.strip("[]").split(",") if c.strip()]
            if not codes:
                continue

            if not src or src == "nan": src = codes[0]
            if not dst or dst == "nan": dst = codes[-1]

            name, runs = name_map.get(tno, ("", ""))
            text = (f"Train {tno}"
                    + (f" — {name}" if name else "")
                    + (f" ({runs})"  if runs  else "")
                    + f". From {src} to {dst}. Stops ({total_stp}): {' > '.join(codes)}."
                    + (f" Distance: {total_dist} km." if total_dist and total_dist != "nan" else ""))

            seen_trains.add(tno)
            documents.append(Document(
                page_content=text,
                metadata={
                    "source_type"        : "train_route",
                    "train_no"           : tno,
                    "source_station"     : src,
                    "destination_station": dst,
                    "total_stops"        : int(total_stp) if total_stp.isdigit() else len(codes),
                    "total_distance_km"  : total_dist,
                    "has_times"          : False,
                }
            ))
            extra += 1

        print(f"[INFO] Pass 2 done: +{extra:,} additional trains (stop-codes only)")

    print(f"\n[SUMMARY] {len(documents):,} total docs | {len(seen_trains):,} unique trains")
    print(f"          {with_times:,} with full times ✅ | {len(documents)-with_times:,} stop-codes only")

    if documents:
        sample = documents[0].page_content
        print(f"\n[SAMPLE]\n{sample[:300]}")
        avg = sum(len(d.page_content) for d in documents) // len(documents)
        print(f"\n[INFO] Avg doc length: {avg} chars\n")

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

    print("=" * 65)
    print("train_routes Embedder — WITH SCHEDULE TIMES + Key Rotation")
    print("=" * 65)
    print(f"  Primary  : {CSV_DECODED}")
    print(f"  Fallback : {CSV_SIMPLE}")
    print(f"  Keys     : {len(api_keys)} API key(s) loaded")
    print(f"  Model    : gemini-embedding-001 (3072-dim)")
    print(f"  DB       : {CHROMA_DIR}")
    print(f"  Batch    : {BATCH_SIZE}")
    print()

    client   = chromadb.PersistentClient(path=CHROMA_DIR)
    name_map = build_train_name_map(client)
    documents = build_docs(name_map)
    total     = len(documents)

    if not documents:
        print("[ERROR] No documents built — check CSV paths"); sys.exit(1)

    # ── Resume detection ──────────────────────────────────────────────────────
    try:
        existing = client.get_collection(COLLECTION).count()
    except Exception:
        existing = 0

    chk = load_checkpoint()
    if existing > 0 and chk > 0:
        resume_from = min(existing, chk)
        print(f"[RESUME] {existing:,} docs in DB, checkpoint at {chk:,} — resuming from doc #{resume_from + 1}")
    else:
        resume_from = 0
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

    print(f"\n[START] Embedding {len(documents):,} remaining docs ({total:,} total)")
    print(f"[KEY]   Using key #{key_index + 1} of {len(api_keys)}\n")

    for batch_start in range(0, len(documents), BATCH_SIZE):
        batch     = documents[batch_start: batch_start + BATCH_SIZE]
        batch_end = min(batch_start + BATCH_SIZE, len(documents))
        rate_limit_attempts = 0  # reset per batch

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
                batch_num = (batch_start // BATCH_SIZE) + 1
                total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
                print(f"   [{bar}] {stored:,}/{total:,} ({pct:.1f}%)  "
                      f"batch {batch_num}/{total_batches}  "
                      f"key #{key_index + 1}",
                      flush=True)
                save_checkpoint(stored)
                rate_limit_attempts = 0  # reset on success

                # Pacing — stay under 100 req/min
                if batch_end < len(documents):
                    time.sleep(INTER_SLEEP)
                break  # success — move to next batch

            except Exception as exc:
                err = str(exc)

                # ── Daily quota exhausted → stop cleanly ──────────────────────
                if "PerDay" in err:
                    save_checkpoint(stored)
                    print(f"\n{'='*65}")
                    print(f"[DAILY LIMIT] Key #{key_index+1} daily quota exhausted")
                    print(f"[SAVED] {stored:,}/{total:,} docs checkpointed")
                    print(f"\nTo continue:")
                    print(f"  1. Comment current GOOGLE_API_KEY in .env")
                    print(f"  2. Uncomment the next key")
                    print(f"  3. Run: .venv\\Scripts\\python scripts/embed_routes.py")
                    print(f"{'='*65}")
                    sys.exit(0)

                # ── Per-minute / generic 429 → wait and retry, max 5 times ───
                if "429" in err or "RESOURCE_EXHAUSTED" in err or "PerMinute" in err:
                    rate_limit_attempts += 1
                    if rate_limit_attempts >= 5:
                        save_checkpoint(stored)
                        print(f"\n{'='*65}")
                        print(f"[DAILY LIMIT] Key #{key_index+1} — 5 retries failed, key is exhausted")
                        print(f"[SAVED] {stored:,}/{total:,} docs checkpointed")
                        print(f"\nTo continue:")
                        print(f"  1. Comment current GOOGLE_API_KEY in .env")
                        print(f"  2. Uncomment the next key")
                        print(f"  3. Run: .venv\\Scripts\\python scripts/embed_routes.py")
                        print(f"{'='*65}")
                        sys.exit(0)
                    print(f"\n   [RATE] Rate limit (key #{key_index + 1}) — waiting 65s "
                          f"(attempt {rate_limit_attempts}/5)...")
                    time.sleep(5)
                    continue


                # ── Invalid key → rotate ──────────────────────────────────────
                if "UNAUTHENTICATED" in err or "401" in err or "API_KEY_INVALID" in err:
                    exhausted_keys.add(key_index)
                    print(f"\n   [INVALID] Key #{key_index + 1} — skipping.")
                    found = False
                    for _ in range(len(api_keys)):
                        key_index = (key_index + 1) % len(api_keys)
                        if key_index not in exhausted_keys:
                            found = True; break
                    if not found:
                        print(f"\n[STOP] No valid keys remaining."); sys.exit(1)
                    embeddings = _make_embeddings(api_keys[key_index])
                    if vector_store is not None:
                        vector_store = Chroma(
                            collection_name=COLLECTION,
                            embedding_function=embeddings,
                            client=client,
                        )
                    print(f"   [KEY]  Switched to key #{key_index + 1}")
                    continue

                # ── Network error → short retry ───────────────────────────────
                if "getaddrinfo" in err or "10065" in err or "Server disconnected" in err:
                    print(f"\n   [NET]  Network error — retrying in 10s...")
                    time.sleep(10)
                    continue

                # ── Unknown error → retry ─────────────────────────────────────
                print(f"\n   [ERR]  {err[:120]}")
                print(f"          Retrying in 15s...")
                time.sleep(15)
                continue

    count = client.get_collection(COLLECTION).count() if vector_store else existing
    print(f"\n\n{'=' * 65}")
    print(f"[DONE] '{COLLECTION}': {count:,} / {total:,} documents stored")
    if count >= total:
        print(f"[✓] All route documents embedded successfully with schedule times!")
        if os.path.exists(CHECKPOINT_F):
            os.remove(CHECKPOINT_F)
    else:
        print(f"[!] Partial: {total - count:,} remaining. Re-run to resume.")
    print(f"{'=' * 65}")
    print()
    print("Next steps:")
    print("  1. git add chroma_db/")
    print("  2. git commit -m 'chore: train_routes with schedule times'")
    print("  3. git push  →  Render auto-deploys")


if __name__ == "__main__":
    main()
