"""
recover_skipped_docs.py — Recovers the ~400 docs skipped due to disk-full errors.

During re-ingestion, batches at positions 7,350-7,749 in the CSV were skipped
because of "database or disk is full" errors. This script:
  1. Reads exactly those rows from the CSV
  2. Builds enriched docs (same format as reingest_routes_enriched.py)
  3. ADDS them to the existing train_routes collection (no delete)

Usage:
    .venv\\Scripts\\python scripts/recover_skipped_docs.py

Configure START_ROW and END_ROW below based on your skip range.
"""
from __future__ import annotations

import os, sys, json, time, re
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

# ── Config — adjust if skip range was different ───────────────────────────────
START_ROW  = 7350   # first skipped CSV row (0-indexed)
END_ROW    = 7750   # last skipped CSV row (exclusive)
BATCH_SIZE = 50
COLLECTION = "train_routes"
CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
DATA_DIR   = os.getenv("DATA_COLLECTIONS_DIR",
                 os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
CSV_PATH   = os.path.join(DATA_DIR, "train_routes.csv")

# ── Key loading (no break on gaps) ───────────────────────────────────────────
def _load_api_keys() -> list[str]:
    keys = []
    v = os.getenv("GOOGLE_API_KEY", "").strip()
    if v: keys.append(v)
    for i in range(1, 100):
        v = os.getenv(f"GOOGLE_API_KEY_{i}", "").strip()
        if v: keys.append(v)
    return keys

def _make_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=api_key)

# ── Train name lookup ─────────────────────────────────────────────────────────
def build_train_name_map(client) -> dict:
    try:
        col = client.get_collection("trains")
        result = col.get(limit=20000, include=["documents", "metadatas"])
        name_map = {}
        for doc, meta in zip(result["documents"], result["metadatas"]):
            tn = str(meta.get("train_no", "")).strip()
            if not tn: continue
            name, runs = "", ""
            m = re.search(r"\(([^)]+)\)", doc)
            if m: name = m.group(1).strip()
            m2 = re.search(r"Runs on:\s*([^.]+)", doc)
            if m2: runs = m2.group(1).strip()
            name_map[tn] = (name, runs)
        print(f"[INFO] Loaded {len(name_map):,} train names")
        return name_map
    except Exception as e:
        print(f"[WARN] trains collection error: {e}"); return {}

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    api_keys = _load_api_keys()
    if not api_keys:
        print("[ERROR] No API keys"); sys.exit(1)

    print("=" * 60)
    print(f"Recovery: CSV rows {START_ROW}–{END_ROW-1} ({END_ROW-START_ROW} docs)")
    print("=" * 60)
    print(f"  Keys: {len(api_keys)}")

    client   = chromadb.PersistentClient(path=CHROMA_DIR)
    name_map = build_train_name_map(client)

    # Read ONLY the skipped rows
    print(f"[INFO] Reading rows {START_ROW}–{END_ROW-1} from CSV...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    skipped_df = df.iloc[START_ROW:END_ROW]
    print(f"[INFO] {len(skipped_df)} rows to recover")

    # Check which train_nos are already in collection
    print("[INFO] Checking which trains are already in collection...")
    try:
        existing_col = client.get_collection(COLLECTION)
        existing_data = existing_col.get(limit=20000, include=["metadatas"])
        existing_train_nos = {m.get("train_no","") for m in existing_data["metadatas"]}
        print(f"[INFO] Collection has {len(existing_train_nos):,} unique train_nos")
    except Exception as e:
        print(f"[WARN] Could not read existing collection: {e}")
        existing_train_nos = set()

    # Build docs for missing trains only
    docs: list[Document] = []
    already_present = 0
    for _, row in skipped_df.iterrows():
        train_no   = str(row.get("train_number","")).strip()
        src        = str(row.get("source_station","")).strip()
        dst        = str(row.get("dest_station","")).strip()
        total_dist = str(row.get("total_distance_km","")).strip()
        total_stops= str(row.get("total_stops","")).strip()
        stop_codes = str(row.get("stop_codes","")).strip()

        if not train_no or train_no in ("nan",""): continue

        if train_no in existing_train_nos:
            already_present += 1
            continue  # already in collection — skip

        try:    codes = json.loads(stop_codes)
        except: codes = [c.strip().strip('"') for c in stop_codes.strip("[]").split(",") if c.strip()]
        if not codes: continue

        if not src or src=="nan": src = codes[0]
        if not dst or dst=="nan": dst = codes[-1]

        name, runs = name_map.get(train_no, ("",""))
        name_part  = f" — {name}" if name else ""
        days_part  = f" ({runs})"  if runs  else ""
        stop_seq   = " > ".join(codes)

        text = f"Train {train_no}{name_part}{days_part}. From {src} to {dst}."
        if total_stops and total_stops not in ("nan","N/A"):
            text += f" Stops ({total_stops}): {stop_seq}."
        else:
            text += f" Stops: {stop_seq}."
        if total_dist and total_dist not in ("nan","N/A"):
            text += f" Distance: {total_dist} km."

        docs.append(Document(
            page_content=text,
            metadata={"source_type":"train_route","train_no":train_no,
                      "source_station":src,"destination_station":dst,
                      "total_stops": int(total_stops) if total_stops.isdigit() else len(codes),
                      "total_distance_km":total_dist}
        ))

    print(f"[INFO] {already_present} already in collection, {len(docs)} need recovery")
    if not docs:
        print("[OK] Nothing to recover!"); return

    # Embed and add (no collection delete — just add)
    key_idx = 0
    exhausted: set[int] = set()
    embeddings = _make_embeddings(api_keys[key_idx])
    total_batches = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE
    stored = 0
    vector_store = None

    print(f"[START] Adding {len(docs)} recovered docs in {total_batches} batches\n")

    for batch_start in range(0, len(docs), BATCH_SIZE):
        batch = docs[batch_start:batch_start+BATCH_SIZE]

        while True:
            try:
                if vector_store is None:
                    vector_store = Chroma(client=client, collection_name=COLLECTION,
                                          embedding_function=embeddings)
                vector_store.add_documents(batch)
                stored += len(batch)
                pct = stored / len(docs) * 100
                print(f"\r  [{('█'*int(pct/5))+'░'*(20-int(pct/5))}] {stored}/{len(docs)} ({pct:.0f}%)",
                      end="", flush=True)
                time.sleep(1.5); break

            except Exception as exc:
                err = str(exc)
                if "PerDay" in err:
                    exhausted.add(key_idx)
                    found = False
                    for _ in range(len(api_keys)):
                        key_idx = (key_idx+1) % len(api_keys)
                        if key_idx not in exhausted: found=True; break
                    if not found:
                        print(f"\n[STOP] All keys exhausted. {stored}/{len(docs)} recovered.")
                        sys.exit(0)
                    embeddings = _make_embeddings(api_keys[key_idx])
                    print(f"\n[KEY] Rotated to key #{key_idx+1}"); time.sleep(65); continue
                if "UNAUTHENTICATED" in err or "401" in err:
                    exhausted.add(key_idx)
                    for _ in range(len(api_keys)):
                        key_idx = (key_idx+1) % len(api_keys)
                        if key_idx not in exhausted: break
                    embeddings = _make_embeddings(api_keys[key_idx])
                    print(f"\n[INVALID] Bad key, rotated to #{key_idx+1}"); continue
                if "429" in err or "RESOURCE_EXHAUSTED" in err or "PerMinute" in err:
                    print(f"\n[RATE] Waiting 65s..."); time.sleep(65); continue
                if "disk is full" in err or "code: 13" in err:
                    print(f"\n[DISK] Disk still full! Free space and re-run."); sys.exit(1)
                if "getaddrinfo" in err or "10065" in err:
                    print(f"\n[NET] Network error, retrying..."); time.sleep(10); continue
                print(f"\n[SKIP] Unknown error: {err[:80]}"); break

    try:
        count = client.get_collection(COLLECTION).count()
    except: count = "?"
    print(f"\n\n{'='*60}")
    print(f"[DONE] Recovered {stored}/{len(docs)} docs")
    print(f"       Collection now has {count} total docs")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
