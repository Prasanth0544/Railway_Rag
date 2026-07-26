"""
embed_local.py — Embed Mumbai Metro/Local rail schedule data.

Sources (joined): schedule_trips.csv + schedule_stop_times.csv +
                  schedule_stations.csv + schedule_station_names.csv +
                  schedule_lines.csv + schedule_trip_calendar.csv

Join chain:
  trip.line_id        -> lines.id       (get line name)
  trip.calendar_id    -> calendar.id    (get running days)
  trip.stop_times_id  -> stop_times.id  (get all stops for this trip)
  stop_times.stn_id   -> stations.id    (get station code)
  stations.id         -> stn_names.station_id (get station name, lang=0 = English)

Strategy: One Document per trip (train service)

Document format:
  Mumbai Metro Line 1 trip from Ghatkopar (M-GHT) to Versova (M-VVR):
  departs 06:00, runs Daily. Stops (12): Ghatkopar > Jagruti Nagar > Asalpha > ...

Estimated docs: ~25,826 trips
Avg doc size: ~200 chars

Usage:
    .venv\Scripts\python scripts/embed_local.py
"""

from __future__ import annotations

import os, sys, time, json
from datetime import datetime
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

COLLECTION   = "local_schedules"
BATCH_SIZE   = 50
CHROMA_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
CHECKPOINT_F = os.path.join(CHROMA_DIR, ".embed_checkpoint.json")
DATA_DIR     = os.getenv("DATA_COLLECTIONS_DIR",
                         os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))

DAYS_MAP = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


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


def _seconds_to_hhmm(seconds: int) -> str:
    """Convert seconds-from-midnight to HH:MM string."""
    h = (seconds // 3600) % 24
    m = (seconds % 3600) // 60
    return f"{h:02d}:{m:02d}"


def _running_days(days_str: str) -> str:
    """Convert '1,1,1,1,1,1,1' -> 'Daily', '1,0,0,0,0,0,1' -> 'Mon, Sun'"""
    bits = [int(x) for x in str(days_str).split(",") if x.strip().isdigit()]
    if not bits:
        return "Unknown"
    if all(b == 1 for b in bits):
        return "Daily"
    if bits == [0,1,1,1,1,1,0]:
        return "Weekdays"
    if bits == [1,0,0,0,0,0,1]:
        return "Weekends"
    return ", ".join(DAYS_MAP[i] for i, b in enumerate(bits) if b == 1)


def load_local_documents() -> list[Document]:
    """Join all schedule_*.csv files and build one Document per trip."""

    def _path(name: str) -> str:
        return os.path.join(DATA_DIR, name)

    print("[LOAD] Reading schedule CSV files...")
    trips    = pd.read_csv(_path("schedule_trips.csv"), low_memory=False)
    stops    = pd.read_csv(_path("schedule_stop_times.csv"), low_memory=False)
    stations = pd.read_csv(_path("schedule_stations.csv"), low_memory=False)
    stn_names= pd.read_csv(_path("schedule_station_names.csv"), low_memory=False)
    lines    = pd.read_csv(_path("schedule_lines.csv"), low_memory=False)
    calendar = pd.read_csv(_path("schedule_trip_calendar.csv"), low_memory=False)

    print(f"  trips={len(trips):,}  stop_times={len(stops):,}  "
          f"stations={len(stations):,}  lines={len(lines):,}  "
          f"calendar={len(calendar):,}")

    # Build lookups
    # station_id -> {"code": ..., "name": ...}
    stn_name_map = (stn_names[stn_names["lang"] == 0]
                    .set_index("station_id")["name"].to_dict())
    stn_code_map = stations.set_index("id")["code"].to_dict()
    stn_lookup   = {sid: {"code": stn_code_map.get(sid, "?"),
                          "name": stn_name_map.get(sid, "")}
                    for sid in stn_code_map}

    # line_id -> line name
    line_map = lines.set_index("id")["name"].to_dict()

    # calendar_id -> running days string
    cal_map  = {row["id"]: _running_days(row["running_days_array"])
                for _, row in calendar.iterrows()}

    # stop_times_id -> sorted list of (stop_seq, stn_id, arr_offset, dep_offset, distance)
    stop_groups: dict[int, list] = {}
    for _, row in stops.iterrows():
        gid = int(row["id"])
        stop_groups.setdefault(gid, []).append({
            "seq"      : int(row["stop_seq"]),
            "stn_id"   : int(row["stn_id"]),
            "arr_off"  : int(row["arr_time_offset"]),
            "dep_off"  : int(row["dep_time_offset"]),
            "distance" : int(row["distance"]),
        })
    # Sort each group by stop_seq
    for gid in stop_groups:
        stop_groups[gid].sort(key=lambda x: x["seq"])

    documents = []
    skipped   = 0

    for _, trip in trips.iterrows():
        trip_id       = str(trip.get("id", "")).strip()
        trip_name     = str(trip.get("name", "")).strip()
        line_id       = int(trip.get("line_id", -1))
        calendar_id   = int(trip.get("calendar_id", -1))
        stop_times_id = int(trip.get("stop_times_id", -1))
        start_time    = int(trip.get("start_time", 0))

        line_name   = line_map.get(line_id, f"Line {line_id}")
        running_days= cal_map.get(calendar_id, "Daily")
        departs     = _seconds_to_hhmm(start_time)

        stop_list   = stop_groups.get(stop_times_id, [])
        if not stop_list:
            skipped += 1
            continue

        # Build stop sequence
        stn_labels = []
        for s in stop_list:
            info  = stn_lookup.get(s["stn_id"], {})
            code  = info.get("code", "?")
            name  = info.get("name", "")
            label = f"{name} ({code})" if name else code
            stn_labels.append(label)

        origin = stn_labels[0] if stn_labels else "?"
        dest   = stn_labels[-1] if stn_labels else "?"
        total  = len(stn_labels)

        stop_seq_str = " > ".join(stn_labels)

        text = (f"{line_name} trip from {origin} to {dest}. "
                f"Departs {departs}, runs {running_days}. "
                f"Stops ({total}): {stop_seq_str}.")

        documents.append(Document(
            page_content=text,
            metadata={
                "source_type"  : "local_schedule",
                "trip_id"      : trip_id,
                "line_name"    : line_name,
                "departure"    : departs,
                "running_days" : running_days,
                "total_stops"  : total,
            }
        ))

    print(f"[OK] Built {len(documents):,} local schedule documents (skipped {skipped})")
    return documents


def _make_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)


def main() -> None:
    api_keys = _load_api_keys()
    if not api_keys:
        print("[ERROR] No GOOGLE_API_KEY found in .env"); sys.exit(1)

    print("=" * 60)
    print("local_schedules Embedder (Mumbai Metro/Local)")
    print("=" * 60)
    print(f"  Keys : {len(api_keys)} API key(s) loaded")
    print(f"  DB   : {CHROMA_DIR}")
    print()

    documents = load_local_documents()
    total     = len(documents)
    if not documents:
        print("[ERROR] No documents built."); sys.exit(1)

    print(f"\n[SAMPLE 1] {documents[0].page_content[:200]}")
    print(f"[SAMPLE 2] {documents[100].page_content[:200]}")
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
                    print(f"\n   [QUOTA] Key #{key_index + 1} exhausted ({stored:,}/{total:,}).")
                    found = False
                    for _ in range(len(api_keys)):
                        key_index = (key_index + 1) % len(api_keys)
                        if key_index not in exhausted_keys: found = True; break
                    if not found:
                        print(f"\n[STOP] All keys exhausted. Progress saved."); sys.exit(0)
                    embeddings = _make_embeddings(api_keys[key_index])
                    remaining = len(api_keys) - len(exhausted_keys)
                    print(f"   [KEY]  Rotated to key #{key_index + 1} ({remaining} fresh remaining)")
                    time.sleep(65); continue
                if "429" in err or "RESOURCE_EXHAUSTED" in err or "PerMinute" in err:
                    print(f"\n   [RATE] Rate limit — waiting 65s..."); time.sleep(65); continue
                if "getaddrinfo" in err or "10065" in err or "Server disconnected" in err:
                    print(f"\n   [NET]  Network error — retrying in 10s..."); time.sleep(10); continue
                print(f"\n   [SKIP] Error for batch {batch_start}-{batch_end}: {err[:100]}"); break

    count = client.get_collection(COLLECTION).count() if vector_store else existing
    print(f"\n\n{'='*60}\n[DONE] '{COLLECTION}': {count:,}/{total:,} docs stored\n{'='*60}")


if __name__ == "__main__":
    main()
