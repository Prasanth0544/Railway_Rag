"""
preprocess.py — Convert all data sources into LangChain Documents

Data Sources:
  1. data/railway_rules.csv       -> 183 railway rules documents
  2. train_info.csv (12k trains)  -> natural-language train documents
  3. station_info.csv (10k)       -> natural-language station documents
  4. train_route_decoded.csv      -> route documents (JSON route column)
  5. ticket_classes.csv + service_tax.csv -> reference documents

Column names come from the actual MongoDB export structure (erail APK data).
"""

import os
import json
import pandas as pd
from langchain_core.documents import Document

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DATA_COLLECTIONS_DIR = os.getenv(
    "DATA_COLLECTIONS_DIR",
    r"C:\Users\prasa\Documents\RailWayData\csv_of_railway\data_collections",
)


# ─────────────────────────────────────────────
# HELPER: Safe string
# ─────────────────────────────────────────────

def _safe(val, default="N/A") -> str:
    """Convert a value to string, return default if null/empty."""
    if val is None:
        return default
    s = str(val).strip()
    if s in ("", "nan", "NaN", "None", "NaT"):
        return default
    return s


# ─────────────────────────────────────────────
# 1. RAILWAY RULES (from data/railway_rules.csv)
# ─────────────────────────────────────────────

def load_rules_documents() -> list[Document]:
    """
    Load 183 railway rules from data/railway_rules.csv.
    Columns: category, sub_category, rule_title, rule_detail
    """
    csv_path = os.path.join(DATA_DIR, "railway_rules.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] railway_rules.csv not found at {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    documents = []

    for _, row in df.iterrows():
        category = _safe(row.get("category"), "General")
        sub_cat  = _safe(row.get("sub_category"), "")
        title    = _safe(row.get("rule_title"), "Rule")
        detail   = _safe(row.get("rule_description"), "")  # correct column name

        parts = [f"[Railway Rule - {category}]"]
        if sub_cat != "N/A":
            parts.append(f"Topic: {sub_cat}.")
        parts.append(f"{title}.")
        if detail != "N/A":
            parts.append(detail)

        text = " ".join(parts)
        documents.append(Document(
            page_content=text,
            metadata={
                "source_type" : "rule",
                "category"    : category,
                "sub_category": sub_cat,
                "rule_title"  : title,
            }
        ))

    print(f"[OK] Loaded {len(documents)} rule documents from railway_rules.csv")
    return documents


# ─────────────────────────────────────────────
# 2. TRAIN INFO (train_info.csv — 12,813 trains)
# ─────────────────────────────────────────────

def load_train_documents() -> list[Document]:
    """
    Load train_info.csv (12,813 rows).

    Actual columns (from MongoDB export):
      train_number, train_name, train_type, train_category,
      src_station_code, src_station_name, dest_station_code, dest_station_name,
      departure_time, arrival_time, distance_km, duration,
      running_days_text, classes, speed_type, zone_erail, total_stops
    """
    csv_path = os.path.join(DATA_COLLECTIONS_DIR, "train_info.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] train_info.csv not found at {csv_path}")
        return []

    df = pd.read_csv(csv_path, low_memory=False)
    documents = []
    skipped = 0

    for _, row in df.iterrows():
        train_no   = _safe(row.get("train_number"), "")
        train_name = _safe(row.get("train_name"), "")
        src_code   = _safe(row.get("src_station_code"), "")
        src_name   = _safe(row.get("src_station_name"), "")
        dst_code   = _safe(row.get("dest_station_code"), "")
        dst_name   = _safe(row.get("dest_station_name"), "")

        # Skip rows with missing essentials
        if not train_no or src_code == "N/A" or dst_code == "N/A":
            skipped += 1
            continue

        dep_time    = _safe(row.get("departure_time"), "N/A")
        arr_time    = _safe(row.get("arrival_time"), "N/A")
        train_type  = _safe(row.get("train_type") or row.get("train_category"), "Express")
        runs_on     = _safe(row.get("running_days_text"), "N/A")
        distance    = _safe(row.get("distance_km"), "N/A")
        duration    = _safe(row.get("duration"), "N/A")
        classes     = _safe(row.get("classes"), "N/A")
        speed_type  = _safe(row.get("speed_type"), "N/A")
        zone        = _safe(row.get("zone_erail"), "N/A")
        total_stops = _safe(row.get("total_stops"), "N/A")

        # Source and destination with both code and name
        src_display = f"{src_name} ({src_code})" if src_name != "N/A" else src_code
        dst_display = f"{dst_name} ({dst_code})" if dst_name != "N/A" else dst_code

        text = (
            f"Train {train_no} ({train_name}) is a {train_type} train "
            f"running from {src_display} to {dst_display}."
        )
        if runs_on != "N/A":
            text += f" Runs on: {runs_on}."
        if dep_time != "N/A" or arr_time != "N/A":
            text += f" Departure: {dep_time}, Arrival: {arr_time}."
        if distance != "N/A":
            text += f" Distance: {distance} km."
        if duration != "N/A":
            text += f" Duration: {duration}."
        if total_stops != "N/A":
            text += f" Total stops: {total_stops}."
        if classes != "N/A":
            # classes may be integer 0/1 — skip if not meaningful
            try:
                int(classes)
            except (ValueError, TypeError):
                text += f" Classes: {classes}."
        if speed_type != "N/A":
            text += f" Speed type: {speed_type}."
        if zone != "N/A":
            text += f" Zone: {zone}."

        documents.append(Document(
            page_content=text,
            metadata={
                "source_type"        : "train",
                "train_no"           : str(train_no),
                "train_name"         : str(train_name),
                "source_station"     : src_code,
                "destination_station": dst_code,
                "train_type"         : train_type,
            }
        ))

    print(f"[OK] Loaded {len(documents)} train documents (skipped {skipped} incomplete)")
    return documents


# ─────────────────────────────────────────────
# 3. STATION INFO (station_info.csv — 9,956)
# ─────────────────────────────────────────────

def load_station_documents() -> list[Document]:
    """
    Load station_info.csv.

    Actual columns (from MongoDB export):
      station_code, title (station name), lat, lng, city,
      wifi_station, uber_available, pop, lines
    Also joins station_zones.csv for zone info.
    """
    info_path  = os.path.join(DATA_COLLECTIONS_DIR, "station_info.csv")
    zones_path = os.path.join(DATA_COLLECTIONS_DIR, "station_zones.csv")

    if not os.path.exists(info_path):
        print(f"[WARN] station_info.csv not found at {info_path}")
        return []

    df = pd.read_csv(info_path, low_memory=False)

    # Join zones if available
    if os.path.exists(zones_path):
        try:
            zones_df = pd.read_csv(zones_path, low_memory=False)
            if "station_code" in zones_df.columns and "station_code" in df.columns:
                zone_col = [c for c in zones_df.columns if "zone" in c.lower()]
                if zone_col:
                    df = df.merge(
                        zones_df[["station_code", zone_col[0]]],
                        on="station_code", how="left"
                    )
                    df.rename(columns={zone_col[0]: "zone"}, inplace=True)
        except Exception as e:
            print(f"[WARN] Could not merge station_zones: {e}")

    documents = []
    skipped = 0

    for _, row in df.iterrows():
        code = _safe(row.get("station_code"), "")
        name = _safe(row.get("title") or row.get("station_name"), "")

        if not code or not name:
            skipped += 1
            continue

        lat  = _safe(row.get("lat") or row.get("latitude"), "N/A")
        lng  = _safe(row.get("lng") or row.get("longitude"), "N/A")
        city = _safe(row.get("city"), "N/A")
        zone = _safe(row.get("zone"), "N/A")
        wifi = _safe(row.get("wifi_station"), "N/A")

        text = f"Station {code} - {name}"
        if city != "N/A":
            text += f", located in {city}."
        else:
            text += "."
        if zone != "N/A":
            text += f" Railway zone: {zone}."
        if lat != "N/A" and lng != "N/A":
            text += f" Coordinates: {lat}N, {lng}E."
        if wifi == "1" or wifi == "True" or wifi == "true":
            text += " WiFi available at this station."

        documents.append(Document(
            page_content=text,
            metadata={
                "source_type" : "station",
                "station_code": code,
                "station_name": name,
                "city"        : city,
                "zone"        : zone,
            }
        ))

    print(f"[OK] Loaded {len(documents)} station documents (skipped {skipped})")
    return documents


# ─────────────────────────────────────────────
# 4. TRAIN ROUTES (train_route_decoded.csv)
# Route is stored as a JSON array in 'route' column
# ─────────────────────────────────────────────

def load_train_route_documents(max_trains: int = 3000) -> list[Document]:
    """
    Load train_route_decoded.csv.

    Actual structure:
      - train_number: train number (int)
      - stops: JSON array of stops, each with:
          {station_code, arrival, departure, halt_minutes, distance_km, day, is_stopping}
      - total_stops: int
      - total_distance_km: float
      - source_station: origin station code
      - dest_station: destination station code

    Builds one Document per train summarizing its route.
    max_trains: limit number of trains to embed (None = all 10,456).
    """
    csv_path = os.path.join(DATA_COLLECTIONS_DIR, "train_route_decoded.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] train_route_decoded.csv not found at {csv_path}")
        return []

    df = pd.read_csv(csv_path, low_memory=False)

    if max_trains:
        df = df.head(max_trains)

    documents = []
    skipped = 0

    for _, row in df.iterrows():
        train_no   = _safe(row.get("train_number"), "")
        train_name = _safe(row.get("train_name") or row.get("title"), "")
        src        = _safe(row.get("source_station"), "")
        dst        = _safe(row.get("dest_station"), "")
        total_dist = _safe(row.get("total_distance_km"), "")
        total_stops_val = _safe(row.get("total_stops"), "")

        if not train_no:
            skipped += 1
            continue

        stops_raw = row.get("stops")
        if pd.isna(stops_raw) or not stops_raw:
            skipped += 1
            continue

        try:
            stops = json.loads(str(stops_raw))
        except (json.JSONDecodeError, ValueError):
            skipped += 1
            continue

        if not stops or len(stops) < 2:
            skipped += 1
            continue

        # Build station sequence string
        station_seq = " > ".join(
            s.get("station_code", "?") for s in stops if s.get("is_stopping", True)
        )

        # First and last stop
        first = stops[0].get("station_code", src)
        last  = stops[-1].get("station_code", dst)

        # Build natural language document
        text = f"Train {train_no}"
        if train_name and train_name != "N/A":
            text += f" ({train_name})"
        text += f" route from {first} to {last}."
        if total_stops_val != "N/A":
            text += f" Total stops: {total_stops_val}."
        if total_dist != "N/A":
            text += f" Total distance: {total_dist} km."
        text += f" Station sequence: {station_seq}."

        documents.append(Document(
            page_content=text,
            metadata={
                "source_type"        : "train_route",
                "train_no"           : str(train_no),
                "train_name"         : train_name if train_name != "N/A" else "",
                "source_station"     : first,
                "destination_station": last,
                "total_stops"        : int(total_stops_val) if total_stops_val.isdigit() else len(stops),
            }
        ))

    print(f"[OK] Loaded {len(documents)} train route documents (skipped {skipped})")
    return documents


# ─────────────────────────────────────────────
# 5. REFERENCE DATA (ticket classes, service tax)
# ─────────────────────────────────────────────

def load_reference_documents() -> list[Document]:
    """Load ticket_classes.csv and service_tax.csv as reference documents."""
    documents = []

    for fname, label in [("ticket_classes.csv", "Ticket Class"), ("service_tax.csv", "Service Tax")]:
        fpath = os.path.join(DATA_COLLECTIONS_DIR, fname)
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath)
        count = 0
        for _, row in df.iterrows():
            parts = []
            for col in df.columns:
                val = _safe(row.get(col), "")
                if val and val != "N/A":
                    parts.append(f"{col}: {val}")
            if parts:
                text = f"{label} Info - " + ". ".join(parts) + "."
                documents.append(Document(
                    page_content=text,
                    metadata={"source_type": "reference", "ref_type": label.lower().replace(" ", "_")}
                ))
                count += 1
        print(f"  [{label}] {count} records loaded")

    print(f"[OK] Loaded {len(documents)} reference documents total")
    return documents


# ─────────────────────────────────────────────
# MAIN LOADER
# ─────────────────────────────────────────────

def load_all_documents(include_routes: bool = True) -> dict:
    """Load all documents from all sources. Returns dict keyed by collection name."""
    result = {}

    print("\n[1] Loading Railway Rules...")
    result["railway_rules"] = load_rules_documents()

    print("\n[2] Loading Train Info (12k trains)...")
    result["trains"] = load_train_documents()

    print("\n[3] Loading Station Info (10k stations)...")
    result["stations"] = load_station_documents()

    if include_routes:
        print("\n[4] Loading Train Routes (first 3k trains)...")
        result["train_routes"] = load_train_route_documents(max_trains=3000)
    else:
        result["train_routes"] = []

    print("\n[5] Loading Reference Data...")
    result["references"] = load_reference_documents()

    total = sum(len(v) for v in result.values())
    print(f"\n{'='*55}")
    print(f"TOTAL DOCUMENTS: {total:,}")
    for k, v in result.items():
        print(f"   {k}: {len(v):,}")
    print(f"{'='*55}")

    return result


# Quick test
if __name__ == "__main__":
    docs = load_all_documents(include_routes=False)
    for category, doc_list in docs.items():
        if doc_list:
            print(f"\n[{category.upper()}]")
            print(f"  {doc_list[0].page_content[:250]}")
            print(f"  Metadata: {doc_list[0].metadata}")
