"""
preprocess.py — Convert all data sources into LangChain Documents

Data Sources:
  1. data/railway_rules.csv         -> 183 railway rules documents
  2. train_info.csv (12k trains)    -> natural-language train documents
  3. station_info.csv (10k)         -> station location & amenity documents
  4. station_zones.csv              -> zone mappings LINKED to station_info
  5. station_aka_info.csv           -> alternate station names LINKED to station_info
  6. train_route_decoded.csv        -> route documents with stop-level timings,
                                       station names injected via station lookup
  7. ticket_classes.csv + service_tax.csv -> reference documents

Linking strategy:
  build_station_lookup() merges station_info + station_zones + station_aka_info
  by station_code to produce a single dict used by both station doc loader
  and the route doc loader to inject full station names & aliases into embeddings.
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
# LINKED STATION LOOKUP TABLE
# Merges station_info + station_zones + station_aka_info
# ─────────────────────────────────────────────

def build_station_lookup() -> dict:
    """
    Build a dict keyed by station_code containing enriched station info:
      {
        'name'    : canonical station name (from station_info.title),
        'city'    : city name,
        'zone'    : railway zone (from station_zones),
        'zone_name': full zone name,
        'lat'     : latitude,
        'lng'     : longitude,
        'wifi'    : True/False,
        'aka'     : [list of alternate names from station_aka_info]
      }

    Used by route builder to inject station names into stop sequences,
    and by station builder to embed alternate names for fuzzy matching.
    """
    info_path  = os.path.join(DATA_COLLECTIONS_DIR, "station_info.csv")
    zones_path = os.path.join(DATA_COLLECTIONS_DIR, "station_zones.csv")
    aka_path   = os.path.join(DATA_COLLECTIONS_DIR, "station_aka_info.csv")

    lookup: dict = {}

    # --- Base: station_info ---
    if os.path.exists(info_path):
        df = pd.read_csv(info_path, low_memory=False)
        for _, row in df.iterrows():
            code = _safe(row.get("station_code"), "")
            if not code:
                continue
            lookup[code] = {
                "name"     : _safe(row.get("title") or row.get("station_name"), ""),
                "city"     : _safe(row.get("city"), ""),
                "zone"     : "",
                "zone_name": "",
                "lat"      : _safe(row.get("lat") or row.get("latitude"), ""),
                "lng"      : _safe(row.get("lng") or row.get("longitude"), ""),
                "wifi"     : str(row.get("wifi_station", "0")) in ("1", "True", "true"),
                "aka"      : [],
            }
        print(f"[Lookup] station_info loaded: {len(lookup)} stations")

    # --- Enrich: station_zones (join on station_code) ---
    if os.path.exists(zones_path):
        zones_df = pd.read_csv(zones_path, low_memory=False)
        enriched = 0
        for _, row in zones_df.iterrows():
            code = _safe(row.get("station_code"), "")
            if code in lookup:
                lookup[code]["zone"]      = _safe(row.get("zone") or row.get("zone_name"), "")
                lookup[code]["zone_name"] = _safe(row.get("zone_name") or row.get("zone_mapping_name"), "")
                # Also fill name from zones if missing in station_info
                if not lookup[code]["name"]:
                    lookup[code]["name"] = _safe(row.get("station_name"), "")
                enriched += 1
        print(f"[Lookup] station_zones enriched: {enriched} stations")

    # --- Enrich: station_aka_info (alternate names, join on station_code) ---
    if os.path.exists(aka_path):
        aka_df = pd.read_csv(aka_path, low_memory=False)
        enriched = 0
        for _, row in aka_df.iterrows():
            code = _safe(row.get("station_code"), "")
            aka_name = _safe(row.get("title") or row.get("aka_name"), "")
            if code and aka_name:
                if code not in lookup:
                    lookup[code] = {"name": aka_name, "city": "", "zone": "",
                                    "zone_name": "", "lat": "", "lng": "",
                                    "wifi": False, "aka": []}
                if aka_name not in lookup[code]["aka"] and aka_name != lookup[code]["name"]:
                    lookup[code]["aka"].append(aka_name)
                    enriched += 1
        print(f"[Lookup] station_aka_info added {enriched} alternate names")

    print(f"[Lookup] Final station lookup: {len(lookup)} unique station codes")
    return lookup


# Singleton — built once, shared by all loaders
_STATION_LOOKUP: dict | None = None

def get_station_lookup() -> dict:
    global _STATION_LOOKUP
    if _STATION_LOOKUP is None:
        _STATION_LOOKUP = build_station_lookup()
    return _STATION_LOOKUP


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
    Load station_info.csv enriched with station_zones and station_aka_info
    via the shared get_station_lookup() table.

    Each document embeds:
      - canonical name, city, zone / zone_name
      - GPS coordinates
      - WiFi / Uber availability
      - All alternate names (AKAs) so fuzzy name queries resolve correctly.
        e.g. 'Lokmanya Tilak Terminus' and 'Kurla Terminus' both map to LTT.
    """
    info_path = os.path.join(DATA_COLLECTIONS_DIR, "station_info.csv")
    if not os.path.exists(info_path):
        print(f"[WARN] station_info.csv not found at {info_path}")
        return []

    df = pd.read_csv(info_path, low_memory=False)
    lookup = get_station_lookup()          # merged lookup with zones + AKAs

    documents = []
    skipped = 0

    for _, row in df.iterrows():
        code = _safe(row.get("station_code"), "")
        if not code:
            skipped += 1
            continue

        # Pull enriched data from lookup (already merged with zones + AKA)
        info      = lookup.get(code, {})
        name      = info.get("name") or _safe(row.get("title") or row.get("station_name"), "")
        city      = info.get("city") or _safe(row.get("city"), "")
        zone      = info.get("zone") or ""
        zone_name = info.get("zone_name") or ""
        lat       = info.get("lat")  or _safe(row.get("lat")  or row.get("latitude"),  "")
        lng       = info.get("lng")  or _safe(row.get("lng")  or row.get("longitude"), "")
        wifi      = info.get("wifi", False) or str(row.get("wifi_station", "0")) in ("1",)
        akas      = info.get("aka", [])

        if not name:
            skipped += 1
            continue

        # --- Build natural-language document ---
        text = f"Station {code} — {name}"
        if city and city != "N/A":
            text += f", located in {city}."
        else:
            text += "."

        if zone and zone != "N/A":
            if zone_name and zone_name != zone:
                text += f" Railway zone: {zone} ({zone_name})."
            else:
                text += f" Railway zone: {zone}."

        if lat and lat != "N/A" and lng and lng != "N/A":
            text += f" Coordinates: {lat}N, {lng}E."

        if wifi:
            text += " WiFi available at this station."

        # Embed alternate names explicitly so vector search matches them
        if akas:
            text += f" Also known as: {', '.join(akas)}."

        documents.append(Document(
            page_content=text,
            metadata={
                "source_type" : "station",
                "station_code": code,
                "station_name": name,
                "city"        : city if city != "N/A" else "",
                "zone"        : zone if zone != "N/A" else "",
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

        # Build station sequence string — inject names from lookup table
        # (stop JSON only has station_code; no station_name field)
        lookup = get_station_lookup()
        stopping_stops = [s for s in stops if s.get("is_stopping", True)]

        station_seq_parts = []
        for s in stopping_stops:
            code = s.get("station_code", "?")
            stn  = lookup.get(code, {})
            name = stn.get("name", "")
            if name and name != code:
                station_seq_parts.append(f"{name} ({code})")
            else:
                station_seq_parts.append(code)

        station_seq = " > ".join(station_seq_parts)

        # First and last stop — resolve names from lookup
        first = stops[0].get("station_code", src)
        last  = stops[-1].get("station_code", dst)
        first_name = lookup.get(first, {}).get("name", "")
        last_name  = lookup.get(last,  {}).get("name", "")

        first_display = f"{first_name} ({first})" if first_name and first_name != first else first
        last_display  = f"{last_name} ({last})"  if last_name  and last_name  != last  else last

        # Build natural language document
        text = f"Train {train_no}"
        if train_name and train_name != "N/A":
            text += f" ({train_name})"
        text += f" route from {first_display} to {last_display}."
        if total_stops_val != "N/A":
            text += f" Total stops: {total_stops_val}."
        if total_dist != "N/A":
            text += f" Total distance: {total_dist} km."
        text += f" Station sequence: {station_seq}."

        # Per-stop schedule — resolve station name from lookup, embed arrival/departure/halt
        stop_lines = []
        for s in stopping_stops:
            code = s.get("station_code", "?")
            stn  = lookup.get(code, {})
            name = stn.get("name", "")
            arr  = s.get("arrival",      "")
            dep  = s.get("departure",    "")
            day  = s.get("day",          "")
            halt = s.get("halt_minutes", "")
            dist = s.get("distance_km",  "")

            label = f"{name} ({code})" if name and name != code else code
            parts = [label]
            if arr  and str(arr)  not in ("", "None", "N/A", "First", "Last"): parts.append(f"arr {arr}")
            if dep  and str(dep)  not in ("", "None", "N/A", "First", "Last"): parts.append(f"dep {dep}")
            if day  and str(day)  not in ("", "None", "N/A"): parts.append(f"day {day}")
            if halt and str(halt) not in ("", "None", "N/A", "0"):  parts.append(f"halt {halt} min")
            if dist and str(dist) not in ("", "None", "N/A", "0.0"): parts.append(f"{dist} km")
            stop_lines.append(", ".join(parts))

        if stop_lines:
            text += " Schedule: " + " | ".join(stop_lines) + "."

        documents.append(Document(
            page_content=text,
            metadata={
                "source_type"        : "train_route",
                "train_no"           : str(train_no),
                "train_name"         : train_name if train_name != "N/A" else "",
                "source_station"     : first,
                "destination_station": last,
                "total_stops"        : int(total_stops_val) if str(total_stops_val).isdigit() else len(stops),
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
