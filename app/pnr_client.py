"""
pnr_client.py — Live PNR Status Enquiry Client
Fetches real-time PNR details from public JSON endpoints and fallback scrapers.
"""

import os
import re
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

try:
    from bs4 import BeautifulSoup  # type: ignore[import-untyped]
except ImportError:
    BeautifulSoup = None  # type: ignore[assignment,misc]

# Cache config
CACHE_TTL_SECONDS = 600  # 10 minutes
_pnr_cache: Dict[str, Dict[str, Any]] = {}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}


def get_pnr_status(pnr: str) -> dict:
    """
    Get live PNR status.
    Uses memory cache before querying live endpoints.
    """
    pnr = str(pnr).strip()
    if not re.match(r"^\d{10}$", pnr):
        return {
            "success": False,
            "pnr": pnr,
            "error": "Invalid PNR format. Must be a 10-digit number.",
            "fetched_at": datetime.now().isoformat(),
            "from_cache": False
        }

    # Check cache
    cached = _get_from_cache(pnr)
    if cached:
        return cached

    # Try ConfirmTkt API
    result = _fetch_confirmtkt(pnr)
    if result and result.get("success"):
        _set_cache(pnr, result)
        return result

    # Try RailYatri Fallback
    result = _fetch_railyatri_scraper(pnr)
    if result and result.get("success"):
        _set_cache(pnr, result)
        return result

    return {
        "success": False,
        "pnr": pnr,
        "error": "PNR status currently unavailable. Make sure PNR is valid and try again later.",
        "fetched_at": datetime.now().isoformat(),
        "from_cache": False
    }


def _get_from_cache(pnr: str) -> Optional[dict]:
    if pnr in _pnr_cache:
        entry = _pnr_cache[pnr]
        if datetime.now() < entry["expires_at"]:
            age = round((datetime.now() - entry["fetched_at"]).total_seconds())
            return {**entry["data"], "from_cache": True, "cache_age_seconds": age}
    return None


def _set_cache(pnr: str, data: dict):
    # Evict oldest entries when cache grows too large
    if len(_pnr_cache) >= 500:
        oldest_keys = sorted(_pnr_cache, key=lambda k: _pnr_cache[k]["fetched_at"])[:50]
        for k in oldest_keys:
            del _pnr_cache[k]
    now = datetime.now()
    _pnr_cache[pnr] = {
        "data": data,
        "fetched_at": now,
        "expires_at": now + timedelta(seconds=CACHE_TTL_SECONDS)
    }


def _fetch_confirmtkt(pnr: str) -> Optional[dict]:
    """Fetch PNR status from ConfirmTkt backend JSON API."""
    url = f"https://api.confirmtkt.com/api/pnr/pnrstatus?pnrNo={pnr}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        print(f"[PNR] ConfirmTkt URL: {url} -> HTTP {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            # If Pnr is returned, it is a valid response (even if message is returned, let's verify keys)
            if data and (data.get("Pnr") or data.get("PnrNo") or data.get("TrainNo")):
                # Normalize keys to a standard payload
                return {
                    "success": True,
                    "source": "ConfirmTkt",
                    "pnr": data.get("Pnr") or pnr,
                    "train_no": data.get("TrainNo", ""),
                    "train_name": data.get("TrainName", ""),
                    "date_of_journey": data.get("Doj", ""),
                    "booking_class": data.get("Class", ""),
                    "from_station": data.get("From", ""),
                    "to_station": data.get("To", ""),
                    "boarding_station": data.get("BoardingPoint", ""),
                    "chart_prepared": data.get("ChartPrepared", False),
                    "passengers": [
                        {
                            "passenger_no": p.get("PassengerNo", idx + 1),
                            "booking_status": p.get("BookingStatus", ""),
                            "current_status": p.get("CurrentStatus", ""),
                            "coach": p.get("Coach", ""),
                            "berth": p.get("Berth", 0),
                        }
                        for idx, p in enumerate(data.get("PassengerList", []))
                    ],
                    "fetched_at": datetime.now().isoformat(),
                    "from_cache": False
                }
    except Exception as e:
        print(f"[PNR] ConfirmTkt error: {e}")
    return None


def _fetch_railyatri_scraper(pnr: str) -> Optional[dict]:
    """Scrape PNR status from RailYatri HTML page."""
    url = f"https://www.railyatri.in/pnr-status/{pnr}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        print(f"[PNR] RailYatri URL: {url} -> HTTP {resp.status_code}")
        if resp.status_code == 200 and resp.text:
            if BeautifulSoup is None:
                print("[PNR] BeautifulSoup not available — skipping RailYatri scrape")
                return None
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # --- Robust Selector Parsing ---
            train_no = ""
            train_name = ""
            from_station = ""
            to_station = ""
            date_of_journey = ""
            booking_class = ""
            chart_prepared = False
            passengers = []

            # 1. Parse Train Details
            for el in soup.find_all(class_="train-info"):
                train_text = el.get_text()
                if "TRAIN NAME" in train_text:
                    num_match = re.search(r"\b(\d{5})\b", train_text)
                    if num_match:
                        train_no = num_match.group(1)
                    clean_name = train_text.replace("TRAIN NAME :", "").strip()
                    if num_match:
                        clean_name = clean_name.replace(num_match.group(1), "").strip()
                    # Clean special figure dash types and replace with standard hyphen
                    clean_name = re.sub(r"[\u2010-\u2015\u2212\u2014]", "-", clean_name)
                    clean_name = re.sub(r"\s+", " ", clean_name)
                    clean_name = clean_name.lstrip("-").strip()
                    train_name = clean_name
                    break

            # 2. Parse Stations
            route_el = soup.find(class_="train-route")
            if route_el:
                cols = route_el.find_all(class_="pnr-bold-txt")
                if len(cols) >= 2:
                    from_station = cols[0].get_text().strip()
                    to_station = cols[1].get_text().strip()

            # 3. Parse Boarding & Class Details
            boarding_el = soup.find(class_="boarding-detls")
            if boarding_el:
                cols = boarding_el.find_all(class_="pnr-bold-txt")
                if len(cols) >= 2:
                    date_of_journey = cols[0].get_text().strip()
                    booking_class = cols[1].get_text().strip()

            # 4. Parse Chart Status
            chart_el = soup.find(class_="chart-status-txt")
            if chart_el:
                chart_prepared = "not prepared" not in chart_el.get_text().lower()

            # 5. Parse Passengers List
            pax_items = soup.find_all("li", class_="PNRPasList")
            if pax_items:
                for idx, item in enumerate(pax_items):
                    pax_count_el = item.find(class_="paxListCount")
                    pax_no = pax_count_el.get_text().strip().replace(".", "") if pax_count_el else str(idx + 1)
                    
                    status_types = item.find_all(class_="statusType")
                    if len(status_types) >= 2:
                        booking_status = status_types[0].get_text().strip()
                        current_status = status_types[1].get_text().strip()
                        coach_berth = status_types[2].get_text().strip() if len(status_types) >= 3 else "--"
                        
                        coach = ""
                        berth = 0
                        if coach_berth and coach_berth != "--":
                            parts = coach_berth.split("/")
                            if len(parts) >= 1:
                                coach = parts[0]
                            if len(parts) >= 2:
                                try:
                                    berth = int(parts[1])
                                except ValueError:
                                    pass
                        else:
                            # Fallback if coach/berth is inline current status e.g. CNF/B1/22
                            coach_match = re.search(r"\b([A-Z]\d+)\b", current_status)
                            if coach_match:
                                coach = coach_match.group(1)
                            berth_match = re.search(r"\b(\d+)\b", current_status)
                            if berth_match:
                                berth = int(berth_match.group(1))

                        passengers.append({
                            "passenger_no": int(pax_no) if pax_no.isdigit() else idx + 1,
                            "booking_status": booking_status,
                            "current_status": current_status,
                            "coach": coach,
                            "berth": berth
                        })
            else:
                # --- Fallback to original Table Row scraper if layout changes ---
                page_text = soup.get_text()
                rows = soup.find_all("tr")
                p_count = 1
                for row in rows:
                    row_text = row.get_text()
                    if "Passenger" in row_text and ("CNF" in row_text or "W/L" in row_text or "RAC" in row_text):
                        cells = [c.get_text().strip() for c in row.find_all(["td", "th"])]
                        if len(cells) >= 3:
                            booking_status = cells[1]
                            current_status = cells[2]
                            coach = ""
                            berth = 0
                            coach_match = re.search(r"\b([A-Z]\d+)\b", current_status)
                            if coach_match:
                                coach = coach_match.group(1)
                            berth_match = re.search(r"\b(\d+)\b", current_status)
                            if berth_match:
                                berth = int(berth_match.group(1))
                                
                            passengers.append({
                                "passenger_no": p_count,
                                "booking_status": booking_status,
                                "current_status": current_status,
                                "coach": coach,
                                "berth": berth
                            })
                            p_count += 1
                
                if not train_no:
                    train_match = re.search(r"Train\s+(?:Name|No)[\s:]+(\d{5})\s*-\s*([^|\n]+)", page_text, re.IGNORECASE)
                    train_no = train_match.group(1) if train_match else ""
                    train_name = train_match.group(2).strip() if train_match else ""
                if not date_of_journey:
                    doj_match = re.search(r"Date\s+of\s+Journey[\s:]+(\d{1,2}[-\s][a-zA-Z]{3,}[-\s]\d{2,4})", page_text, re.IGNORECASE)
                    date_of_journey = doj_match.group(1).strip() if doj_match else ""
                chart_prepared = "chart prepared" in page_text.lower() and "chart not prepared" not in page_text.lower()

            if passengers or train_no:
                return {
                    "success": True,
                    "source": "RailYatri Scraper",
                    "pnr": pnr,
                    "train_no": train_no,
                    "train_name": train_name,
                    "date_of_journey": date_of_journey,
                    "booking_class": booking_class,
                    "from_station": from_station,
                    "to_station": to_station,
                    "boarding_station": from_station,  # Boarding station is usually from station
                    "chart_prepared": chart_prepared,
                    "passengers": passengers,
                    "fetched_at": datetime.now().isoformat(),
                    "from_cache": False
                }
    except Exception as e:
        print(f"[PNR] RailYatri scrape error: {e}")
    return None


def format_pnr_status_for_llm(pnr_data: dict) -> str:
    """Format PNR status dict into a readable string for the LLM context."""
    if not pnr_data.get("success"):
        return f"⚠️ LIVE PNR DATA UNAVAILABLE: {pnr_data.get('error', 'Unknown error')}"

    lines = [
        f"=== LIVE PNR STATUS (Source: {pnr_data.get('source', 'ConfirmTkt')} | "
        f"Fetched: {pnr_data.get('fetched_at', '')[:16]} IST) ===",
        f"PNR: {pnr_data['pnr']}",
        f"Train: {pnr_data['train_no']} - {pnr_data['train_name']}",
        f"Date of Journey: {pnr_data['date_of_journey']}",
        f"Booking Class: {pnr_data['booking_class'] or 'N/A'}",
        f"Route: {pnr_data['from_station'] or 'N/A'} -> {pnr_data['to_station'] or 'N/A'} (Boarding: {pnr_data['boarding_station'] or 'N/A'})",
        f"Chart Status: {'PREPARED ✅' if pnr_data['chart_prepared'] else 'NOT PREPARED ⚠️'}",
        "\nPassenger Details:"
    ]

    for p in pnr_data["passengers"]:
        coach_info = f" (Coach: {p['coach']}, Berth: {p['berth']})" if p['coach'] else ""
        lines.append(
            f"  - Passenger {p['passenger_no']}: Booking Status: {p['booking_status']} | "
            f"Current Status: {p['current_status']}{coach_info}"
        )

    lines.append("=" * 50)
    return "\n".join(lines)
