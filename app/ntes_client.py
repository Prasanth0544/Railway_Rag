"""
ntes_client.py — NTES Live Train Data Client

Fetches live train running status from India's National Train Enquiry System.
Includes 5-minute in-memory cache per session to avoid hammering the API.
Gracefully returns error dict if NTES is unavailable (never raises exceptions).

Usage:
    from app.ntes_client import get_train_running_status, format_live_status_for_llm
    status = get_train_running_status("17225")
"""

import re
import requests
import time
from datetime import datetime, timedelta
from typing import Optional

# ── In-memory cache (per server session, 5-min TTL) ──────────────
from app.logger import get_logger
logger = get_logger("app.ntes_client")

_live_cache: dict[str, dict] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes

# ── HTTP config ───────────────────────────────────────────────────
REQUEST_TIMEOUT = 8  # seconds

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Referer": "https://enquiry.indianrail.gov.in/mntes",
}

# ── Known API endpoints (try in order) ───────────────────────────
NTES_ENDPOINTS = [
    # Primary: NTES official
    "https://enquiry.indianrail.gov.in/NTES/GetTrainRunningStatus",
    # Alternate NTES path
    "https://enquiry.indianrail.gov.in/mntes/GetTrainRunningStatus",
]

ERAIL_TRAIN_ENDPOINT = "https://erail.in/rail/getTrains.aspx"


# ─────────────────────────────────────────────────────────────────
# CACHE HELPERS
# ─────────────────────────────────────────────────────────────────

def _is_cached(train_no: str) -> bool:
    if train_no not in _live_cache:
        return False
    return datetime.now() < _live_cache[train_no]["expires_at"]


def _get_from_cache(train_no: str) -> Optional[dict]:
    if _is_cached(train_no):
        data = dict(_live_cache[train_no]["data"])
        data["from_cache"] = True
        data["cache_age_seconds"] = int(
            (datetime.now() - _live_cache[train_no]["fetched_at"]).total_seconds()
        )
        return data
    return None


def _set_cache(train_no: str, data: dict) -> None:
    _live_cache[train_no] = {
        "data": data,
        "fetched_at": datetime.now(),
        "expires_at": datetime.now() + timedelta(seconds=CACHE_TTL_SECONDS),
    }


def clear_cache(train_no: Optional[str] = None) -> None:
    """Clear cache for specific train, or all if train_no=None."""
    if train_no:
        _live_cache.pop(train_no, None)
    else:
        _live_cache.clear()


# ─────────────────────────────────────────────────────────────────
# MAIN PUBLIC API
# ─────────────────────────────────────────────────────────────────

def get_train_running_status(train_no: str) -> dict:
    """
    Get live running status for a train.

    Returns a dict with:
        success       : bool
        train_no      : str
        train_name    : str  (if available)
        current_station: str (if available)
        delay_minutes : int
        status        : str ("On time" / "X min late" / "Arrived" / "Cancelled")
        source        : str ("NTES" / "erail.in" / "cache")
        fetched_at    : str (ISO timestamp)
        error         : str (only if success=False)
        from_cache    : bool
    """
    train_no = str(train_no).strip()
    logger.info(f"[NTES] Request for train {train_no}")

    # 1. Cache hit
    cached = _get_from_cache(train_no)
    if cached:
        logger.info(f"[NTES] Cache hit — age {cached.get('cache_age_seconds', '?')}s")
        return cached

    # 2. Try NTES primary endpoints
    for endpoint in NTES_ENDPOINTS:
        result = _fetch_ntes(train_no, endpoint)
        if result and result.get("success"):
            _set_cache(train_no, result)
            return result

    # 3. Fallback: erail.in
    result = _fetch_erail(train_no)
    if result and result.get("success"):
        _set_cache(train_no, result)
        return result

    # 4. All sources failed
    logger.info(f"[NTES] All sources failed for train {train_no}")
    return {
        "success": False,
        "train_no": train_no,
        "error": "Live data unavailable — NTES and backup sources not responding.",
        "fetched_at": datetime.now().isoformat(),
        "from_cache": False,
    }


def get_station_live_board(station_code: str, board_type: str = "ARR") -> dict:
    """
    Get live arrivals or departures for a station.
    board_type: "ARR" (arrivals) or "DEP" (departures)
    """
    station_code = station_code.upper().strip()
    cache_key = f"station_{station_code}_{board_type}"

    # Use same cache mechanism
    if cache_key in _live_cache and datetime.now() < _live_cache[cache_key]["expires_at"]:
        return {**_live_cache[cache_key]["data"], "from_cache": True}

    try:
        resp = requests.get(
            "https://enquiry.indianrail.gov.in/NTES/GetArrivalDeparture",
            params={"stnCode": station_code, "type": board_type},
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            result = {
                "success": True,
                "station_code": station_code,
                "board_type": board_type,
                "trains": data if isinstance(data, list) else data.get("trains", []),
                "source": "NTES",
                "fetched_at": datetime.now().isoformat(),
                "from_cache": False,
            }
            _live_cache[cache_key] = {
                "data": result,
                "fetched_at": datetime.now(),
                "expires_at": datetime.now() + timedelta(seconds=CACHE_TTL_SECONDS),
            }
            return result
    except Exception as e:
        logger.info(f"[NTES] Station board failed for {station_code}: {e}")

    return {
        "success": False,
        "station_code": station_code,
        "error": "Live station board unavailable.",
        "fetched_at": datetime.now().isoformat(),
        "from_cache": False,
    }


# ─────────────────────────────────────────────────────────────────
# FETCH IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────

def _fetch_ntes(train_no: str, endpoint: str) -> Optional[dict]:
    """Fetch from an NTES servlet endpoint using CSRF token post verification."""
    try:
        session = requests.Session()
        session.headers.update(HEADERS)

        # 1. Fetch homepage to initialize cookies
        home_url = "https://enquiry.indianrail.gov.in/mntes/"
        session.get(home_url, timeout=REQUEST_TIMEOUT)

        # 2. Get CSRF Token
        csrf_url = f"https://enquiry.indianrail.gov.in/mntes/GetCSRFToken?t={int(time.time() * 1000)}"
        r_csrf = session.get(csrf_url, timeout=REQUEST_TIMEOUT)
        
        csrf_data = {}
        if r_csrf.status_code == 200:
            match = re.search(r"name='([^']+)'\s+value='([^']+)'", r_csrf.text)
            if not match:
                match = re.search(r'name="([^"]+)"\s+value="([^"]+)"', r_csrf.text)
            if match:
                csrf_data[match.group(1)] = match.group(2)

        # 3. Format current date as journey date (dd-MMM-yyyy)
        # Note: %b yields 'Jul', 'Aug', etc.
        j_date = datetime.now().strftime("%d-%b-%Y")

        # 4. POST to tr servlet
        tr_url = "https://enquiry.indianrail.gov.in/mntes/tr"
        params = {
            "opt": "TrainRunning",
            "subOpt": "fullR",
            "trainNo": train_no,
            "jDate": j_date,
            "date": "0",
            "startDay": "0"
        }
        params.update(csrf_data)

        resp = session.post(tr_url, data=params, timeout=REQUEST_TIMEOUT)
        logger.info(f"[NTES] tr POST status: {resp.status_code} | length: {len(resp.text)}")
        
        if resp.status_code == 200 and len(resp.text) > 100:
            result = _parse_ntes_text(train_no, resp.text)
            # Fallback: if train hasn't started running yet for the server's "today",
            # query yesterday's date which is likely the currently active run.
            if result and (not result.get("current_station") or result.get("current_station") == "Station Info Loaded"):
                yesterday = datetime.now() - timedelta(days=1)
                y_date = yesterday.strftime("%d-%b-%Y")
                logger.info(f"[NTES] Train not started for today's date ({j_date}). Trying yesterday's date ({y_date})...")
                
                # Fetch a fresh CSRF token for the fallback request
                r_csrf_y = session.get(csrf_url, timeout=REQUEST_TIMEOUT)
                csrf_data_y = {}
                if r_csrf_y.status_code == 200:
                    match_y = re.search(r"name='([^']+)'\s+value='([^']+)'", r_csrf_y.text)
                    if not match_y:
                        match_y = re.search(r'name="([^"]+)"\s+value="([^"]+)"', r_csrf_y.text)
                    if match_y:
                        csrf_data_y[match_y.group(1)] = match_y.group(2)
                
                params_y = {
                    "opt": "TrainRunning",
                    "subOpt": "fullR",
                    "trainNo": train_no,
                    "jDate": y_date,
                    "date": "0",
                    "startDay": "0"
                }
                params_y.update(csrf_data_y)
                
                resp_y = session.post(tr_url, data=params_y, timeout=REQUEST_TIMEOUT)
                logger.info(f"[NTES] Yesterday fallback tr POST status: {resp_y.status_code} | length: {len(resp_y.text)}")
                if resp_y.status_code == 200 and len(resp_y.text) > 100:
                    result_y = _parse_ntes_text(train_no, resp_y.text)
                    logger.info(f"[NTES] Yesterday fallback parsed: {result_y}")
                    if result_y and result_y.get("current_station") and result_y.get("current_station") != "Station Info Loaded":
                        return result_y
            return result
    except Exception as e:
        logger.info(f"[NTES] Error fetching live status: {e}")
    return None


def _fetch_erail(train_no: str) -> Optional[dict]:
    """Fallback: erail.in schedule/status API."""
    try:
        resp = requests.get(
            ERAIL_TRAIN_ENDPOINT,
            params={
                "TrainNo": train_no,
                "DataSource": "0",
                "Language": "0",
                "Cache": "2",
            },
            headers={**HEADERS, "Referer": "https://erail.in/"},
            timeout=REQUEST_TIMEOUT,
        )
        logger.info(f"[NTES] erail.in → HTTP {resp.status_code}")
        if resp.status_code == 200 and resp.text.strip():
            return _parse_erail_response(train_no, resp.text)
    except Exception as e:
        logger.info(f"[NTES] erail.in error: {e}")
    return None


# ─────────────────────────────────────────────────────────────────
# PARSERS
# ─────────────────────────────────────────────────────────────────

def _parse_ntes_json(train_no: str, data: dict) -> Optional[dict]:
    """Parse NTES JSON response into standard format."""
    try:
        # Handle wrapped responses
        if isinstance(data, list) and data:
            data = data[0]
        if not isinstance(data, dict):
            return None

        # Extract fields — NTES uses various field names
        train_name = (
            data.get("TrainName") or data.get("trainName") or
            data.get("Train_Name") or data.get("train_name") or ""
        )
        current_stn = (
            data.get("CurrentStation") or data.get("currentStation") or
            data.get("CurrStation") or data.get("curr_station") or ""
        )
        delay_raw = (
            data.get("DelayedBy") or data.get("lateBy") or
            data.get("Delay") or data.get("delay") or 0
        )
        status_raw = (
            data.get("Status") or data.get("status") or
            data.get("TrainStatus") or data.get("trainStatus") or ""
        )

        delay_min = _parse_delay(delay_raw)

        return {
            "success": True,
            "source": "NTES",
            "train_no": train_no,
            "train_name": str(train_name).title(),
            "current_station": str(current_stn).title(),
            "delay_minutes": delay_min,
            "status": status_raw or ("On time" if delay_min == 0 else f"{delay_min} min late"),
            "fetched_at": datetime.now().isoformat(),
            "from_cache": False,
        }
    except Exception as e:
        logger.info(f"[NTES] JSON parse error: {e}")
        return None


def _parse_ntes_text(train_no: str, text: str) -> Optional[dict]:
    """Parse NTES HTML response using BeautifulSoup to extract live running status."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        clean_lines = [line.strip() for line in soup.get_text().splitlines() if line.strip()]
        if not clean_lines:
            return None

        # 1. Extract Train Name from first few lines (e.g., "17225 - AMARAVATHI EXP")
        train_name = ""
        for line in clean_lines[:10]:
            if train_no in line and "-" in line:
                train_name = line.replace(train_no, "").replace("-", "").strip()
                break

        # 2. Extract Current Position and Delay from text
        # Look for "Departed from" or "Arrived at" or "Expected arrival at"
        # E.g. "Departed from MARKAPUR ROAD(MRK) at 23:56 09-Jul (Delay: 00:45)"
        status_phrase = ""
        current_station = ""
        delay_minutes = 0

        # Try to find the exact status line (often line 6 or matching prefix)
        status_prefixes = ("departed from", "arrived at", "expected arrival at", "expected departure at")
        for line in clean_lines:
            line_lower = line.lower()
            if any(line_lower.startswith(prefix) for prefix in status_prefixes) or "delay" in line_lower:
                status_phrase = line
                break

        # Fallback if no prefix matches but we see a line starting with "Departed"
        if not status_phrase:
            for line in clean_lines:
                if "departed from" in line.lower() or "arrived at" in line.lower():
                    status_phrase = line
                    break

        if status_phrase:
            # Parse delay minutes (e.g. "Delay: 00:45" or "Delay: 45 Min" or "Delay: Delay 00:45")
            delay_match = re.search(r"Delay:\s*(\d{2}):(\d{2})", status_phrase, re.IGNORECASE)
            if not delay_match:
                delay_match = re.search(r"Delay[- ]+Delay\s*(\d{2}):(\d{2})", status_phrase, re.IGNORECASE)
            if not delay_match:
                delay_match = re.search(r"Delay:\s*(\d+)\s*Min", status_phrase, re.IGNORECASE)
                
            if delay_match:
                if len(delay_match.groups()) == 2:
                    hours = int(delay_match.group(1))
                    mins = int(delay_match.group(2))
                    delay_minutes = hours * 60 + mins
                else:
                    delay_minutes = int(delay_match.group(1))
            elif "on time" in status_phrase.lower() or "right time" in status_phrase.lower():
                delay_minutes = 0
            
            # Extract current station name
            # E.g. "Departed from MARKAPUR ROAD(MRK) at 23:56" -> "MARKAPUR ROAD"
            stn_match = re.search(r"(?:Departed from|Arrived at|Expected arrival at|Expected departure at)\s+([^(]+)", status_phrase, re.IGNORECASE)
            if stn_match:
                current_station = stn_match.group(1).strip()
        else:
            status_phrase = "Running status parsed successfully."

        # Parse station timeline with platform and delay details
        stations_data = []
        for outer_div in soup.find_all("div", style=lambda s: s and "display:flex" in s):
            pf_span = outer_div.find("span", class_="w3-orange")
            if not pf_span:
                continue
                
            left_col = outer_div.find("div", style=lambda s: s and "float:left" in s)
            if not left_col:
                continue
                
            b_tag = left_col.find("b")
            if not b_tag:
                continue
            station_name = b_tag.get_text().strip()
            platform = pf_span.get_text().strip()
            
            container_div = pf_span.find_parent("div")
            station_code = ""
            if container_div:
                parts = container_div.get_text().strip().split()
                if parts:
                    station_code = parts[0]
                    
            right_col = outer_div.find("div", style=lambda s: s and "float:right" in s)
            scheduled_time = ""
            actual_time = ""
            delay_text = "On Time"
            
            if right_col:
                spans = right_col.find_all("span")
                if len(spans) >= 1:
                    scheduled_time = spans[0].get_text().strip()
                if len(spans) >= 2:
                    actual_text = spans[1].get_text().strip()
                    delay_badge = spans[1].find("span", class_="w3-round")
                    if delay_badge:
                        delay_text = delay_badge.get_text().strip()
                        actual_time = actual_text.replace(delay_text, "").strip()
                    else:
                        actual_time = actual_text
                        if "on time" in actual_text.lower() or "right time" in actual_text.lower():
                            delay_text = "On Time"
                        else:
                            delay_text = "On Time"
                            
            stations_data.append({
                "name": station_name,
                "code": station_code,
                "platform": platform,
                "scheduled_time": scheduled_time,
                "actual_time": actual_time,
                "delay": delay_text
            })

        return {
            "success": True,
            "source": "NTES",
            "train_no": train_no,
            "train_name": train_name or "Express Train",
            "current_station": current_station or "Station Info Loaded",
            "delay_minutes": delay_minutes,
            "status": status_phrase,
            "stations_timeline": stations_data,
            "fetched_at": datetime.now().isoformat(),
            "from_cache": False,
        }
    except Exception as e:
        logger.info(f"[NTES] HTML text parse error: {e}")
        return None


def _parse_erail_response(train_no: str, text: str) -> Optional[dict]:
    """Parse erail.in response — provides schedule info at minimum."""
    # erail returns pipe/comma separated data — extract what we can
    try:
        lines = text.strip().split("\n")
        if not lines or not lines[0]:
            return None

        return {
            "success": True,
            "source": "erail.in",
            "train_no": train_no,
            "train_name": "",
            "current_station": "",  # erail gives schedule, not live position
            "delay_minutes": 0,
            "status": "Schedule data only (no live position from erail.in)",
            "raw_schedule": text[:300],
            "fetched_at": datetime.now().isoformat(),
            "from_cache": False,
        }
    except Exception as e:
        logger.info(f"[NTES] erail parse error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────

def _parse_delay(value) -> int:
    """Parse delay from various formats → integer minutes."""
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in ("", "0", "on time", "right time"):
            return 0
        match = re.search(r"(\d+)", value)
        if match:
            return int(match.group(1))
    return 0


def format_live_status_for_llm(status: dict) -> str:
    """
    Format NTES status dict into a readable string for the LLM context.
    This is injected into the Gemini prompt as live data context.
    """
    if not status.get("success"):
        return (
            f"⚠️ LIVE DATA UNAVAILABLE for train {status.get('train_no', '?')}\n"
            f"Reason: {status.get('error', 'Unknown error')}\n"
            f"Please use the scheduled timetable data below (labeled STATIC)."
        )

    lines = [
        f"=== LIVE DATA (Source: {status.get('source', 'NTES')} | "
        f"Fetched: {status.get('fetched_at', '')[:16]} IST) ===",
    ]
    if status.get("from_cache"):
        lines[0] += f" [Cached — {status.get('cache_age_seconds', '?')}s ago]"

    if status.get("train_name"):
        lines.append(f"Train: {status['train_no']} — {status['train_name']}")
    else:
        lines.append(f"Train: {status['train_no']}")

    if status.get("current_station"):
        lines.append(f"Current Location: {status['current_station']}")

    delay = status.get("delay_minutes", 0)
    if delay == 0:
        lines.append("Running Status: ON TIME ✅")
    else:
        lines.append(f"Running Status: {delay} MINUTES LATE ⚠️")

    if status.get("status"):
        lines.append(f"Status Detail: {status['status']}")

    if status.get("stations_timeline"):
        lines.append("\nLive Station-wise Platforms, Delays and Schedule Info:")
        for stn in status["stations_timeline"]:
            lines.append(
                f"  - {stn['name']} ({stn['code']}): {stn['platform']} | "
                f"Scheduled: {stn['scheduled_time']} | Expected/Actual: {stn['actual_time']} | "
                f"Delay: {stn['delay']}"
            )

    lines.append("=" * 50)
    return "\n".join(lines)
