"""
ntes_client.py — Live Train Status Client

Priority order on cloud (Render, Heroku, etc.):
  1. RapidAPI Indian Railways (works from all IPs — cloud-friendly)
  2. erail.in (lightweight fallback)
  3. NTES (only attempted locally — blocked on cloud IPs)

Priority order locally:
  1. NTES (direct, no API key needed)
  2. erail.in
  3. RapidAPI (if key is set)

Set RAPIDAPI_KEY in your .env / Render env vars to unlock cloud live tracking.
"""

import os
import re
import requests
import time
from datetime import datetime, timedelta
from typing import Optional

from app.logger import get_logger
logger = get_logger("app.ntes_client")

# ── In-memory cache (per server session, 5-min TTL) ──────────────
_live_cache: dict[str, dict] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes

# ── HTTP config ───────────────────────────────────────────────────
REQUEST_TIMEOUT = (8, 15)   # Reduced: (connect, read) — fail fast on cloud blocks
NTES_TIMEOUT    = (5, 10)   # Even tighter for NTES (known to block cloud IPs)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Referer": "https://enquiry.indianrail.gov.in/mntes",
}

# ── RapidAPI config ───────────────────────────────────────────────
# API: IRCTC | INDIAN RAILWAY PNR STATUS (irctc-indian-railway-pnr-status.p.rapidapi.com)
# Endpoint: GET /live-train/{trainNo}/status
# Params: startDay (0=today, 1=yesterday), from (station code), date (DD-MMM-YYYY)
RAPIDAPI_KEY  = os.getenv("RAPIDAPI_KEY", "").strip()   # strip() removes accidental trailing \n from Render env var paste

RAPIDAPI_HOST = "irctc-indian-railway-pnr-status.p.rapidapi.com"
RAPIDAPI_BASE = f"https://{RAPIDAPI_HOST}"

ERAIL_TRAIN_ENDPOINT = "https://erail.in/rail/getTrains.aspx"

# ── NTES endpoints ────────────────────────────────────────────────
NTES_ENDPOINTS = [
    "https://enquiry.indianrail.gov.in/NTES/GetTrainRunningStatus",
    "https://enquiry.indianrail.gov.in/mntes/GetTrainRunningStatus",
]


# ─────────────────────────────────────────────────────────────────
# CLOUD DETECTION
# ─────────────────────────────────────────────────────────────────

def _is_cloud_deployment() -> bool:
    """
    Detect if running on a cloud server (Render, Heroku, Railway, etc.).
    On cloud servers, NTES is blocked — skip it to save time.
    """
    cloud_env_vars = ["RENDER", "HEROKU_APP_NAME", "RAILWAY_ENVIRONMENT",
                      "FLY_APP_NAME", "PORT"]
    return any(os.getenv(v) for v in cloud_env_vars)


_IS_CLOUD = _is_cloud_deployment()


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
        source        : str ("RapidAPI" / "NTES" / "erail.in" / "cache")
        fetched_at    : str (ISO timestamp)
        error         : str (only if success=False)
        from_cache    : bool
    """
    train_no = str(train_no).strip()
    logger.info(f"[NTES] Request for train {train_no} (cloud={_IS_CLOUD})")

    # 1. Cache hit
    cached = _get_from_cache(train_no)
    if cached:
        logger.info(f"[NTES] Cache hit — age {cached.get('cache_age_seconds', '?')}s")
        return cached

    # 2. RapidAPI — works from cloud servers (primary on cloud, tertiary locally)
    if RAPIDAPI_KEY:
        result = _fetch_rapidapi(train_no)
        if result and result.get("success"):
            _set_cache(train_no, result)
            return result

    # 3. NTES — only try locally (cloud IPs are blocked, wastes 15–25s per attempt)
    if not _IS_CLOUD:
        for endpoint in NTES_ENDPOINTS:
            result = _fetch_ntes(train_no, endpoint)
            if result and result.get("success"):
                _set_cache(train_no, result)
                return result
    else:
        logger.info("[NTES] Skipping NTES on cloud deployment (IPs blocked by indianrail.gov.in)")

    # 4. Fallback: erail.in (lightweight, usually works)
    result = _fetch_erail(train_no)
    if result and result.get("success"):
        _set_cache(train_no, result)
        return result

    # 5. All sources failed
    no_rapidapi_hint = ""
    if not RAPIDAPI_KEY:
        no_rapidapi_hint = " Set RAPIDAPI_KEY in Render env vars for reliable cloud live tracking."
    logger.info(f"[NTES] All sources failed for train {train_no}")
    return {
        "success": False,
        "train_no": train_no,
        "error": f"Live data unavailable — all sources failed or timed out.{no_rapidapi_hint}",
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

    if cache_key in _live_cache and datetime.now() < _live_cache[cache_key]["expires_at"]:
        return {**_live_cache[cache_key]["data"], "from_cache": True}

    # Try RapidAPI station board first
    if RAPIDAPI_KEY:
        try:
            resp = requests.get(
                f"{RAPIDAPI_BASE}/liveAt/{station_code}",
                headers={
                    "x-rapidapi-key": RAPIDAPI_KEY,
                    "x-rapidapi-host": RAPIDAPI_HOST,
                },
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 200:
                data = resp.json()
                trains = data.get("body", data) if isinstance(data, dict) else data
                result = {
                    "success": True,
                    "station_code": station_code,
                    "board_type": board_type,
                    "trains": trains if isinstance(trains, list) else [],
                    "source": "RapidAPI",
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
            logger.info(f"[NTES] RapidAPI station board failed for {station_code}: {e}")

    # Fallback: NTES station board (local only)
    if not _IS_CLOUD:
        try:
            resp = requests.get(
                "https://enquiry.indianrail.gov.in/NTES/GetArrivalDeparture",
                params={"stnCode": station_code, "type": board_type},
                headers=HEADERS,
                timeout=NTES_TIMEOUT,
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

def _fetch_rapidapi(train_no: str) -> Optional[dict]:
    """
    Fetch live running status from RapidAPI.
    API: IRCTC | INDIAN RAILWAY PNR STATUS
    Host: irctc-indian-railway-pnr-status.p.rapidapi.com
    Endpoint: GET /live-train/{trainNo}/status
    Params: startDay (0=today, 1=yesterday)
    Works from cloud server IPs (Render, Heroku, etc.).
    """
    if not RAPIDAPI_KEY:
        return None

    # Try today first, then yesterday (train may have started previous day)
    for start_day in [0, 1]:
        try:
            url = f"{RAPIDAPI_BASE}/live-train/{train_no}/status"
            resp = requests.get(
                url,
                params={"startDay": start_day},
                headers={
                    "Content-Type": "application/json",
                    "x-rapidapi-key": RAPIDAPI_KEY,
                    "x-rapidapi-host": RAPIDAPI_HOST,
                },
                timeout=REQUEST_TIMEOUT,
            )
            logger.info(f"[NTES] RapidAPI → HTTP {resp.status_code} (startDay={start_day})")

            if resp.status_code == 200:
                data = resp.json()
                # This API returns: {"success": true, "data": {...}}
                if not data.get("success"):
                    logger.info(f"[NTES] RapidAPI returned success=false: {data.get('message', '')}")
                    continue

                body = data.get("data", data)
                if not body:
                    continue

                result = _parse_rapidapi_response(train_no, body)
                if result:
                    return result

            elif resp.status_code in (404, 400, 204):
                continue  # try next startDay
            else:
                logger.info(f"[NTES] RapidAPI unexpected HTTP {resp.status_code}")
                break

        except Exception as e:
            logger.info(f"[NTES] RapidAPI error: {e}")
            break

    return None


def _parse_rapidapi_response(train_no: str, body: dict) -> Optional[dict]:
    """
    Parse the IRCTC Indian Railway PNR Status API response.

    Expected format:
    {
      "train_no": "17225",
      "train_name": "AMARAVATHI EXP",
      "position": "Departed from GUNDALUKAMMA(GKM) at 22:54 26-Jul",
      "current_station": "GUNDALUKAMMA",
      "delay_minutes": 0,
      "progress_percent": 32,
      "stations": [
        {
          "station_name": "NARASAPUR",
          "platform": "1",
          "arrival": "First",
          "arrival_delay_minutes": 0,
          "departure": "16:21",
          "departure_delay_minutes": 1,
          "status": "departed"
        }, ...
      ],
      "running_instances": [...]
    }
    """
    try:
        if not isinstance(body, dict):
            return None

        train_name  = body.get("train_name", "")
        current_stn = body.get("current_station", "")
        position    = body.get("position", "")       # e.g. "Departed from GKM at 22:54"
        delay_min   = int(body.get("delay_minutes") or 0)
        progress    = body.get("progress_percent", 0)

        # Compute max delay from all departed stations (delay_minutes field = 0 often means
        # the overall journey is on-time but individual stops may have delays)
        stations_raw = body.get("stations", [])
        if delay_min == 0 and isinstance(stations_raw, list):
            departed_delays = [
                stn.get("departure_delay_minutes", 0) or 0
                for stn in stations_raw
                if stn.get("status") == "departed"
            ]
            if departed_delays:
                delay_min = departed_delays[-1]  # most recent departed station delay

        # Build clean stations timeline
        stations_timeline = []
        for stn in (stations_raw if isinstance(stations_raw, list) else []):
            if not isinstance(stn, dict):
                continue
            arr_delay = stn.get("arrival_delay_minutes") or 0
            dep_delay = stn.get("departure_delay_minutes") or 0
            delay_str = "On Time"
            if dep_delay > 0:
                delay_str = f"{dep_delay} min late"
            elif arr_delay > 0:
                delay_str = f"{arr_delay} min late"

            stations_timeline.append({
                "name":           stn.get("station_name", ""),
                "code":           "",   # this API doesn't return station codes in list
                "platform":       str(stn.get("platform") or ""),
                "scheduled_time": stn.get("arrival", "") if stn.get("arrival") != "First" else "origin",
                "actual_time":    stn.get("departure", "") if stn.get("departure") != "Last" else "terminus",
                "delay":          delay_str,
                "status":         stn.get("status", ""),  # "departed" / "upcoming" / "current"
            })

        # Build human-readable status string from position field
        if position:
            status_str = position
        elif delay_min == 0:
            status_str = "Running on time"
        else:
            status_str = f"{delay_min} min late"

        return {
            "success":           True,
            "source":            "RapidAPI",
            "train_no":          train_no,
            "train_name":        str(train_name).title() if train_name else "",
            "current_station":   str(current_stn).title() if current_stn else "",
            "delay_minutes":     delay_min,
            "status":            status_str,
            "progress_percent":  progress,
            "stations_timeline": stations_timeline,
            "fetched_at":        datetime.now().isoformat(),
            "from_cache":        False,
        }
    except Exception as e:
        logger.info(f"[NTES] RapidAPI parse error: {e}")
        return None


def _fetch_ntes(train_no: str, endpoint: str) -> Optional[dict]:
    """Fetch from NTES — local only (cloud IPs are blocked by indianrail.gov.in)."""
    try:
        session = requests.Session()
        session.headers.update(HEADERS)

        home_url = "https://enquiry.indianrail.gov.in/mntes/"
        session.get(home_url, timeout=NTES_TIMEOUT)

        csrf_url = f"https://enquiry.indianrail.gov.in/mntes/GetCSRFToken?t={int(time.time() * 1000)}"
        r_csrf = session.get(csrf_url, timeout=NTES_TIMEOUT)

        csrf_data = {}
        if r_csrf.status_code == 200:
            match = re.search(r"name='([^']+)'\s+value='([^']+)'", r_csrf.text)
            if not match:
                match = re.search(r'name="([^"]+)"\s+value="([^"]+)"', r_csrf.text)
            if match:
                csrf_data[match.group(1)] = match.group(2)

        j_date = datetime.now().strftime("%d-%b-%Y")
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

        resp = session.post(tr_url, data=params, timeout=NTES_TIMEOUT)
        logger.info(f"[NTES] tr POST status: {resp.status_code} | length: {len(resp.text)}")

        if resp.status_code == 200 and len(resp.text) > 100:
            result = _parse_ntes_text(train_no, resp.text)
            if result and (not result.get("current_station") or result.get("current_station") == "Station Info Loaded"):
                yesterday = datetime.now() - timedelta(days=1)
                y_date = yesterday.strftime("%d-%b-%Y")
                logger.info(f"[NTES] Trying yesterday's date ({y_date})...")

                r_csrf_y = session.get(csrf_url, timeout=NTES_TIMEOUT)
                csrf_data_y = {}
                if r_csrf_y.status_code == 200:
                    match_y = re.search(r"name='([^']+)'\s+value='([^']+)'", r_csrf_y.text)
                    if not match_y:
                        match_y = re.search(r'name="([^"]+)"\s+value="([^"]+)"', r_csrf_y.text)
                    if match_y:
                        csrf_data_y[match_y.group(1)] = match_y.group(2)

                params_y = {**params, "jDate": y_date}
                params_y.update(csrf_data_y)
                resp_y = session.post(tr_url, data=params_y, timeout=NTES_TIMEOUT)
                if resp_y.status_code == 200 and len(resp_y.text) > 100:
                    result_y = _parse_ntes_text(train_no, resp_y.text)
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

def _parse_ntes_text(train_no: str, text: str) -> Optional[dict]:
    """Parse NTES HTML response using BeautifulSoup to extract live running status."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, "html.parser")

        for script in soup(["script", "style"]):
            script.extract()

        clean_lines = [line.strip() for line in soup.get_text().splitlines() if line.strip()]
        if not clean_lines:
            return None

        train_name = ""
        for line in clean_lines[:10]:
            if train_no in line and "-" in line:
                train_name = line.replace(train_no, "").replace("-", "").strip()
                break

        status_phrase = ""
        current_station = ""
        delay_minutes = 0

        status_prefixes = ("departed from", "arrived at", "expected arrival at", "expected departure at")
        for line in clean_lines:
            line_lower = line.lower()
            if any(line_lower.startswith(prefix) for prefix in status_prefixes) or "delay" in line_lower:
                status_phrase = line
                break

        if not status_phrase:
            for line in clean_lines:
                if "departed from" in line.lower() or "arrived at" in line.lower():
                    status_phrase = line
                    break

        if status_phrase:
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

            stn_match = re.search(r"(?:Departed from|Arrived at|Expected arrival at|Expected departure at)\s+([^(]+)", status_phrase, re.IGNORECASE)
            if stn_match:
                current_station = stn_match.group(1).strip()
        else:
            status_phrase = "Running status parsed successfully."

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
        hint = ""
        if "RAPIDAPI_KEY" in status.get("error", ""):
            hint = "\nTip: Ask the admin to set RAPIDAPI_KEY in Render environment variables."
        return (
            f"⚠️ LIVE DATA UNAVAILABLE for train {status.get('train_no', '?')}\n"
            f"Reason: {status.get('error', 'Unknown error')}{hint}\n"
            f"Please use the scheduled timetable data below (labeled STATIC)."
        )

    current_stn = status.get("current_station", "").strip()
    has_live_tracking = bool(current_stn and current_stn not in ("Station Info Loaded", ""))

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

    if has_live_tracking:
        lines.append(f"Current Location: {current_stn}")
        delay = status.get("delay_minutes", 0)
        if delay == 0:
            lines.append("Running Status: ON TIME ✅")
        else:
            lines.append(f"Running Status: {delay} MINUTES LATE ⚠️")
    else:
        lines.append("Running Status: ⚠️ REAL-TIME GPS/LOCATION DATA UNAVAILABLE FOR THIS TRAIN")
        lines.append("Note: Public APIs returned schedule info only; live GPS tracking was not reported.")

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
