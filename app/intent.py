"""
intent.py — Query Intent Classifier v2.0

Classifies user queries into coarse + fine-grained intents.

Coarse intents (backward-compatible with main.py routing):
  STATIC       — only ChromaDB static data needed
  LIVE         — only real-time API needed
  HYBRID       — both ChromaDB + live API needed
  OUT_OF_DOMAIN — not a railway query

Fine-grained intent_category (for smarter routing):
  LIVE_STATUS         — current position / running status of a specific train
  PNR_STATUS          — PNR number lookup
  CANCELLATION_TODAY  — is a train cancelled / diverted today? (live)
  PLATFORM_TODAY      — which platform is the train arriving on today? (live)
  SCHEDULE_QUERY      — timetable: arrival / departure times (static DB)
  BETWEEN_STATIONS    — trains running from A to B (static DB)
  STATION_INFO        — station details, code, zone, division (static DB)
  CANCELLATION_RULES  — cancellation charges / refund policy (static DB)
  COACH_QUERY         — coach positions / train composition (static DB)
  GENERAL_INFO        — fares, luggage, rules, TTE duties, etc. (static DB)
  HYBRID              — needs both live + static
  OUT_OF_DOMAIN       — not a railway query
"""

import re
import threading
from typing import Optional

# ── Fine-grained intent category constants ──────────────────────────
INTENT_LIVE_STATUS        = "LIVE_STATUS"
INTENT_PNR_STATUS         = "PNR_STATUS"
INTENT_CANCELLATION_TODAY = "CANCELLATION_TODAY"
INTENT_PLATFORM_TODAY     = "PLATFORM_TODAY"
INTENT_SCHEDULE_QUERY     = "SCHEDULE_QUERY"
INTENT_BETWEEN_STATIONS   = "BETWEEN_STATIONS"
INTENT_STATION_INFO       = "STATION_INFO"
INTENT_CANCELLATION_RULES = "CANCELLATION_RULES"
INTENT_COACH_QUERY        = "COACH_QUERY"
INTENT_GENERAL_INFO       = "GENERAL_INFO"
INTENT_HYBRID             = "HYBRID"
INTENT_OUT_OF_DOMAIN      = "OUT_OF_DOMAIN"

# Maps fine-grained category → coarse intent (for main.py backward compat)
_COARSE: dict[str, str] = {
    INTENT_LIVE_STATUS:        "LIVE",
    INTENT_PNR_STATUS:         "LIVE",
    INTENT_CANCELLATION_TODAY: "LIVE",
    INTENT_PLATFORM_TODAY:     "LIVE",
    INTENT_SCHEDULE_QUERY:     "STATIC",
    INTENT_BETWEEN_STATIONS:   "STATIC",
    INTENT_STATION_INFO:       "STATIC",
    INTENT_CANCELLATION_RULES: "STATIC",
    INTENT_COACH_QUERY:        "STATIC",
    INTENT_GENERAL_INFO:       "STATIC",
    INTENT_HYBRID:             "HYBRID",
    INTENT_OUT_OF_DOMAIN:      "OUT_OF_DOMAIN",
}

# ── Phrase-level patterns (evaluated in ORDER — FIRST MATCH WINS) ───
# Format: (compiled_regex, intent_category, confidence)
# More specific / higher-priority patterns come FIRST.
_PATTERNS: list[tuple[re.Pattern, str, float]] = []

def _p(pattern: str, category: str, confidence: float = 0.95) -> None:
    """Register a compiled pattern."""
    _PATTERNS.append((re.compile(pattern, re.IGNORECASE), category, confidence))


# ── 0. Self-introduction / meta-questions (must come FIRST — these have NO
#       railway keywords so would otherwise fall into OUT_OF_DOMAIN) ──────
_p(r'\b(tell\s+me\s+about\s+yourself|who\s+are\s+you|what\s+(can|do)\s+you\s+do'
   r'|introduce\s+yourself|your\s+capabilities?|what\s+is\s+railgpt'
   r'|about\s+railgpt|what\s+can\s+railgpt\s+do|what\s+are\s+you'
   r'|how\s+can\s+you\s+help|what\s+do\s+you\s+know)\b',
   INTENT_GENERAL_INFO, 0.90)

# ── 1. PNR (highest priority) ───────────────────────────────────────
_p(r'\b\d{10}\b',                                           INTENT_PNR_STATUS, 1.00)
_p(r'\bpnr\b',                                              INTENT_PNR_STATUS, 0.99)

# ── 2. Live cancellation / diversion today ──────────────────────────
_p(r'\b(is|has)\b.{0,40}\b(cancelled|canceled|suspended|diverted)\b.{0,25}\b(today|now|tonight|currently)\b',
   INTENT_CANCELLATION_TODAY, 0.95)
_p(r'\b(cancelled|canceled|suspended|diverted)\b.{0,25}\b(today|now|tonight)\b',
   INTENT_CANCELLATION_TODAY, 0.92)
_p(r'\b(is|has)\b.{0,20}\btrain\b.{0,25}\b(cancelled|diverted|not running)\b',
   INTENT_CANCELLATION_TODAY, 0.88)

# ── 3. Platform today (live) ────────────────────────────────────────
_p(r'\b(which|what)\b.{0,12}\bplatform\b.{0,35}\b(today|now|tonight|coming|arriving)\b',
   INTENT_PLATFORM_TODAY, 0.92)
_p(r'\bplatform\b.{0,20}\b(today|now|tonight)\b',
   INTENT_PLATFORM_TODAY, 0.88)

# ── 4. Live running status (current position, delay, real-time) ─────
# "where is train 12727" / "where is 12727"
_p(r'\b(where\s+is|where\'s)\b.{0,35}\b\d{4,5}\b',          INTENT_LIVE_STATUS, 0.98)
# "running status of 12727" / "live status of 12727"
_p(r'\b(running|live)\s+status\b.{0,30}\b\d{4,5}\b',         INTENT_LIVE_STATUS, 0.97)
_p(r'\b\d{4,5}\b.{0,20}\b(running|live)\s+status\b',         INTENT_LIVE_STATUS, 0.97)
# "spot train 12727" / "spot 12727"
_p(r'\bspot\b.{0,25}\b(train\b.{0,12})?\d{4,5}\b',           INTENT_LIVE_STATUS, 0.96)
# "is 12727 running / on time / delayed / late"
_p(r'\b(is|has)\b.{0,25}\b\d{4,5}\b.{0,25}\b(running|on time|delayed|late|left|departed|started|reached)\b',
   INTENT_LIVE_STATUS, 0.93)
# "12727 has arrived / left / reached"
_p(r'\b\d{4,5}\b.{0,25}\b(has|have)\b.{0,12}\b(arrived|reached|passed|left|departed|crossed)\b',
   INTENT_LIVE_STATUS, 0.93)
# "current position / location of 12727"
_p(r'\b(current|real.?time)\b.{0,18}\b(position|location|station|status)\b.{0,25}\b\d{4,5}\b',
   INTENT_LIVE_STATUS, 0.93)
_p(r'\b\d{4,5}\b.{0,25}\b(current|real.?time)\b.{0,18}\b(position|location)\b',
   INTENT_LIVE_STATUS, 0.92)
# "12727 running late today" / "12727 delayed"
_p(r'\b\d{4,5}\b.{0,25}\b(delayed|late|on time|running)\b.{0,15}\b(today|now|tonight|currently)?\b',
   INTENT_LIVE_STATUS, 0.88)
# "eta of 12727" / "expected arrival of 12727"
_p(r'\b(eta|expected arrival|expected at)\b.{0,35}\b\d{4,5}\b', INTENT_LIVE_STATUS, 0.90)
_p(r'\b\d{4,5}\b.{0,25}\b(eta|expected arrival)\b',            INTENT_LIVE_STATUS, 0.88)

# ── 5. Schedule / timetable (STATIC — arrival/departure TIME) ───────
# CRITICAL FIX: "what time does/will X arrive/reach/depart at Y?" → STATIC
# These patterns MUST come BEFORE keyword fallback picks up "arrive" as LIVE.

# "what time does/will 12727 arrive/reach/depart/stop at BZA?"
_p(r'\b(what time|when|timing)\b.{0,35}\b\d{4,5}\b.{0,40}\b(arrive|reach|depart|leave|stop|pass|start)\b',
   INTENT_SCHEDULE_QUERY, 0.97)
_p(r'\b(what time|timing)\b.{0,35}\b(arrive|reach|depart|leave|stop)\b.{0,35}\b\d{4,5}\b',
   INTENT_SCHEDULE_QUERY, 0.96)
# "arrival / departure time of 12727 at BZA"
_p(r'\b(arrival|departure)\s+time\b.{0,35}\b\d{4,5}\b',       INTENT_SCHEDULE_QUERY, 0.97)
_p(r'\b\d{4,5}\b.{0,25}\b(arrival|departure)\s+time\b',        INTENT_SCHEDULE_QUERY, 0.96)
# "when does/will 12727 arrive/reach/depart/stop?"
_p(r'\b(when does|when will|what time does|what time will)\b.{0,25}\b\d{4,5}\b',
   INTENT_SCHEDULE_QUERY, 0.96)
# "stops of 12727" / "schedule of 12727" / "route of 12727" / "timetable of 12727"
_p(r'\b(stops|schedule|timetable|route|halts?|stations?)\b.{0,12}\b(of|for)\b.{0,25}\b\d{4,5}\b',
   INTENT_SCHEDULE_QUERY, 0.98)
_p(r'\b\d{4,5}\b.{0,18}\b(stops|schedule|timetable|route|halts?)\b',
   INTENT_SCHEDULE_QUERY, 0.97)
# "how long does 12727 halt at BZA?"
_p(r'\b(how long|halt|halting)\b.{0,15}\b(at|in|for)\b.{0,35}\b\d{4,5}\b',
   INTENT_SCHEDULE_QUERY, 0.92)
_p(r'\b\d{4,5}\b.{0,25}\b(halt|halts|halting|stops for)\b',   INTENT_SCHEDULE_QUERY, 0.91)
# "12727 timing at BZA" / "timing of 12727"
_p(r'\b\d{4,5}\b.{0,25}\btiming\b',                           INTENT_SCHEDULE_QUERY, 0.90)
_p(r'\btiming\b.{0,25}\b\d{4,5}\b',                           INTENT_SCHEDULE_QUERY, 0.90)
# "what time does the train start from VSKP?" (no train number)
_p(r'\b(what time|when)\b.{0,25}\b(start|depart|leave|arrive)\b.{0,25}\bfrom\b',
   INTENT_SCHEDULE_QUERY, 0.88)
# "12727 BZA arrival" / "12727 arrives at"  (habitual/schedule sense)
_p(r'\b\d{4,5}\b.{0,25}\b(arrives?|departs?|reaches?)\b.{0,10}\bat\b',
   INTENT_SCHEDULE_QUERY, 0.87)
# "day 2 arrival at" / "which day does 12727 reach Delhi?"
_p(r'\b(which day|day \d)\b.{0,25}\b\d{4,5}\b',               INTENT_SCHEDULE_QUERY, 0.88)
_p(r'\b\d{4,5}\b.{0,25}\b(which day|day \d)\b',               INTENT_SCHEDULE_QUERY, 0.88)

# ── 6. Trains between two stations ─────────────────────────────────
_p(r'\btrains?\b.{0,18}\b(from|between)\b.{1,50}\b(to|and)\b', INTENT_BETWEEN_STATIONS, 0.95)
_p(r'\b(which|what|list)\s+trains?\b.{0,25}\b(go|run|pass|connect|available)\b.{0,30}\b(from|between|via)\b',
   INTENT_BETWEEN_STATIONS, 0.94)
_p(r'\b(direct|connecting)\s+trains?\b.{0,25}\b(from|between)\b', INTENT_BETWEEN_STATIONS, 0.93)
_p(r'\bhow\s+to\s+(go|travel|reach)\b.{0,35}\b(from|to)\b',   INTENT_BETWEEN_STATIONS, 0.88)
_p(r'\b(from|between)\b.{1,35}\bto\b.{1,35}\b(trains?|service|connection)\b',
   INTENT_BETWEEN_STATIONS, 0.87)

# ── 7. Station information ──────────────────────────────────────────
_p(r'\b(about|details?|info(rmation)?)\b.{0,18}\bstation\b',   INTENT_STATION_INFO, 0.91)
_p(r'\bstation\s+(code|zone|division|region)\b.{0,25}\bof\b',  INTENT_STATION_INFO, 0.93)
_p(r'\b(zone|division)\b.{0,18}\bstation\b',                   INTENT_STATION_INFO, 0.89)
_p(r'\bwhat\s+is\b.{0,25}\b(code|zone)\b.{0,25}\bstation\b',  INTENT_STATION_INFO, 0.88)

# ── 8. Cancellation charges / refund rules (static) ─────────────────
_p(r'\b(cancellation|refund)\s+(charges?|fee|fees|rules?|policy|amount|percentage)\b',
   INTENT_CANCELLATION_RULES, 0.97)
_p(r'\bhow\s+much\b.{0,25}\b(refund|deducted|charged|penalty)\b.{0,25}\b(cancel|cancellation)\b',
   INTENT_CANCELLATION_RULES, 0.96)
_p(r'\b(tdr|ticket\s+deposit\s+receipt)\b',                    INTENT_CANCELLATION_RULES, 0.93)
_p(r'\b(cancel|cancellation)\b.{0,25}\b(policy|rule|how|procedure|process)\b',
   INTENT_CANCELLATION_RULES, 0.90)

# ── 9. Coach query ──────────────────────────────────────────────────
_p(r'\b(which|where)\b.{0,18}\b(coach|bogie|compartment)\b',  INTENT_COACH_QUERY, 0.92)
_p(r'\b(position|order|sequence|arrangement)\b.{0,18}\b(coach|bogie)\b',
   INTENT_COACH_QUERY, 0.91)
_p(r'\bcoach\b.{0,12}\b[SsBbAaCcDdEeHhPpJjFf]\d+\b',         INTENT_COACH_QUERY, 0.94)
_p(r'\b(composition|makeup|formation)\b.{0,18}\b(of|for)?\b.{0,25}\btrain\b',
   INTENT_COACH_QUERY, 0.89)

# ── 10. General railway rules / info (static) ───────────────────────
_p(r'\b(luggage|baggage)\b.{0,18}\b(limit|allowance|rules?|permitted|allowed|weight)\b',
   INTENT_GENERAL_INFO, 0.94)
_p(r'\b(tte|ticket\s+collector|ticket\s+examiner)\b.{0,25}\b(duties?|responsibilities?|power|fine|role)\b',
   INTENT_GENERAL_INFO, 0.93)
_p(r'\b(tatkal|premium\s+tatkal|pqwl|gnwl|rswl|rac|ckwl)\b.{0,25}\b(rules?|charges?|how|booking|quota)\b',
   INTENT_GENERAL_INFO, 0.92)
_p(r'\b(senior\s+citizen|handicapped|disabled|ex.?serviceman)\b.{0,25}\b(concession|discount|quota|reservation)\b',
   INTENT_GENERAL_INFO, 0.92)
_p(r'\b(fare|ticket\s+price|cost|charge)\b.{0,25}\b(from|to|between|for|in|class)\b',
   INTENT_GENERAL_INFO, 0.90)
_p(r'\b(sleeper|ac\s+\d?|first\s+class|second\s+class)\b.{0,20}\b(rules?|limit|weight|allowance|fare|price)\b',
   INTENT_GENERAL_INFO, 0.89)
_p(r'\b(rac|wl|waiting\s+list|confirm(ed)?|cnf)\b.{0,20}\b(meaning|what|rules?|policy|status)\b',
   INTENT_GENERAL_INFO, 0.89)
_p(r'\bwhat\s+is\b.{0,10}\b(rac|gnwl|pqwl|rswl|ckwl|cnf|tdr|tatkal)\b',
   INTENT_GENERAL_INFO, 0.90)


# ── Keyword fallback sets (used only when no pattern matches) ───────
# IMPORTANT: "arrive", "departure", "reach", "run", "when", "start" are
# NOT in LIVE_KEYWORDS anymore — they are schedule words.

_LIVE_KEYWORDS_FALLBACK = frozenset([
    "live", "running", "status", "spot",
    "late", "delay", "delayed",
    "track", "location", "current", "position",
    "cancelled", "canceled", "suspended",
    "is it on time", "where is", "expected arrival",
    "commenced", "commencing",
])

_SCHEDULE_KEYWORDS_FALLBACK = frozenset([
    "arrive", "arrives", "arrival", "arriving",
    "depart", "departs", "departure", "departing",
    "reach", "reaches", "reaching",
    "start", "starts", "starting",
    "run", "runs", "timing", "timings",
    "when", "what time", "how long",
    "halt", "halts", "timetable", "schedule",
    "stops", "route", "pass", "passes",
])

_STATIC_KEYWORDS_FALLBACK = frozenset([
    "cancel", "cancellation", "refund", "charge", "charges", "fee", "fees",
    "rule", "rules", "luggage", "limit", "allowance", "penalty", "fine",
    "tte", "duty", "duties", "policy", "policies", "weight",
    "fare", "ticket price", "class", "sleeper", "ac", "general", "quota",
    "stops of", "schedule of", "timetable of", "route of", "passing through",
    "how to book", "senior citizen", "concession", "tatkal", "premium tatkal",
    "tdr", "rac", "gnwl", "pqwl", "rswl", "ckwl", "cnf", "wl",
    "vande bharat", "shatabdi", "rajdhani", "duronto",
    "garib rath", "jan shatabdi", "humsafar", "amrit bharat", "tejas",
    "coach", "berth",
])

_ROUTING_KEYWORDS_FALLBACK = frozenset([
    "between", "from", "to", "via", "through",
])

_GENERAL_RAILWAY_KEYWORDS = frozenset([
    "train", "trains", "railway", "railways", "station", "stations",
    "ticket", "tickets", "fare", "fares", "coach", "berth", "pnr",
    "irctc", "ntes", "tte", "rac", "wl", "gnwl", "pqwl", "rswl",
    "ckwl", "tq", "tqwl", "cnf", "tdr", "tatkal", "sleeper", "ac",
    "general", "quota", "luggage", "baggage", "cancellation", "refund",
    "schedule", "timetable", "delay", "delayed", "running", "status",
    "route", "routes", "passengers", "passenger", "catering", "pantry",
    "vande bharat", "tejas", "shatabdi", "rajdhani", "duronto",
    "garib rath", "jan shatabdi", "humsafar", "amrit bharat",
    "sampark kranti", "suburban", "express", "mail", "superfast", "sf",
    "loco", "engine", "rake", "wagon", "rpf", "cris", "bpc", "cantt",
    "junction", "jn", "platform", "pf", "halt", "stop",
])

# ── Station name cache ──────────────────────────────────────────────
_station_tokens_cached: set | None = None
_station_codes_cached:  set | None = None
_station_cache_lock = threading.Lock()

_STATION_STOP_WORDS = frozenset([
    "is", "in", "to", "on", "or", "now", "at", "for", "the", "and",
    "train", "trains", "station", "stations", "route", "routes", "status",
    "spot", "where", "late", "delay", "arrive", "departure", "platform",
    "track", "reach", "when", "location", "current", "time", "what", "how",
    "running", "today", "tomorrow", "yesterday", "daily", "weekly",
    "will", "expected", "supposed", "which", "does", "did", "has", "have",
])


def _load_station_cache() -> tuple[set, set]:
    global _station_tokens_cached, _station_codes_cached
    if _station_tokens_cached is not None:
        return _station_tokens_cached, _station_codes_cached
    with _station_cache_lock:
        if _station_tokens_cached is not None:
            return _station_tokens_cached, _station_codes_cached
        tokens: set[str] = set()
        codes:  set[str] = set()
        _ignore = frozenset({
            "junction", "jn", "junctions", "cabin", "road", "halt",
            "crossing", "station", "town", "city", "north", "south",
            "east", "west", "central", "new", "old", "and", "the",
            "via", "pass", "under", "over", "bridge",
        })
        try:
            from scripts.preprocess import build_station_lookup
            lookup = build_station_lookup()
            for code, info in lookup.items():
                codes.add(code.lower())
                for part in re.findall(r"\b[a-zA-Z]{3,}\b", info.get("name", "").lower()):
                    if part not in _ignore:
                        tokens.add(part)
                for aka in info.get("aka", []):
                    for part in re.findall(r"\b[a-zA-Z]{3,}\b", aka.lower()):
                        if part not in _ignore:
                            tokens.add(part)
        except Exception:
            pass
        _station_tokens_cached = tokens
        _station_codes_cached  = codes
    return _station_tokens_cached, _station_codes_cached


def _has_station_name(query: str) -> bool:
    try:
        tokens, codes = _load_station_cache()
        words = set(re.findall(r"\b[a-zA-Z]{2,}\b", query.lower())) - _STATION_STOP_WORDS
        return any(w in codes or w in tokens for w in words)
    except Exception:
        words = re.findall(r"\b[A-Z][a-z]+\b", query)
        filtered = [w for w in words if w.lower() not in (
            "train", "express", "mail", "superfast", "running", "status",
            "what", "where", "how", "when", "is", "the",
        )]
        return bool(filtered)


def _extract_train_number(query: str) -> str | None:
    """Extract first 4-5 digit train number from query."""
    # Prefer 5-digit first
    m = re.search(r'\b(\d{5})\b', query)
    if m:
        return m.group(1)
    # Fall back to 4-digit if preceded by train-related word
    m = re.search(r'(?:train|no\.?|number|#)\s*(\d{4})\b', query, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _extract_pnr_number(query: str) -> str | None:
    """Extract first 10-digit PNR number from query."""
    m = re.search(r'\b(\d{10})\b', query)
    return m.group(1) if m else None


def _extract_station_code(query: str) -> str | None:
    """Extract potential station code (2–4 uppercase chars)."""
    _exclude = frozenset({
        "AC", "CC", "TTE", "PNR", "PDF", "PNG", "JPG", "RAG",
        "LLM", "API", "IST", "ETA", "RPF", "EMU", "DMU", "MEX",
    })
    for code in re.findall(r'\b([A-Z]{2,4})\b', query):
        if code not in _exclude:
            return code
    return None


def _word_match(keyword_set: frozenset, text: str) -> list[str]:
    """Return keywords from the set that appear as whole words in text."""
    return [kw for kw in keyword_set if re.search(rf'\b{re.escape(kw)}\b', text)]


# ── Public API ──────────────────────────────────────────────────────

def classify_intent(query: str) -> dict:
    """
    Classify the query intent.

    Returns:
        {
            "intent":          "STATIC" | "LIVE" | "HYBRID" | "OUT_OF_DOMAIN",
            "intent_category": one of the INTENT_* constants,
            "confidence":      float (0.0–1.0),
            "train_no":        str | None,
            "station_code":    str | None,
            "pnr":             str | None,
            "is_pnr":          bool,
            "reasons":         list[str],
        }
    """
    query_lower = query.lower().strip()
    reasons: list[str] = []

    # ── Step 1: Extract entities ──────────────────────────────────
    pnr       = _extract_pnr_number(query)
    train_no  = _extract_train_number(query)
    station_code = _extract_station_code(query)
    station_detected = station_code is not None or _has_station_name(query)

    # ── Step 2: Pattern-based detection (first match wins) ────────
    for pattern, category, confidence in _PATTERNS:
        if pattern.search(query):
            reasons.append(f"Pattern match → {category}")
            coarse = _COARSE[category]
            is_pnr = (category == INTENT_PNR_STATUS)
            return {
                "intent":          coarse,
                "intent_category": category,
                "confidence":      confidence,
                "train_no":        train_no,
                "station_code":    station_code,
                "pnr":             pnr if is_pnr else None,
                "is_pnr":         is_pnr,
                "reasons":         reasons,
            }

    # ── Step 3: Keyword scoring fallback ─────────────────────────
    live_matches     = _word_match(_LIVE_KEYWORDS_FALLBACK, query_lower)
    schedule_matches = _word_match(_SCHEDULE_KEYWORDS_FALLBACK, query_lower)
    static_matches   = _word_match(_STATIC_KEYWORDS_FALLBACK, query_lower)
    routing_matches  = _word_match(_ROUTING_KEYWORDS_FALLBACK, query_lower)
    general_railway  = any(re.search(rf'\b{re.escape(kw)}\b', query_lower) for kw in _GENERAL_RAILWAY_KEYWORDS)

    # ── Step 4: OUT_OF_DOMAIN guard ───────────────────────────────
    if (not train_no and not pnr and not station_detected
            and not live_matches and not static_matches
            and not schedule_matches and not routing_matches
            and not general_railway):
        return {
            "intent":          "OUT_OF_DOMAIN",
            "intent_category": INTENT_OUT_OF_DOMAIN,
            "confidence":      1.0,
            "train_no":        None,
            "station_code":    None,
            "pnr":             None,
            "is_pnr":          False,
            "reasons":         ["No railway signals found in query"],
        }

    # ── Step 5: Score-based routing ───────────────────────────────
    # Cap scoring: max 1.0 per category regardless of keyword count
    live_score     = min(1.0, 0.35 * len(live_matches))
    schedule_score = min(1.0, 0.30 * len(schedule_matches))
    static_score   = min(1.0, 0.30 * len(static_matches) + 0.20 * len(routing_matches))

    if live_matches:
        reasons.append(f"Live keywords: {live_matches}")
    if schedule_matches:
        reasons.append(f"Schedule keywords: {schedule_matches}")
    if static_matches:
        reasons.append(f"Static keywords: {static_matches}")

    # Train number present but no other signals → show train info (STATIC)
    if train_no and not live_matches and not schedule_matches and not static_matches:
        static_score += 0.4
        reasons.append("Train number with no other signals → default to STATIC train info")

    # Station boosts static
    if station_detected:
        static_score += 0.3
        reasons.append("Station name/code detected → boost STATIC score")

    # Train name patterns boost static
    if any(p in query_lower for p in ("express", "mail", " sf ", "superfast", "passenger")):
        static_score += 0.2
        reasons.append("Train classification name detected → boost STATIC")

    # ── Step 6: Determine final category ─────────────────────────
    # Schedule keywords alone → SCHEDULE_QUERY (STATIC)
    if schedule_score > live_score and schedule_score > 0.2:
        category   = INTENT_SCHEDULE_QUERY
        confidence = min(1.0, schedule_score + 0.1)

    elif live_score > 0.05 and (static_score > 0.05 or schedule_score > 0.05):
        # Both live + static signals → HYBRID
        category   = INTENT_HYBRID
        confidence = min(1.0, (live_score + max(static_score, schedule_score)) / 2.0)
        reasons.append("Both live and static/schedule signals → HYBRID")

    elif live_score > static_score:
        category   = INTENT_LIVE_STATUS
        confidence = min(1.0, live_score)
        if not train_no:
            reasons.append("LIVE detected but no train number in query")

    elif static_score > 0 or general_railway:
        category   = INTENT_GENERAL_INFO
        confidence = min(1.0, max(0.5, static_score))

    else:
        # Fallback to HYBRID to be safe
        category   = INTENT_HYBRID
        confidence = 0.45
        reasons.append("Low confidence → fallback to HYBRID")

    coarse = _COARSE[category]
    return {
        "intent":          coarse,
        "intent_category": category,
        "confidence":      round(confidence, 2),
        "train_no":        train_no,
        "station_code":    station_code,
        "pnr":             None,
        "is_pnr":          False,
        "reasons":         reasons,
    }


# ── Legacy shim for any code using old function names ───────────────
def extract_train_number(query: str) -> str | None:
    return _extract_train_number(query)

def extract_pnr_number(query: str) -> str | None:
    return _extract_pnr_number(query)

def extract_station_code(query: str) -> str | None:
    return _extract_station_code(query)

def has_station_name(query: str) -> bool:
    return _has_station_name(query)
