"""
intent.py — Query Intent Classifier

Classifies user questions into:
  - STATIC: only requires static rules/routes database (ChromaDB)
  - LIVE: only requires real-time NTES info (no database context needed)
  - HYBRID: requires both (e.g. static route info + live delays)

Uses optimized keyword checklists and regex rules for high performance and accuracy.
"""

import re# ── Keywords defining LIVE queries ──────────────────────────────
LIVE_KEYWORDS = [
    "live", "running", "status", "spot", "where", "late", "delay", "delayed",
    "arrive", "arriving", "arrived", "arrives", "departure", "departing", "departed", "departs",
    "platform", "pf", "cancelled", "running status", "is it on time", "where is", "expected arrival",
    "track", "reach", "reaching", "reached", "reaches", "when", "location", "current", "position",
    "start", "started", "starts", "starting", "left", "commence", "commenced", "commencing", "run", "runs"
]

# ── Keywords defining STATIC queries ────────────────────────────
STATIC_KEYWORDS = [
    "cancel", "cancellation", "refund", "charge", "charges", "fee", "fees",
    "rule", "rules", "luggage", "limit", "allowance", "penalty", "fine",
    "tte", "duty", "duties", "policy", "policies", "weight", "extra bag",
    "fare", "ticket price", "class", "sleeper", "ac", "general", "quota",
    "stops of", "schedule of", "timetable", "route of", "passing through",
    "how to book", "senior citizen", "concession", "tatkal", "premium tatkal",
    "stop", "stops", "timetable", "schedule"
]

# ── Station keywords indicating spatial/routing queries ─────────
ROUTING_KEYWORDS = ["between", "from", "to", "via", "through"]


def extract_train_number(query: str) -> str | None:
    """Extract first 5-digit train number from query."""
    match = re.search(r"\b(\d{5})\b", query)
    return match.group(1) if match else None


def extract_pnr_number(query: str) -> str | None:
    """Extract first 10-digit PNR number from query."""
    match = re.search(r"\b(\d{10})\b", query)
    return match.group(1) if match else None


def extract_station_code(query: str) -> str | None:
    """Extract potential station code (3-4 uppercase characters)."""
    matches = re.findall(r"\b([A-Z]{2,4})\b", query)
    for code in matches:
        if code not in ("AC", "CC", "TTE", "PNR", "PDF", "PNG", "JPG", "RAG", "LLM", "API"):
            return code
    return None


_station_tokens_cached = None
_station_codes_cached = None

def get_station_tokens_and_codes():
    global _station_tokens_cached, _station_codes_cached
    if _station_tokens_cached is not None:
        return _station_tokens_cached, _station_codes_cached
    
    tokens = set()
    codes = set()
    try:
        from scripts.preprocess import build_station_lookup
        lookup = build_station_lookup()
        ignore_tokens = {
            "junction", "jn", "junctions", "cabin", "road", "halt", "crossing", 
            "station", "town", "city", "north", "south", "east", "west", "central", 
            "new", "old", "and", "the", "via", "pass", "under", "over", "bridge"
        }
        for code, info in lookup.items():
            codes.add(code.lower())
            name_parts = re.findall(r"\b[a-zA-Z]{3,}\b", info.get("name", "").lower())
            for part in name_parts:
                if part not in ignore_tokens:
                    tokens.add(part)
            for aka in info.get("aka", []):
                aka_parts = re.findall(r"\b[a-zA-Z]{3,}\b", aka.lower())
                for part in aka_parts:
                    if part not in ignore_tokens:
                        tokens.add(part)
    except Exception:
        pass
    _station_tokens_cached = tokens
    _station_codes_cached = codes
    return tokens, codes


def has_station_name(query: str) -> bool:
    """Check if any word in the query is a valid station name or code."""
    try:
        tokens, codes = get_station_tokens_and_codes()
        
        # Extract word tokens from query
        query_words = set(re.findall(r"\b[a-zA-Z]{2,}\b", query.lower()))
        
        # Stop words to ignore so they don't trigger station detection
        stop_words = {
            "is", "in", "to", "on", "or", "now", "at", "for", "the", "and", "train", "trains",
            "station", "stations", "route", "routes", "status", "spot", "where", "late", "delay",
            "arrive", "departure", "platform", "pf", "track", "reach", "when", "location", "current",
            "time", "what", "how", "running", "today", "tomorrow", "yesterday", "daily", "weekly",
            "will", "expected", "supposed"
        }
        query_words = query_words - stop_words
        
        for word in query_words:
            if word in codes or word in tokens:
                return True
    except Exception as e:
        # Fallback to simple capitalized words regex
        words = re.findall(r"\b[A-Z][a-z]+\b", query)
        filtered = [w for w in words if w.lower() not in ("train", "express", "mail", "superfast", "running", "status", "what", "where", "how", "when", "is", "the")]
        if filtered:
            return True
    return False


def classify_intent(query: str) -> dict:
    """
    Classify the query intent into STATIC, LIVE, or HYBRID.

    Returns:
        {
            "intent": "STATIC" | "LIVE" | "HYBRID",
            "confidence": float (0.0 to 1.0),
            "train_no": str | None,
            "station_code": str | None,
            "pnr": str | None,
            "is_pnr": bool,
            "reasons": list[str]
        }
    """
    query_lower = query.lower()
    pnr = extract_pnr_number(query)
    if pnr or "pnr" in query_lower:
        return {
            "intent": "LIVE",
            "confidence": 1.0,
            "train_no": None,
            "station_code": None,
            "pnr": pnr,
            "is_pnr": True,
            "reasons": ["PNR status query detected"]
        }

    train_no = extract_train_number(query)
    station_code = extract_station_code(query)
    station_detected = station_code is not None or has_station_name(query)

    live_score = 0.0
    static_score = 0.0
    reasons = []

    # 1. Regex checks for direct patterns
    if train_no and any(pat in query_lower for pat in ("where is", "spot", "running status", "status of")):
        live_score += 0.8
        reasons.append("Contains train number and tracking phrase ('where is' / 'spot' / 'status')")

    if train_no and any(pat in query_lower for pat in ("late", "delay", "on time", "expected at", "expected arrival")):
        live_score += 0.7
        reasons.append("Contains train number and late/delay indicator")

    if train_no and any(pat in query_lower for pat in ("start", "depart", "leave", "run", "cancel")):
        live_score += 0.7
        reasons.append("Contains train number and transit/commencement indicator ('start' / 'depart' / 'leave' / 'run' / 'cancel')")

    # 2. Whole word matching helper
    def count_word_matches(words_list, text):
        matches = []
        for kw in words_list:
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, text):
                matches.append(kw)
        return matches

    live_matches = count_word_matches(LIVE_KEYWORDS, query_lower)
    static_matches = count_word_matches(STATIC_KEYWORDS, query_lower)
    routing_matches = count_word_matches(ROUTING_KEYWORDS, query_lower)

    # Specific phrase matches
    if "expected at" in query_lower or "expected arrival" in query_lower:
        live_matches.append("expected")

    if live_matches:
        live_score += 0.3 * len(live_matches)
        reasons.append(f"Matched live keywords: {live_matches}")

    if static_matches:
        static_score += 0.3 * len(static_matches)
        reasons.append(f"Matched static keywords: {static_matches}")

    if routing_matches:
        static_score += 0.2 * len(routing_matches)
        reasons.append(f"Matched routing keywords: {routing_matches}")

    # If a station is detected, boost static score because they probably want station route/timing info
    if station_detected:
        static_score += 0.3
        reasons.append("Station name or code detected in query")

    # Specific check: if query has a train name (e.g. "Godavari Express")
    if any(pat in query_lower for pat in ("express", "mail", "sf", "superfast", "passenger")):
        static_score += 0.25
        reasons.append("Matched train classification indicator")

    # If it has a train number but no other keywords, default to STATIC (to show train info)
    if train_no and not live_matches and not static_matches and not station_detected:
        static_score += 0.4
        reasons.append("Contains train number with no specific live/station indicators (defaulting to static train view)")

    # 3. Determine final intent
    if live_score > 0.05 and static_score > 0.05:
        intent = "HYBRID"
        confidence = min(1.0, (live_score + static_score) / 2.0)
        reasons.append("Matched both static and live indicators (hybrid)")
    elif live_score > static_score:
        intent = "LIVE"
        confidence = min(1.0, live_score)
        if not train_no:
            reasons.append("LIVE intent detected but no train number was found in query")
    else:
        intent = "STATIC"
        confidence = min(1.0, max(0.5, static_score))

    # Fallback to HYBRID if confidence is low to ensure user gets all relevant info
    if confidence < 0.35 and intent != "STATIC":
        intent = "HYBRID"
        confidence = 0.5
        reasons.append("Low confidence score, routing to HYBRID to be safe")

    return {
        "intent": intent,
        "confidence": round(confidence, 2),
        "train_no": train_no,
        "station_code": station_code,
        "pnr": None,
        "is_pnr": False,
        "reasons": reasons
    }
