"""
rag.py — RAG Chain with LLM Provider Switching

Supports:
  1. Google Gemini API (cloud — default for deployment)
  2. LM Studio local server (localhost:1234 — free, offline, for development)
  3. OpenRouter (cloud — stealth/ox-alpha or any OpenRouter model)

Set LLM_PROVIDER in .env to switch:
  LLM_PROVIDER=gemini      → uses GOOGLE_API_KEY
  LLM_PROVIDER=lmstudio    → uses LOCAL_API_BASE (no key needed)
  LLM_PROVIDER=openrouter  → uses OPENROUTER_API_KEY + OPENROUTER_MODEL

Note: Embeddings always use Gemini gemini-embedding-001 regardless of LLM_PROVIDER.
"""

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from app.retriever import get_unified_retriever


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

from app.logger import get_logger
logger = get_logger("app.rag")

SYSTEM_PROMPT = """You are RailGPT — an expert Indian Railways assistant powered by AI.
You are a closed-domain assistant. Only answer questions about Indian Railways.
For completely unrelated topics (cooking, sports, weather etc.), politely decline and redirect
to railway queries. However, general follow-up questions in a railway conversation are fine.

==========================
SELF-INTRODUCTION
==========================

If the user asks "who are you?", "tell me about yourself", "what can you do?",
"introduce yourself", or any similar meta-question about your identity or capabilities,
respond with a friendly self-introduction:

  Hello! I am **RailGPT** — your dedicated AI assistant for all things related to **Indian Railways**.

  Here is how I can help you:
  • **Train Schedules & Routes**: Find timetables, intermediate stops, days of operation,
    and trains running between stations.
  • **Live Train Running Status**: Check real-time tracking, delay updates, current location,
    and estimated arrival times.
  • **PNR Status**: Track your 10-digit PNR booking status, coach/berth allocation,
    and chart preparation status.
  • **Railway Rules & Policies**: Get official info on cancellation fees, refunds, luggage
    allowances, Tatkal/Premium Tatkal quotas, and travel rules.
  • **Classes & Quotas**: Understand travel classes (1A, 2A, 3A, 3E, SL, CC, etc.)
    and reservation quotas (GN, TQ, PT, LD, PH, DF, etc.).
  • **Station Information**: Look up station codes, platform details, zones,
    and nearby connectivity.

  How can I help you with your journey or railway query today?


You have access to three real-time and static sources:

1. KNOWLEDGE BASE (RAG) — ChromaDB vector search
   • Train schedules, stops, departure/arrival times
   • Station info, codes, zones, platforms
   • Railway rules: fares, cancellation, refunds, luggage, quotas, reservations
   • TTE duties, pass rules, tatkal, premium tatkal

2. LIVE TRAIN STATUS API (NTES / erail.in)
   • Real-time running position, current station, last reported location
   • Delay in minutes, ETA at next station
   • NOTE: GPS coverage is NOT available for ALL trains.
     Many trains (esp. non-express, older routes) have no real-time GPS tracking.
     When GPS data is absent, say clearly: "Real-time GPS location is not available
     for this train from public APIs. Check the IRCTC app or NTES app for the latest status."

3. LIVE PNR STATUS API (ConfirmTkt / erail.in)
   • Booking status (CNF / WL / RAC)
   • Coach, berth, passenger details
   • Chart prepared status

==========================
SOURCE SELECTION
==========================

• "Where is train X?", "Is train X late?", "running status"   → Live Train Status API
• PNR number (10 digits) queries                               → PNR Status API
• Schedules, routes, stops, fares, rules, stations             → Knowledge Base (RAG)
• Combine sources when relevant — never ignore available data.

==========================
SOURCE PRIORITY
==========================

Live API overrides RAG for real-time data. RAG is authoritative for static data (rules, schedules).
If APIs error out or timeout:
  • Clearly say live data is temporarily unavailable.
  • Use any schedule/route data from RAG to answer partially.
  • NEVER fabricate live positions, delays, or PNR status.
  • NEVER say a train is "on time" or "running" if you have no live data confirming it.

==========================
ANTI-HALLUCINATION RULES
==========================

1. NEVER invent train numbers, names, departure times, or station codes.
2. NEVER say "no information available" if ANY partial context exists — use what you have.
3. NEVER say a train "has no stops" between A and B — check ALL route documents first.
4. Only say information is missing when context is completely empty AND API returned nothing.
5. Do NOT confuse train numbers — 12727 ≠ 12728. Always double-check number match.
6. Do NOT fabricate platform numbers — only state them if explicitly in the data.

==========================
CORE RESPONSE RULES
==========================

1. Always use train number AND name together: "12727 — Godavari Superfast Express"
2. Always include station code with name: "Vijayawada (BZA)", "Secunderabad (SC)"
3. All times are in IST (24-hour format): "14:30 IST"
4. Use conversation history to resolve references: "its stops", "that train", "same route"
5. Do not expose internal source names: never say "according to RAG" or "Live API says"

==========================
CLASS CODES (use when relevant)
==========================

1A  = First AC (single/double cabin)
2A  = Second AC (4-berth)
3A  = Third AC (6-berth)
SL  = Sleeper Class
CC  = AC Chair Car
2S  = Second Sitting
GN  = General / Unreserved
EC  = Executive Chair Car (Vande Bharat / Shatabdi premium)
3E  = Third AC Economy (newer trains)

==========================
QUOTA TYPES (for reservation queries)
==========================

GN   = General Quota
TQ   = Tatkal Quota (opens 1 day before, premium fare)
PT   = Premium Tatkal (opens 1 day before, dynamic fare)
LD   = Ladies Quota
PH   = Physically Handicapped Quota
DF   = Defence Quota
HO   = Head Office (railway staff)
GNWL = General Waitlist
PQWL = Pooled Quota Waitlist
RLWL = Remote Location Waitlist
RSWL = Roadside Waitlist
RAC  = Reservation Against Cancellation (confirmed travel, shared berth)

==========================
PNR STATUS CODES
==========================

CNF   = Confirmed (with coach/berth)
RAC   = Reservation Against Cancellation (travel allowed, share berth)
WL#   = Waitlist number (e.g., WL4 = 4th on waitlist)
GNWL# = General Waitlist
PQWL# = Pooled Quota Waitlist
CAN   = Cancelled
RELEASED = Seat released for other passengers

==========================
STATION CODES — COMMON REFERENCES
==========================

NDLS = New Delhi          | BCT  = Mumbai Central
SBC  = Bengaluru City     | MAS  = Chennai Central
HYB  = Hyderabad Deccan   | SC   = Secunderabad Junction
NPA  = Hyderabad Nampally | BZA  = Vijayawada Junction
VSKP = Visakhapatnam Jn   | RJY  = Rajahmundry
GNT  = Guntur Junction    | TPTY = Tirupati
NLR  = Nellore            | KZJ  = Kazipet Junction
OGL  = Ongole             | GDR  = Gudur Junction
MGLA = Mangalagiri        | BBS  = Bhubaneswar
PURI = Puri               | PUNE = Pune Junction
AMD  = Ahmedabad Junction

Map user abbreviations to official names:
  "Vizag" → Visakhapatnam (VSKP)
  "Hyd" / "Hyderabad" → check context: city centre = HYB (Deccan/Nampally), north = SC (Secunderabad)
  "Secunderabad" → Secunderabad Junction (SC)
  "Nampally" → Hyderabad Nampally (NPA) or Hyderabad Deccan (HYB)
  "Bangalore" → Bengaluru City (SBC)
  "Bombay" / "Mumbai" → Mumbai Central (BCT) or CST/LTT
  "Madras" → Chennai Central (MAS)
  "Delhi" → New Delhi (NDLS) or Old Delhi (DLI)

==========================
RESPONSE LENGTH
==========================

• Live status / PNR / Yes-No     → Concise (2–4 sentences). Include all key fields.
• Full schedule ("stops of 12727") → Complete. NEVER truncate. List ALL stops in order.
• Rule / policy queries          → Full rule text. Never summarize. Quote accurately.
• Route queries (A to B trains)  → Context is pre-trimmed to key stops per train.
                                   List EVERY qualifying train. Never say "no trains" if data exists.

==========================
ROUTE QUERIES — CRITICAL
==========================

If user asks for trains between Station A (FROM) and Station B (TO):
1. Scan ALL retrieved route documents — do not stop at first match.
2. The context you receive is PRE-TRIMMED — each train doc shows only:
   • Train origin (where the train starts)
   • Station A — the user's boarding point (with departure time if available)
   • Station B — the user's alighting point (with arrival time if available)
   • Train destination (where the train ends)
   Intermediate stops between A and B are intentionally omitted.
3. A train qualifies if BOTH Station A and Station B appear in its trimmed doc.
   Travel direction: Station A must appear BEFORE Station B in the sequence.
4. List EVERY qualifying train with:
   • Train number and name
   • Running days (e.g. "Daily", "Mon only", "Mon Wed Fri") — ALWAYS include this
   • Departure from A, Arrival at B (from the trimmed context)
   • Train origin and final destination (for full journey context)
   • Available classes (if present in doc)
5. Sort trains: list Daily trains FIRST, then weekly/occasional trains.
   Label seasonal/special trains clearly: "(Seasonal — verify on NTES before travel)"
6. NEVER say "no direct trains" if any qualifying train exists in context.
7. Do NOT ask for more stops — the trimmed context is intentional and sufficient.

==========================
SCHEDULE / STOP QUERIES (Specific Train Number)
==========================

When user asks for the full schedule/stops of a SPECIFIC train (e.g. "stops of 12727"):
• Return EVERY station in correct order — these docs are NOT trimmed.
• Include arrival time, departure time, halt duration at each stop.
• Do NOT skip or truncate any stations.
• State the day offset if a train runs overnight: "Day 2" for next-day arrivals.
• CRITICAL: Station names MUST come from the retrieved context only.
  Do NOT use your own training-data knowledge to fill in station names.
  If a stop shows only a code (e.g. "NRT"), write it as "NRT" — do NOT
  guess or invent a city name. The retrieved station docs will have the
  correct name (e.g. "Station NRT — Narasaraopet") — use that.

==========================
TRAIN OVERVIEW FORMAT
==========================

When user asks for an "overview", "details", "info" or "tell me about" a specific train,
return ALL of the following fields (from retrieved context only — never fabricate):

• Train Number & Name: e.g. "17225 — Marathwada Express"
• Source / Origin: departure station with code
• Destination: arrival station with code
• Days of Operation: e.g. "Daily", "Monday, Thursday", "Bi-weekly (Fri, Sun)"
• Departure Time: from origin station (IST)
• Arrival Time: at destination (IST, with Day 2/3 if overnight)
• Total Distance: in km (if available)
• Total Duration: in hours and minutes (if available)
• Total Intermediate Stops: count
• Railway Zone: e.g. "South Central Railway (SCR)"
• Train Type: e.g. "Superfast Express", "Mail/Express (MEX)", "Passenger"
• Available Classes: list class codes if present in data

If any field is genuinely not in the retrieved context, omit that field — do NOT guess.

==========================
LIVE STATUS — HONESTY RULES
==========================

When live train status is requested:

✅ If GPS data IS available (last station, ETA, delay): Report it fully.
   Example: "Train 12642 departed Tilak Bridge (TKJ) at 21:17 IST. Next stop:
   Agra Cantt (AGC) at 23:27 IST. Currently running on time."

⚠️ If only "running on time" with NO location data: Be honest.
   Say: "According to available sources, train [number] is currently operational
   and running on schedule. However, real-time GPS location data is not available
   through public APIs for this train right now. For exact location, check the
   NTES app (enquiry.indianrail.gov.in) or 'Where is my train' (139 helpline)."

❌ NEVER say "running on time" if no API confirmed it.
❌ NEVER fabricate a station location or delay figure.

==========================
RULE / POLICY QUERIES
==========================

Include:
  • Rule category (Cancellation, Luggage, Tatkal, Reservation, Refund, etc.)
  • Complete rule description — quote or closely follow retrieved text
  • Amounts, percentages, time limits
  • Exceptions if mentioned in the retrieved context
  • List ALL applicable rules if multiple exist

==========================
PNR STATUS RESPONSE FORMAT
==========================

Return ALL available fields:
  • PNR Number
  • Train: [number — name]
  • Date of Journey
  • From → To
  • Class
  • Quota
  • Booking Status (at time of booking)
  • Current Status: CNF / RAC / WL#
  • Coach & Berth (if CNF)
  • Chart Prepared: Yes / No
  • Passenger-wise breakdown if multiple passengers

==========================
CONTEXT
==========================

{context}
"""

HUMAN_PROMPT = "{question}"



# ─────────────────────────────────────────────
# LLM FACTORY — Gemini, LM Studio, or OpenRouter
# ─────────────────────────────────────────────

def get_llm():
    """
    Return the configured LLM.

    LLM_PROVIDER=gemini      → ChatGoogleGenerativeAI (gemini-3.6-flash)
    LLM_PROVIDER=lmstudio    → ChatOpenAI pointing at http://localhost:1234
    LLM_PROVIDER=openrouter  → ChatOpenAI pointing at https://openrouter.ai/api/v1
                               Uses OPENROUTER_API_KEY and OPENROUTER_MODEL (default: stealth/ox-alpha)

    Embeddings always use Gemini gemini-embedding-001 regardless of this setting.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()

    if provider == "lmstudio":
        from langchain_openai import ChatOpenAI
        base_url = os.getenv("LOCAL_API_BASE", "http://localhost:1234/v1")
        model    = os.getenv("LOCAL_MODEL_NAME", "local-model")

        logger.info(f"🖥️  LLM: LM Studio @ {base_url} (model: {model})")
        return ChatOpenAI(
            base_url=base_url,
            api_key="lm-studio",         # LM Studio ignores this but OpenAI client needs it
            model=model,
            temperature=0.3,
            max_tokens=1024,
        )

    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI
        api_key    = os.getenv("OPENROUTER_API_KEY", "")
        model_name = os.getenv("OPENROUTER_MODEL", "stealth/ox-alpha")

        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set. Cannot use LLM_PROVIDER=openrouter.")

        logger.info(f"🌐  LLM: OpenRouter ({model_name})")
        return ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=model_name,
            temperature=0.3,
            max_tokens=8192,
            default_headers={
                "HTTP-Referer": "https://github.com/Prasanth0544/Railway_Rag",
                "X-Title": "RailGPT — Indian Railways Assistant",
            },
        )

    else:  # Default: Gemini
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key    = os.getenv("GOOGLE_API_KEY", "")
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

        logger.info(f"☁️  LLM: Google Gemini ({model_name})")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
            max_output_tokens=8192,       # 2048 was too small for full 65-80 stop schedules
        )


import re as _re

# ─────────────────────────────────────────────────────────────────
# SMART CONTEXT BUILDER
# Query-type-aware document formatter.
# Sends only what Gemini needs per query type — no more, no less.
# ─────────────────────────────────────────────────────────────────

# Per-strategy character budgets
# A 65-stop route doc is ~2,000 chars; give enough room for train info + full stops
_BUDGET = {
    "train_number":  15_000,  # full route doc (up to 65 stops ~2,000c) + train info
    "route_search":  12_000,  # 8-15 compact route docs for multi-train comparison
    "rules_refund":  15_000,  # full rules without truncation (rules avg ~400c each)
    "station_info":   8_000,  # station doc + nearby trains info
    "class_amenity": 10_000,  # train + rules + reference docs
    "default":       18_000,  # fallback — generous budget for unclassified queries
}

_RULES_KEYWORDS = frozenset([
    "cancel", "cancellation", "refund", "charges", "charge", "fine",
    "penalty", "luggage", "luggage limit", "allowance", "tte", "duty",
    "duties", "rule", "rules", "tatkal", "rac", "gnwl", "pqwl", "ckwl",
    "rswl", "wl", "waitlist", "waiting list", "tdr", "policy", "policies",
    "concession", "senior citizen", "divyaang", "handicapped", "quota",
    "premium tatkal", "booking", "chain pull", "without ticket",
    "boarding", "berth change", "food", "catering", "e-catering",
])

_STATION_KEYWORDS = frozenset([
    "station code", "station info", "which zone", "which division",
    "how many platforms", "retiring room", "cloak room", "station master",
    "trains arriving at", "live station", "which station is",
])

_CLASS_KEYWORDS = frozenset([
    "which class", "classes available", "class in", "difference between",
    "sleeper vs", "2a vs", "3a vs", "cc class", "ec class", "pantry car",
    "amenities", "vande bharat", "humsafar", "amrit bharat", "rajdhani class",
    "how many berths", "coach layout", "superfast vs", "express vs",
])


def _build_context(docs: list[Document], max_chars: int) -> str:
    """
    Build context string from docs, respecting max_chars budget.
    Unlike the old approach (stop at first overflow), this version
    tries to fit smaller docs even after skipping a large one.
    """
    parts = []
    total = 0
    skipped = 0

    for i, doc in enumerate(docs, 1):
        content = doc.page_content
        content_len = len(content)
        if total + content_len > max_chars and parts:
            skipped += 1
            continue  # try next — it might be smaller and fit
        total += content_len
        collection = doc.metadata.get("collection", "unknown")
        score      = doc.metadata.get("relevance_score", "N/A")
        parts.append(f"[Doc {i} | {collection} | score: {score}]\n{content}")

    if skipped:
        logger.debug(f"[BUILD_CTX] {total} chars used / {max_chars} budget — {skipped} docs skipped")

    return "\n\n---\n\n".join(parts) if parts else "No relevant documents found."


def smart_format_docs(
    docs: list[Document],
    query: str = "",
    intent: str = "STATIC",
    intent_category: str = "",
) -> str:
    """
    Build LLM context intelligently based on query type.

    Strategy matrix:
      1. train_number  → exact train docs only          (budget: 15,000 chars)
      2. rules_refund  → rules + references docs only   (budget: 15,000 chars)
      3. station_info  → station docs only              (budget:  8,000 chars)
      4. class_amenity → train + rules docs             (budget: 10,000 chars)
      5. route_search  → train_route + train docs only  (budget: 12,000 chars)
      6. default       → all docs, generous budget      (budget: 18,000 chars)
    """
    if not docs:
        return "No relevant documents found."

    # LIVE and PNR answered by external APIs — no ChromaDB context needed
    if intent in ("LIVE", "PNR"):
        return ""

    q = query.lower()

    # Strategy 1: Specific train number query ("stops of 12727", "12727 schedule")
    train_no_match = _re.search(r'\b(\d{5})\b', query)
    if train_no_match:
        num = train_no_match.group(1)
        relevant = [d for d in docs if d.metadata.get("train_no") == num]
        if not relevant:
            relevant = docs[:3]
        logger.debug(f"[SMART_CTX] Strategy=train_number train={num} docs={len(relevant)}")
        return _build_context(relevant, _BUDGET["train_number"])

    # Strategy 2: Rules / Refund / Policy
    if intent_category == "CANCELLATION_RULES" or any(kw in q for kw in _RULES_KEYWORDS):
        rules_docs = [d for d in docs if d.metadata.get("source_type") in ("rule", "reference")]
        if not rules_docs:
            rules_docs = docs
        logger.debug(f"[SMART_CTX] Strategy=rules_refund (intent={intent_category or 'kw'}) docs={len(rules_docs)}")
        return _build_context(rules_docs, _BUDGET["rules_refund"])

    # Strategy 3: Station info — triggered by classifier OR keyword
    # Fixes Issue 4: "Tell me about Vijayawada Junction station" → STATION_INFO intent
    # used to miss narrow _STATION_KEYWORDS list and fall to default strategy
    if intent_category == "STATION_INFO" or any(kw in q for kw in _STATION_KEYWORDS):
        station_docs = [d for d in docs if d.metadata.get("source_type") == "station"]
        if not station_docs:
            station_docs = docs
        logger.debug(f"[SMART_CTX] Strategy=station_info (intent={intent_category or 'kw'}) docs={len(station_docs)}")
        return _build_context(station_docs, _BUDGET["station_info"])

    # Strategy 4: Train class / amenity query
    if any(kw in q for kw in _CLASS_KEYWORDS):
        class_docs = [d for d in docs if d.metadata.get("source_type") in ("train", "rule", "reference")]
        if not class_docs:
            class_docs = docs
        logger.debug(f"[SMART_CTX] Strategy=class_amenity docs={len(class_docs)}")
        return _build_context(class_docs, _BUDGET["class_amenity"])

    # Strategy 5: Route search — triggered by classifier OR keyword/station-code heuristics
    # Fixes Issue 3: BETWEEN_STATIONS intent now directly selects route strategy
    has_route_kw = any(kw in q for kw in [
        "trains from", "trains between", "trains via", "which trains",
        "train from", "train between", "express from", "go from",
        "travel from", "train to", "go to",
    ])
    station_codes = _re.findall(r'\b[A-Z]{2,4}\b', query)
    if intent_category == "BETWEEN_STATIONS" or len(station_codes) >= 2 or has_route_kw:
        route_docs = [d for d in docs if d.metadata.get("source_type") in ("train_route", "train")]
        if not route_docs:
            route_docs = docs
        logger.debug(f"[SMART_CTX] Strategy=route_search (intent={intent_category or 'kw'}) docs={len(route_docs)}")
        return _build_context(route_docs, _BUDGET["route_search"])

    # Strategy 6: Default
    logger.debug(f"[SMART_CTX] Strategy=default docs={len(docs)}")
    return _build_context(docs, _BUDGET["default"])



def format_docs(docs: list[Document]) -> str:
    """Backward-compatible wrapper — use smart_format_docs() for new endpoints."""
    return _build_context(docs, _BUDGET["default"])



def get_sources(docs: list[Document]) -> list[dict]:
    """Extract structured source metadata from retrieved documents."""
    sources = []
    for doc in docs:
        src = {
            "type"            : doc.metadata.get("source_type", "unknown"),
            "relevance_score" : doc.metadata.get("relevance_score", 0.0),
        }

        stype = doc.metadata.get("source_type", "")
        if stype == "train":
            src["train_no"]   = doc.metadata.get("train_no", "")
            src["train_name"] = doc.metadata.get("train_name", "")
        elif stype == "train_route":
            src["train_no"]   = doc.metadata.get("train_no", "")
            src["train_name"] = doc.metadata.get("train_name", "")
        elif stype == "station":
            src["station_code"] = doc.metadata.get("station_code", "")
            src["station_name"] = doc.metadata.get("station_name", "")
        elif stype == "rule":
            src["category"]   = doc.metadata.get("category", "")
            src["rule_title"] = doc.metadata.get("rule_title", "")
        elif stype == "reference":
            src["ref_type"]   = doc.metadata.get("ref_type", "")

        sources.append(src)
    return sources


# ─────────────────────────────────────────────
# RAG CHAIN
# ─────────────────────────────────────────────

class RAGChain:
    """
    Retrieval-Augmented Generation pipeline.

    Retrieve → Format Context → LLM → Structured Response
    """

    def __init__(self):
        self.retriever = get_unified_retriever(top_k=10)
        self.llm       = get_llm()
        self.prompt    = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human",  HUMAN_PROMPT),
        ])
        self.parser = StrOutputParser()
        logger.info("✅ RAG Chain initialized")

    def invoke(self, question: str) -> dict:
        """
        Full RAG pipeline:
          1. Retrieve relevant documents from ChromaDB
          2. Format into context string
          3. Send to LLM with prompt
          4. Return structured response with sources
        """
        import time
        t0 = time.time()

        # Step 1: Retrieve (includes Gemini embedding API call)
        docs = self.retriever.retrieve(question)
        t1 = time.time()
        logger.info(f"[TIMING] Retrieval (embed+search): {t1-t0:.2f}s  ({len(docs)} docs)")

        # Step 2: Confidence Check
        route_docs = [d for d in docs if d.metadata.get("source_type") == "train_route"]
        resolved   = getattr(self.retriever, "_last_all_stations", None)

        if not route_docs and resolved and len(resolved) >= 2:
            from_name = resolved[0][0]
            to_name   = resolved[1][0]
            from_code = resolved[0][1]
            to_code   = resolved[1][1]
            answer = (
                f"No direct trains were found from **{from_name} ({from_code})** "
                f"to **{to_name} ({to_code})** in the available route data.\n\n"
                f"This could mean:\n"
                f"- No train connects these stations directly\n"
                f"- These stations are served by connecting routes\n"
                f"- The data may not include all services\n\n"
                f"Please verify on [NTES](https://enquiry.indianrail.gov.in) or the IRCTC app."
            )
            logger.info(f"[CONFIDENCE] Zero route docs for {from_code}→{to_code} — short-circuit, no LLM call")
            return {
                "question"               : question,
                "answer"                 : answer,
                "sources"                : [],
                "num_documents_retrieved": len(docs),
            }

        # Step 3: Format context (smart, query-type-aware)
        context = smart_format_docs(docs, query=question, intent="STATIC")
        t2 = time.time()
        logger.info(f"[TIMING] Context format: {t2-t1:.2f}s  ({len(context)} chars)")

        # Step 4: Generate answer (Gemini LLM call)
        chain  = self.prompt | self.llm | self.parser
        answer = chain.invoke({"context": context, "question": question})
        t3 = time.time()
        logger.info(f"[TIMING] LLM generation: {t3-t2:.2f}s")
        logger.info(f"[TIMING] TOTAL: {t3-t0:.2f}s")

        # Step 5: Extract sources
        sources = get_sources(docs)

        return {
            "question"               : question,
            "answer"                 : answer,
            "sources"                : sources,
            "num_documents_retrieved": len(docs),
        }



# ─────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────

_rag_chain: RAGChain | None = None


def get_rag_chain() -> RAGChain:
    """Get or create the singleton RAG chain instance."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain
