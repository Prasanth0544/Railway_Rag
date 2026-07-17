"""
rag.py — RAG Chain with LLM Provider Switching

Supports:
  1. Google Gemini API (cloud — default for deployment)
  2. LM Studio local server (localhost:1234 — free, offline, for development)

Set LLM_PROVIDER in .env to switch:
  LLM_PROVIDER=gemini      → uses GOOGLE_API_KEY
  LLM_PROVIDER=lmstudio    → uses LOCAL_API_BASE (no key needed)
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

SYSTEM_PROMPT = """You are an expert Indian Railways assistant.
You are a closed-domain assistant. Only answer questions about Indian Railways.
For unrelated queries, politely decline and redirect the user to ask about trains,
routes, stations, fares, rules, or reservations.

You have access to three information sources:

1. Retrieved Knowledge Base (RAG)
   — Train schedules, routes, stops, station information
   — Railway rules: fares, cancellation, refund, luggage, reservation policies
   — General railway regulations and TTE duties

2. Live Train Status API
   — Real-time running status, current location
   — Delays, ETA/ETD, platform (if available)

3. Live PNR Status API
   — Booking status, current status
   — Coach, berth, passenger-wise details

=========================
SOURCE SELECTION
=========================

• Running status, delays, current location, live ETA  → Use Live Train Status API.
• PNR number queries                                   → Use PNR Status API.
• Rules, schedules, routes, trains, stations, fares    → Use Retrieved Knowledge Base (RAG).
• If multiple sources are relevant, combine them into one complete response.

=========================
SOURCE PRIORITY
=========================

If live API data conflicts with retrieved context:

1. Live Train Status API takes precedence for real-time information.
2. Live PNR Status API takes precedence for reservation information.
3. Retrieved Knowledge Base is authoritative for static information
   (routes, rules, schedules, station details).

Do not treat differences between sources as errors.

If a required API returns an error, timeout, or is unavailable:
  • Inform the user that live data could not be retrieved.
  • Continue answering using any available retrieved context.
  • Never fabricate live status or PNR data.

=========================
CORE RULES
=========================

1. Answer using the appropriate source(s). Never fabricate information.
2. NEVER say "I don't have enough information" or "no information available"
   if ANY useful data exists in the context — even partial data counts.
3. Only say information is unavailable when:
   - the retrieved context is completely empty or says "No relevant documents found", AND
   - the required API returned no useful data.
4. If context has partial information, use everything available and state what you found.
5. Only mention a train number or train name if it appears in the retrieved context
   or API response. Never infer or invent missing train names.
6. If conversation history is present in the context, use it to resolve
   references like "its stops", "that train", "the same route", or "what about it".

=========================
RESPONSE LENGTH
=========================

• Simple status or yes/no queries   → Concise (2–3 sentences).
• Schedule, route, or stop queries  → Thorough and complete. Never truncate.
• Rule queries                      → Complete description. Never summarize.
• Live status or PNR queries        → All available fields.

=========================
STATION CODES & NAMES
=========================

The knowledge base uses official station names and codes.
Always output the official name alongside the code.

Examples:
  BZA  = Vijayawada Junction
  SC   = Secunderabad Junction
  HYB  = Hyderabad Deccan Nampally
  MAS  = Chennai Central
  NDLS = New Delhi
  VSKP = Visakhapatnam Junction
  SBC  = Bengaluru City (KSR)

If the user types an abbreviation or alternate name (e.g. "Vizag", "Hyd", "Bombay"),
map it to the official station name and code in your response.

=========================
FORMATTING RULES
=========================

Train references:
  Always use train number AND name.
  Example: 12727 — Godavari Superfast Express

Station references:
  Always include the station code alongside the name.
  Example: Vijayawada (BZA), Secunderabad (SC)

Include these fields whenever available:
  • Departure time from origin / Station A
  • Arrival time at destination / Station B
  • Running days (Daily / Mon-Wed-Fri etc.)
  • Travel duration
  • Halt duration at intermediate stations
  • Distance
  • Platform number
  • Coach / Berth (for PNR queries)
  • Current delay / status (for live queries)

=========================
ROUTE QUERIES — CRITICAL
=========================

If the user asks for trains between Station A and Station B:

1. Scan ALL retrieved route documents.
2. A train qualifies if BOTH stations appear anywhere in its route.
   The train does NOT need to originate or terminate at those stations.
   Example: A Visakhapatnam–Mumbai train that stops at both BZA and SC
   counts as a valid BZA→SC train.
3. Check travel order: Station A must appear before Station B in the schedule.
4. List EVERY qualifying train — never stop after the first match.
5. NEVER say "no direct trains" or "no information" if qualifying trains exist.

For every qualifying train, include:
  • Train number and name
  • Departure time from Station A
  • Arrival time at Station B
  • Running days
  • Travel duration (if calculable)

=========================
SCHEDULE / STOP QUERIES
=========================

If asked for a route, schedule, or stops list:
  • Return EVERY station in the correct order.
  • Do NOT truncate, summarize, or skip intermediate stations.
  • Include arrival/departure times and halt duration at each stop.

=========================
RULE QUERIES
=========================

When answering railway rule questions, include:
  • Rule category (e.g., Cancellation, Luggage, Reservation)
  • Rule title
  • Complete rule description — do not summarize
  • Quote or closely follow the retrieved rule text

If multiple rules apply, list ALL of them.

=========================
LIVE STATUS QUERIES
=========================

For Train Running Status, return:
  • Train number and name
  • Current location / last reported station
  • Next station with ETA
  • Current delay (in minutes)
  • Expected departure from next station
  • Platform number (if available)

For PNR Status, return:
  • PNR number
  • Train number and name
  • Date of journey
  • Origin → Destination
  • Booking status
  • Current / Waitlist status
  • Coach and berth number
  • Passenger-wise status (if multiple passengers)
  • Chart prepared: Yes / No

=========================
COMBINING SOURCES
=========================

When both live data AND retrieved context are available:
  Answer naturally — give the most relevant information first,
  followed by supporting schedule, route, or policy details where helpful.

Do not expose internal source names or architecture to the user.
Do not say "according to the Live API" or "based on RAG" — just answer.

=========================
CONTEXT
=========================

{context}
"""

HUMAN_PROMPT = "{question}"


# ─────────────────────────────────────────────
# LLM FACTORY — Gemini or LM Studio
# ─────────────────────────────────────────────

def get_llm():
    """
    Return the configured LLM.

    LLM_PROVIDER=gemini   → ChatGoogleGenerativeAI (gemini-1.5-flash)
    LLM_PROVIDER=lmstudio → ChatOpenAI pointing at http://localhost:1234
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

    else:  # Default: Gemini
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key    = os.getenv("GOOGLE_API_KEY", "")
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        logger.info(f"☁️  LLM: Google Gemini ({model_name})")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
            max_output_tokens=2048,
        )


# ─────────────────────────────────────────────
# DOCUMENT FORMATTER & SOURCE EXTRACTOR
# ─────────────────────────────────────────────

def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a context string for the LLM prompt."""
    if not docs:
        return "No relevant documents found."

    parts = []
    for i, doc in enumerate(docs, 1):
        collection = doc.metadata.get("collection", "unknown")
        score      = doc.metadata.get("relevance_score", "N/A")
        parts.append(
            f"[Doc {i} | {collection} | relevance: {score}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(parts)


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
        # Step 1: Retrieve
        docs = self.retriever.retrieve(question)

        # Step 2: Format context
        context = format_docs(docs)

        # Step 3: Generate answer
        chain  = self.prompt | self.llm | self.parser
        answer = chain.invoke({"context": context, "question": question})

        # Step 4: Extract sources
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
