"""
main.py — FastAPI Application
Provides REST API endpoints for the Railway RAG Assistant.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Force UTF-8 for Windows console (prevents UnicodeEncodeError with emojis)
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except AttributeError:
    pass

import asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]

from app.logger import get_logger
logger = get_logger("app.main")


# Load environment variables before anything else
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# Heavy imports deferred to first use — keeps startup fast so Render detects the port quickly
# (app.rag, app.retriever, app.intent, app.ntes_client, app.pnr_client are imported inside functions)



# --- Pydantic Models ---

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Natural language question about Indian Railways",
        json_schema_extra={"examples": ["Which trains run between Vijayawada and Hyderabad?"]},
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session identifier to track train context across queries"
    )


class SourceInfo(BaseModel):
    type: str
    relevance_score: float
    train_no: str | None = None
    train_name: str | None = None
    station_code: str | None = None
    station_name: str | None = None
    category: str | None = None
    rule_title: str | None = None


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceInfo]
    num_documents_retrieved: int
    response_time_ms: float = 0.0
    avg_relevance_score: float = 0.0
    llm_model: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"


class HealthResponse(BaseModel):
    status: str
    message: str
    version: str
    llm_provider: str = ""
    llm_model: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_db: str = "ChromaDB"
    total_documents: int = 0
    collections: dict = {}


# --- App Lifecycle ---

# Singleton for the RAG chain (initialized at startup)
rag_chain = None

# Session context memory for tracking the last queried train number
_session_last_train: dict[str, str] = {}
_SESSION_MAX_SIZE = 1000  # max unique sessions to keep in memory

# Conversation memory — stores last N Q&A pairs per session for multi-turn chat
_session_history: dict[str, list[dict]] = {}  # {session_key: [{"q": ..., "a": ...}, ...]}
_HISTORY_MAX_TURNS = 5  # keep last 5 exchanges per session



import threading
import asyncio

_rag_lock = threading.Lock()
_rag_loading = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start a background thread to warm up RAG chain — port opens immediately."""
    logger.info("Railway RAG Assistant -- Starting up (lazy mode)...")
    logger.info("Port will open immediately. RAG chain warms up in background.")
    logger.info("API is live — Swagger UI at /docs")
    # Kick off background warm-up so first query isn't slow
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _warmup_rag_chain)
    yield  # App runs here
    logger.info("Shutting down Railway RAG Assistant...")
    global rag_chain
    rag_chain = None


def _warmup_rag_chain():
    """Background thread: pre-load RAG chain so first query is instant."""
    global rag_chain, _rag_loading
    with _rag_lock:
        if rag_chain is not None or _rag_loading:
            return
        _rag_loading = True
    try:
        logger.info("[warmup] Pre-loading RAG chain in background thread...")
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if provider == "gemini" and (not api_key or api_key == "your-gemini-api-key-here"):
            logger.error("[warmup] GOOGLE_API_KEY not set — RAG chain not loaded.")
            return
        from app.rag import get_rag_chain
        with _rag_lock:
            rag_chain = get_rag_chain()
        logger.info(f"[warmup] RAG chain ready! Provider: {provider.upper()}")
    except Exception as e:
        logger.error(f"[warmup] Failed to load RAG chain: {e}")
    finally:
        _rag_loading = False


def _ensure_rag_chain():
    """Ensure RAG chain is loaded — waits if background warmup is still running."""
    global rag_chain
    if rag_chain is not None:
        return
    # If warmup is still in progress, wait for it (up to 120s)
    for _ in range(120):
        import time
        time.sleep(1)
        if rag_chain is not None:
            return
        if not _rag_loading:
            break
    if rag_chain is None:
        # Warmup didn't work, try loading synchronously
        _warmup_rag_chain()
    if rag_chain is None:
        raise RuntimeError("RAG chain could not be initialized. Check GOOGLE_API_KEY.")


# --- FastAPI App ---

app = FastAPI(
    title="Railway RAG Assistant",
    description=(
        "AI-Powered Indian Railway Information Retrieval System using "
        "Retrieval-Augmented Generation (RAG) with LangChain, ChromaDB, "
        "and Google Gemini API."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
)


from fastapi.staticfiles import StaticFiles
# Mount static web UI files
app.mount("/web", StaticFiles(directory="web"), name="web")


# --- Data Helpers ---

DATA_DIR             = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATA_COLLECTIONS_DIR = os.getenv(
    "DATA_COLLECTIONS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"),  # fallback to ./data/
)


def load_csv(filename: str, directory: str = None) -> list[dict]:
    """Load a CSV file and return as list of dicts."""
    base_dir = directory or DATA_DIR
    filepath = os.path.join(base_dir, filename)
    if not os.path.exists(filepath):
        return []
    df = pd.read_csv(filepath, low_memory=False)
    return df.to_dict(orient="records")


def extract_token_text(chunk) -> str:
    """Extract string content from a streaming chunk, handling list-block formats."""
    if not hasattr(chunk, "content"):
        return str(chunk)
    content = chunk.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            elif hasattr(block, "get"):
                try:
                    parts.append(block.get("text", ""))
                except Exception:
                    parts.append(getattr(block, "text", str(block)))
            else:
                parts.append(getattr(block, "text", str(block)))
        return "".join(parts)
    return str(content)


# --- Endpoints ---

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def root():
    """Health check and welcome endpoint."""
    return HealthResponse(
        status="ok",
        message="Railway RAG Assistant is running! Use POST /ask to query.",
        version="1.0.0",
    )


@app.post("/ask", response_model=AnswerResponse, tags=["RAG"])
async def ask_question(request: QuestionRequest):
    """
    Main RAG query endpoint.

    Send a natural language question about Indian Railways and get an
    AI-generated answer backed by retrieved documents.

    **Example questions:**
    - Which trains run between Vijayawada and Hyderabad?
    - Tell me about train 12727.
    - Which station comes after Eluru?
    - Show trains stopping at Rajahmundry.
    - What are the cancellation charges for AC tickets?
    - What is the luggage allowance for Sleeper class?
    - What are the responsibilities of a TTE?
    """
    try:
        _ensure_rag_chain()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        import time
        t0 = time.time()
        result = rag_chain.invoke(request.question)
        elapsed_ms = round((time.time() - t0) * 1000, 1)

        # Compute avg relevance score from sources
        scores = [s["relevance_score"] for s in result.get("sources", []) if isinstance(s.get("relevance_score"), (int, float))]
        avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0

        return AnswerResponse(
            **result,
            response_time_ms=elapsed_ms,
            avg_relevance_score=avg_score,
            llm_model=os.getenv("GEMINI_MODEL", os.getenv("LOCAL_MODEL_NAME", "gemini-3.1-flash-lite")),
            embedding_model="all-MiniLM-L6-v2",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}",
        )


@app.post("/ask/stream", tags=["RAG"])
async def ask_question_stream(request: QuestionRequest):
    """
    Streaming RAG endpoint — returns answer token-by-token via Server-Sent Events.
    Connect with EventSource in the browser for real-time word-by-word output.
    """
    from fastapi.responses import StreamingResponse
    import json, time

    try:
        _ensure_rag_chain()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    async def event_stream():
        try:
            t0 = time.time()

            # Step 1: Retrieve docs
            docs = rag_chain.retriever.retrieve(request.question)

            # Send sources immediately (before LLM starts)
            from app.rag import get_sources, format_docs
            sources = get_sources(docs)
            scores  = [s["relevance_score"] for s in sources if isinstance(s.get("relevance_score"), (int, float))]
            avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0

            meta = {
                "type": "meta",
                "num_documents_retrieved": len(docs),
                "avg_relevance_score": avg_score,
                "sources": sources,
                "llm_model": os.getenv("GEMINI_MODEL", os.getenv("LOCAL_MODEL_NAME", "gemini-3.1-flash-lite")),
                "embedding_model": "all-MiniLM-L6-v2",
            }
            yield f"data: {json.dumps(meta)}\n\n"

            # Step 2: Stream LLM tokens
            from app.rag import SYSTEM_PROMPT, HUMAN_PROMPT
            from langchain_core.prompts import ChatPromptTemplate
            context = format_docs(docs)
            prompt  = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human",  HUMAN_PROMPT),
            ])
            chain = prompt | rag_chain.llm

            full_answer = ""
            async for chunk in chain.astream({"context": context, "question": request.question}):
                token = extract_token_text(chunk)
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

            # Step 3: Send done event
            elapsed_ms = round((time.time() - t0) * 1000, 1)
            yield f"data: {json.dumps({'type': 'done', 'response_time_ms': elapsed_ms})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/ask/smart", tags=["RAG"])
async def ask_question_smart(request: QuestionRequest, raw_request: Request):
    """
    Intelligent intent-routed RAG endpoint.
    Classifies the user's question first:
      - STATIC: Retrieves relevant documents from ChromaDB, queries LLM.
      - LIVE: Fetches real-time status from NTES, queries LLM.
      - HYBRID: Fetches both NTES status and ChromaDB docs, merges them, queries LLM.
    Streams the output token-by-token.
    """
    from fastapi.responses import StreamingResponse
    import json, time
    from app.intent import classify_intent
    from app.ntes_client import get_train_running_status, format_live_status_for_llm

    try:
        _ensure_rag_chain()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Get session key (fallback to client IP)
    client_ip = raw_request.client.host if raw_request.client else "unknown"
    session_key = request.session_id or client_ip

    async def event_stream():
        try:
            t0 = time.time()
            question = request.question

            # 1. Classify intent
            intent_res = classify_intent(question)
            intent = intent_res["intent"]
            train_no = intent_res["train_no"]

            # If train number isn't explicitly mentioned, try to load it from the session context
            if not train_no:
                train_no = _session_last_train.get(session_key)
                if train_no:
                    # Enrich intent details
                    intent_res["train_no"] = train_no
                    if intent == "STATIC" and "stops" in question.lower() or "route" in question.lower() or "schedule" in question.lower():
                        # If asking about stops/route/schedule dynamically, upgrade to HYBRID to include both details
                        intent = "HYBRID"

            # If a train number was explicitly matched or resolved, save it to session
            if train_no:
                _session_last_train[session_key] = train_no
                # Guard against unbounded session memory growth
                if len(_session_last_train) > _SESSION_MAX_SIZE:
                    # Remove oldest 200 entries
                    for k in list(_session_last_train.keys())[:200]:
                        del _session_last_train[k]

            # 2. Retrieve data based on intent
            docs = []
            live_status = None
            live_context = ""
            sources = []
            warnings = []

            is_pnr = intent_res.get("is_pnr", False)
            pnr_val = intent_res.get("pnr")

            if is_pnr and pnr_val:
                try:
                    pnr_status = get_pnr_status(pnr_val)
                    if pnr_status.get("success"):
                        live_context = format_pnr_status_for_llm(pnr_status)
                        sources.insert(0, {
                            "type": "pnr_status",
                            "relevance_score": 1.0,
                            "pnr": pnr_val,
                            "train_no": pnr_status.get("train_no", ""),
                            "train_name": pnr_status.get("train_name", ""),
                            "date_of_journey": pnr_status.get("date_of_journey", ""),
                            "chart_prepared": pnr_status.get("chart_prepared", False),
                            "passengers": pnr_status.get("passengers", []),
                            "source": pnr_status.get("source", "ConfirmTkt"),
                            "fetched_at": pnr_status.get("fetched_at", "")
                        })
                        # Extract and save train number from the verified PNR status to session
                        retrieved_train = pnr_status.get("train_no")
                        if retrieved_train:
                            _session_last_train[session_key] = retrieved_train
                    else:
                        warnings.append(f"PNR status fetch failed: {pnr_status.get('error')}")
                        live_context = f"⚠️ LIVE PNR STATUS UNAVAILABLE: {pnr_status.get('error')}"
                except Exception as exc:
                    warnings.append(f"PNR status fetch failed due to error: {str(exc)}")
                    live_context = f"⚠️ LIVE PNR STATUS UNAVAILABLE due to error: {str(exc)}"
            else:
                # Determine static retrieval
                needs_static = (intent in ("STATIC", "HYBRID"))
                # Determine live status fetch
                needs_live = (intent in ("LIVE", "HYBRID")) and train_no is not None

                # Executing static retrieval if needed
                if needs_static:
                    # Append train number to retrieval question if it's not present in original query
                    query_for_retrieval = question
                    if train_no and train_no not in question:
                        query_for_retrieval = f"{question} train {train_no}"
                    
                    docs = rag_chain.retriever.retrieve(query_for_retrieval)
                    from app.rag import get_sources
                    sources = get_sources(docs)

                # Executing live status fetch if needed
                if needs_live:
                    try:
                        live_status = get_train_running_status(train_no)
                        if live_status.get("success"):
                            live_context = format_live_status_for_llm(live_status)
                            # Add a special LIVE source entry
                            sources.insert(0, {
                                "type": "live_status",
                                "relevance_score": 1.0,
                                "train_no": train_no,
                                "train_name": live_status.get("train_name", ""),
                                "status": live_status.get("status", ""),
                                "current_station": live_status.get("current_station", ""),
                                "source": live_status.get("source", "NTES"),
                                "fetched_at": live_status.get("fetched_at", "")
                            })
                        else:
                            warnings.append(f"Live status fetch failed: {live_status.get('error')}")
                    except Exception as exc:
                        warnings.append(f"Live status fetch failed due to error: {str(exc)}")

                # Fallback if live status was needed but failed / was not found
                if needs_live and not live_context:
                    live_context = (
                        f"⚠️ LIVE RUNNING STATUS UNAVAILABLE for train {train_no or 'unknown'}.\n"
                        "Please rely on static scheduled timetable database details only.\n"
                    )

            # Compute avg relevance score
            scores = [s["relevance_score"] for s in sources if isinstance(s.get("relevance_score"), (int, float))]
            avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0

            # Send metadata SSE event
            meta = {
                "type": "meta",
                "intent": intent,
                "confidence": intent_res["confidence"],
                "train_no": train_no,
                "num_documents_retrieved": len(docs),
                "avg_relevance_score": avg_score,
                "sources": sources,
                "warnings": warnings,
                "llm_model": os.getenv("GEMINI_MODEL" if os.getenv("LLM_PROVIDER") == "gemini" else "LOCAL_MODEL_NAME", "gemini-2.5-flash"),
                "embedding_model": "all-MiniLM-L6-v2",
            }
            yield f"data: {json.dumps(meta)}\n\n"

            # 3. Format prompt context
            from app.rag import SYSTEM_PROMPT, HUMAN_PROMPT
            from langchain_core.prompts import ChatPromptTemplate
            from app.rag import format_docs

            static_context = format_docs(docs) if docs else "No static database context."

            # Merge static + live contexts into final prompt context
            context_parts = []
            if live_context:
                context_parts.append(live_context)
            if static_context:
                context_parts.append(f"=== DATABASE RULES/TIMETABLES (STATIC) ===\n{static_context}")

            # Inject conversation history for multi-turn context
            history = _session_history.get(session_key, [])
            history_context = ""
            if history:
                history_lines = []
                for h in history[-_HISTORY_MAX_TURNS:]:
                    history_lines.append(f"User: {h['q']}")
                    # Truncate long answers to keep context manageable
                    ans_preview = h['a'][:300] + '...' if len(h['a']) > 300 else h['a']
                    history_lines.append(f"Assistant: {ans_preview}")
                history_context = "\n=== CONVERSATION HISTORY (for context) ===\n" + "\n".join(history_lines) + "\n"

            merged_context = "\n\n".join(filter(None, [history_context, live_context, f"=== DATABASE RULES/TIMETABLES (STATIC) ===\n{static_context}" if static_context else None]))

            prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", HUMAN_PROMPT),
            ])
            chain = prompt | rag_chain.llm

            full_answer = ""
            async for chunk in chain.astream({"context": merged_context, "question": question}):
                token = extract_token_text(chunk)
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

            # Save this Q&A pair to conversation history
            if session_key not in _session_history:
                _session_history[session_key] = []
            _session_history[session_key].append({"q": question, "a": full_answer})
            # Trim to max turns
            if len(_session_history[session_key]) > _HISTORY_MAX_TURNS:
                _session_history[session_key] = _session_history[session_key][-_HISTORY_MAX_TURNS:]
            # Guard against unbounded session growth
            if len(_session_history) > _SESSION_MAX_SIZE:
                for k in list(_session_history.keys())[:200]:
                    del _session_history[k]
            # Send done event
            elapsed_ms = round((time.time() - t0) * 1000, 1)
            yield f"data: {json.dumps({'type': 'done', 'response_time_ms': elapsed_ms})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/ask/upload", tags=["RAG"])
async def ask_with_file(
    file: UploadFile = File(...),
    question: str = Form(default=""),
):
    """
    Multi-modal RAG endpoint — accepts an image (PNG/JPG/WEBP) or PDF file
    along with an optional text question.

    Gemini Vision reads the file and generates an answer.
    If a train number or station is detected, ChromaDB context is also provided.

    **Supported file types:** PNG, JPG, JPEG, WEBP, PDF
    **Max file size:** 10 MB
    """
    import time
    from google import genai
    from google.genai import types

    # Validate
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key or api_key == "your-gemini-api-key-here":
        raise HTTPException(status_code=503, detail="GOOGLE_API_KEY not configured for multi-modal.")

    allowed_types = {
        "image/png": "png", "image/jpeg": "jpeg", "image/jpg": "jpg",
        "image/webp": "webp", "application/pdf": "pdf",
    }
    content_type = file.content_type or ""
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. Supported: PNG, JPG, WEBP, PDF",
        )

    # Read file bytes (max 10MB)
    file_bytes = await file.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum 10 MB.")

    try:
        t0 = time.time()
        import re, io, PIL.Image

        # Configure Gemini (new SDK — google.genai)
        client = genai.Client(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        effective_question = question.strip()

        # ── Step 1: Single Gemini call — describe + classify railway relevance ──
        # Returns a structured block we parse without a second API call.
        classification_prompt = (
            "Analyze this file carefully and respond in EXACTLY this format (no extra text):\n\n"
            "DESCRIPTION: <1-2 sentence description of what is in the image/PDF>\n"
            "IS_RAILWAY: YES or NO\n"
            "QUERY: <if railway-related, the core railway topic or question; otherwise leave blank>\n\n"
            "Guidelines:\n"
            "- IS_RAILWAY is YES only if the content is about Indian Railways: train tickets, "
            "timetables, PNR status, station info, railway rules, track maps, fare charts, etc.\n"
            "- IS_RAILWAY is NO for anything else: food, architecture, software diagrams, "
            "computer science topics, personal photos, nature, etc.\n"
            "- DESCRIPTION must always be filled in (even for non-railway content)."
        )

        # Build the classification request (image or PDF)
        gen_config = types.GenerateContentConfig(temperature=0.1, max_output_tokens=256)
        if content_type == "application/pdf":
            pdf_part = types.Part.from_bytes(data=file_bytes, mime_type="application/pdf")
            cls_resp = client.models.generate_content(
                model=model_name,
                contents=[classification_prompt, pdf_part],
                config=gen_config,
            )
        else:
            pil_img = PIL.Image.open(io.BytesIO(file_bytes))
            cls_resp = client.models.generate_content(
                model=model_name,
                contents=[classification_prompt, pil_img],
                config=gen_config,
            )

        cls_text = cls_resp.text.strip() if cls_resp.text else ""
        logger.debug(f"Classification response:\n{cls_text}")

        # Parse structured response
        def _parse_field(label: str, text: str) -> str:
            m = re.search(rf"^{label}:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
            return m.group(1).strip() if m else ""

        description  = _parse_field("DESCRIPTION", cls_text)
        is_railway   = _parse_field("IS_RAILWAY", cls_text).upper().startswith("YES")
        extracted_q  = _parse_field("QUERY", cls_text)

        elapsed_ms = round((time.time() - t0) * 1000, 1)

        # ── Step 2: Non-railway content → skip ChromaDB, return closed-domain message ──
        if not is_railway:
            logger.info("Non-railway content detected — skipping ChromaDB retrieval")
            answer_text = (
                f"📋 **I can see:** {description}\n\n"
                "⚠️ **This is a closed-domain assistant for Indian Railways only.**\n"
                "The uploaded file doesn't appear to contain railway-related information. "
                "Please upload a railway ticket, timetable, station screenshot, or fare chart — "
                "or type a railway-related question directly."
            )
            return {
                "question": question or "Analyze this file",
                "answer": answer_text,
                "sources": [],
                "num_documents_retrieved": 0,
                "response_time_ms": elapsed_ms,
                "avg_relevance_score": 0.0,
                "llm_model": model_name,
                "embedding_model": "all-MiniLM-L6-v2",
                "file_name": file.filename,
                "file_type": content_type,
                "mode": "multi-modal",
                "railway_related": False,
            }

        # ── Step 3: Railway content → ChromaDB retrieval ──────────────────────
        retrieval_query = " ".join(filter(None, [effective_question, extracted_q, description]))
        rag_context = ""
        sources = []
        if rag_chain and retrieval_query.strip():
            docs = rag_chain.retriever.retrieve(retrieval_query)
            if docs:
                rag_context = format_docs(docs)
                sources = get_sources(docs)

        # ── Step 4: Full Gemini Vision answer for railway content ─────────────
        system_instruction = (
            "You are an expert Indian Railways assistant (closed domain).\n"
            "The user has uploaded a railway-related file.\n\n"
            f"What you can see in the file: {description}\n\n"
            "INSTRUCTIONS:\n"
            "1. Start your answer by briefly confirming what you see in the file (1 sentence).\n"
            "2. Then answer the user's question thoroughly using the file content.\n"
            "3. Use train numbers, station codes, times, and PNR details where visible.\n"
            "4. If additional database context is provided below, use it to enrich your answer.\n"
        )
        if rag_context:
            system_instruction += (
                "\n=== RAILWAY DATABASE CONTEXT (use if relevant) ===\n"
                + rag_context[:4000]
            )

        final_question = effective_question or extracted_q or "Please analyse this railway document."

        gen_config_full = types.GenerateContentConfig(temperature=0.3, max_output_tokens=2048)
        if content_type == "application/pdf":
            pdf_part = types.Part.from_bytes(data=file_bytes, mime_type="application/pdf")
            response = client.models.generate_content(
                model=model_name,
                contents=[system_instruction, pdf_part, final_question],
                config=gen_config_full,
            )
        else:
            image = PIL.Image.open(io.BytesIO(file_bytes))
            response = client.models.generate_content(
                model=model_name,
                contents=[system_instruction, image, final_question],
                config=gen_config_full,
            )

        elapsed_ms = round((time.time() - t0) * 1000, 1)
        answer_text = response.text if response.text else "Could not process the file."

        scores = [s["relevance_score"] for s in sources if isinstance(s.get("relevance_score"), (int, float))]
        avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0

        return {
            "question": question or final_question,
            "answer": answer_text,
            "sources": sources,
            "num_documents_retrieved": len(sources),
            "response_time_ms": elapsed_ms,
            "avg_relevance_score": avg_score,
            "llm_model": model_name,
            "embedding_model": "all-MiniLM-L6-v2",
            "file_name": file.filename,
            "file_type": content_type,
            "mode": "multi-modal",
            "railway_related": True,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/live/pnr/{pnr}", tags=["Live Status"])
async def live_pnr_status(pnr: str):
    """Fetch live PNR status from ConfirmTkt or fallback scrapers."""
    from app.pnr_client import get_pnr_status
    res = get_pnr_status(pnr)
    if not res.get("success"):
        raise HTTPException(status_code=400, detail=res.get("error", "Failed to fetch PNR status"))
    return res


@app.get("/trains", tags=["Data"])
async def list_trains(limit: int = 100, offset: int = 0):
    """List trains from the 12k dataset. Use ?limit= and ?offset= for pagination."""
    trains = load_csv("train_info.csv", directory=DATA_COLLECTIONS_DIR)
    if not trains:
        raise HTTPException(status_code=404, detail="train_info.csv not found. Check DATA_COLLECTIONS_DIR in .env")
    page = trains[offset: offset + limit]
    return {
        "total" : len(trains),
        "offset": offset,
        "limit" : limit,
        "trains": page,
    }


@app.get("/stations", tags=["Data"])
async def list_stations(limit: int = 100, offset: int = 0):
    """List stations from the 10k dataset. Use ?limit= and ?offset= for pagination."""
    stations = load_csv("station_info.csv", directory=DATA_COLLECTIONS_DIR)
    if not stations:
        raise HTTPException(status_code=404, detail="station_info.csv not found. Check DATA_COLLECTIONS_DIR in .env")
    page = stations[offset: offset + limit]
    return {
        "total"   : len(stations),
        "offset"  : offset,
        "limit"   : limit,
        "stations": page,
    }


@app.get("/rules", tags=["Data"])
async def list_rules():
    """List all railway rules in the dataset."""
    rules = load_csv("railway_rules.csv")
    if not rules:
        raise HTTPException(status_code=404, detail="Rules dataset not found")
    return {
        "count": len(rules),
        "rules": rules,
    }


@app.get("/trains/{train_no}", tags=["Data"])
async def get_train(train_no: str):
    """Get details of a specific train by train number."""
    trains = load_csv("train_info.csv", directory=DATA_COLLECTIONS_DIR)
    matching = [t for t in trains if str(t.get("train_no") or t.get("trainNo") or "") == train_no]
    if not matching:
        raise HTTPException(
            status_code=404, detail=f"Train {train_no} not found in dataset"
        )
    return {
        "count" : len(matching),
        "trains": matching,
    }


@app.get("/stations/{station_code}", tags=["Data"])
async def get_station(station_code: str):
    """Get details of a specific station by station code."""
    stations = load_csv("station_info.csv", directory=DATA_COLLECTIONS_DIR)
    station_code_upper = station_code.upper()
    matching = [
        s for s in stations
        if str(s.get("station_code") or s.get("code") or "").upper() == station_code_upper
    ]
    if not matching:
        raise HTTPException(
            status_code=404,
            detail=f"Station '{station_code}' not found. Check if DATA_COLLECTIONS_DIR is set correctly.",
        )
    return matching[0]


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Detailed health check — shows LLM provider, model info, and collection stats."""
    import chromadb
    chroma_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
    collections_detail = {}
    total_docs = 0

    if os.path.exists(chroma_dir):
        try:
            client = chromadb.PersistentClient(path=chroma_dir)
            for c in client.list_collections():
                count = client.get_collection(c.name).count()
                collections_detail[c.name] = count
                total_docs += count
        except Exception:
            pass

    provider   = os.getenv("LLM_PROVIDER", "gemini")
    llm_model  = os.getenv("LOCAL_MODEL_NAME", "gemini-1.5-flash") if provider == "lmstudio" else os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    col_summary = [f"{k} ({v} docs)" for k, v in collections_detail.items()]
    status_msg  = f"LLM: {provider.upper()} | ChromaDB collections: {col_summary or 'none — run create_embeddings.py'}"

    return HealthResponse(
        status="ok",
        message=status_msg,
        version="1.0.0",
        llm_provider=provider.upper(),
        llm_model=llm_model,
        embedding_model="all-MiniLM-L6-v2 (sentence-transformers)",
        vector_db="ChromaDB",
        total_documents=total_docs,
        collections=collections_detail,
    )
