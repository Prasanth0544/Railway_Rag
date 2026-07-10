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


# Load environment variables before anything else
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG chain once at startup, cleanup on shutdown."""
    global rag_chain

    print("\n[STARTUP] Railway RAG Assistant -- Starting up...")
    print("=" * 50)

    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    print(f"   LLM Provider : {provider.upper()}")

    if provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key or api_key == "your-gemini-api-key-here":
            print("[ERROR] GOOGLE_API_KEY not set! Please update your .env file.")
            print("   Get a free key at: https://aistudio.google.com")
        else:
            from app.rag import get_rag_chain
            rag_chain = get_rag_chain()
            print("\n[OK] RAG chain ready! (Gemini)")
    else:
        # LM Studio -- no API key needed
        base_url = os.getenv("LOCAL_API_BASE", "http://localhost:1234/v1")
        print(f"   LM Studio    : {base_url}")
        print("   Make sure LM Studio is running with a model loaded!")
        from app.rag import get_rag_chain
        rag_chain = get_rag_chain()
        print("\n[OK] RAG chain ready! (LM Studio)")

    print("=" * 50)
    print("[LIVE] API is live at http://localhost:8000")
    print("[DOCS] Swagger UI at http://localhost:8000/docs\n")

    yield  # App runs here

    # Shutdown
    print("\n[STOP] Shutting down Railway RAG Assistant...")
    rag_chain = None


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
    r"C:\Users\prasa\Documents\RailWayData\csv_of_railway\data_collections",
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
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG chain not initialized. Check your GOOGLE_API_KEY in .env",
        )

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
            llm_model=os.getenv("LOCAL_MODEL_NAME", "gemini-1.5-flash"),
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

    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized.")

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
                "llm_model": os.getenv("LOCAL_MODEL_NAME", "gemini-1.5-flash"),
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

    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized.")

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

            # 2. Retrieve data based on intent
            docs = []
            live_status = None
            live_context = ""
            sources = []
            warnings = []

            is_pnr = intent_res.get("is_pnr", False)
            pnr_val = intent_res.get("pnr")

            if is_pnr and pnr_val:
                from app.pnr_client import get_pnr_status, format_pnr_status_for_llm
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

            merged_context = "\n\n".join(context_parts)

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
    import google.generativeai as genai

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

        # Configure Gemini
        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        model = genai.GenerativeModel(model_name)

        # ── Step 1: Extract intent from the file (quick Gemini Vision call) ──────
        # Always run a quick extraction to get the real question from the image/PDF,
        # then combine with any typed question for retrieval.
        import re, io
        effective_question = question.strip()

        # Always extract from the file so we can retrieve relevant ChromaDB context
        extraction_prompt = (
            "Look at this file carefully. "
            "If it contains a question or query (e.g. text written on a whiteboard, "
            "screenshot of a typed question, or a railway ticket), "
            "extract and return ONLY that question or query as plain text. "
            "If it is a railway ticket, return: 'Train ticket for train [number] from [source] to [destination]'. "
            "If you cannot determine a question, return: 'Analyze this railway document and describe what you see.'"
        )
        if content_type == "application/pdf":
            ext_resp = model.generate_content(
                [extraction_prompt, {"mime_type": "application/pdf", "data": file_bytes}]
            )
        else:
            import PIL.Image
            img_for_extract = PIL.Image.open(io.BytesIO(file_bytes))
            ext_resp = model.generate_content([extraction_prompt, img_for_extract])
        extracted = ext_resp.text.strip() if ext_resp.text else ""

        # Merge: prefer user's typed question + extracted context for retrieval
        if effective_question and extracted:
            retrieval_query = f"{effective_question} {extracted}"
        elif extracted:
            retrieval_query = extracted
            effective_question = extracted  # use extracted as the question if nothing typed
        else:
            retrieval_query = effective_question or "Indian railway information"

        # ── Step 2: Use merged query for ChromaDB retrieval ────────────────
        rag_context = ""
        sources = []
        if rag_chain:
            docs = rag_chain.retriever.retrieve(retrieval_query)
            if docs:
                from app.rag import format_docs, get_sources
                rag_context = format_docs(docs)
                sources = get_sources(docs)

        system_instruction = (
            "You are an expert Indian Railways assistant. Analyze the uploaded file "
            "(image or PDF) and answer the user's question about it. "
            "Be thorough and extract all relevant details.\n"
        )
        if rag_context:
            system_instruction += (
                "\nAdditional railway database context (use if relevant):\n"
                + rag_context[:4000]  # cap context size
            )

        # Send to Gemini Vision
        import PIL.Image
        import io

        if content_type == "application/pdf":
            # For PDFs, upload the raw bytes
            response = model.generate_content(
                [
                    system_instruction,
                    {"mime_type": "application/pdf", "data": file_bytes},
                    question,
                ],
                generation_config={"temperature": 0.3, "max_output_tokens": 2048},
            )
        else:
            # For images, open with PIL
            image = PIL.Image.open(io.BytesIO(file_bytes))
            response = model.generate_content(
                [system_instruction, image, effective_question],
                generation_config={"temperature": 0.3, "max_output_tokens": 2048},
            )

        elapsed_ms = round((time.time() - t0) * 1000, 1)
        answer_text = response.text if response.text else "Could not process the file."

        # Compute avg relevance score from sources
        scores = [s["relevance_score"] for s in sources if isinstance(s.get("relevance_score"), (int, float))]
        avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0

        return {
            "question": question,
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
