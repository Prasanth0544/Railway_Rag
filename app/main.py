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

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
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
    allow_headers=["*"],
)


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
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
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
