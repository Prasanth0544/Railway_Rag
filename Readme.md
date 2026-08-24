# 🚂 Railway RAG Assistant

> **AI-Powered Indian Railways Information System**
> Hybrid RAG · FastAPI · LangChain · ChromaDB · Google Gemini · OpenRouter · Live APIs

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-orange)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-purple)](https://www.trychroma.com)
[![Gemini](https://img.shields.io/badge/Gemini-Flash-blue?logo=google)](https://aistudio.google.com)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-ox--alpha-purple)](https://openrouter.ai)
[![Render](https://img.shields.io/badge/Deployed-Render.com-46E3B7?logo=render)](https://railway-rag.onrender.com)

---

## What is This?

A **production-ready Hybrid RAG** assistant for Indian Railways. Ask any question in plain English and get grounded, accurate answers.

**Intelligence comes from:**
- **35,780+ indexed documents** in ChromaDB (5 collections)
- **12,738 train route docs** with full per-stop arr/dep/halt schedule times
- **Switchable LLM** — Gemini Flash (default), OpenRouter `stealth/ox-alpha`, or LM Studio (local)
- **3072-dimensional embeddings** via `gemini-embedding-001` *(verified live: 3072 dims confirmed)*

> **Embedding model verified:** `gemini-embedding-001` outputs **3072 dimensions** (confirmed via live API).
> `text-embedding-004` returns HTTP 404 on this API — do NOT use it as a replacement.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10, FastAPI, Uvicorn |
| **RAG Framework** | LangChain (LCEL) |
| **LLM (default)** | Google Gemini Flash (`gemini-3.6-flash`) |
| **LLM (cloud alt)** | OpenRouter `stealth/ox-alpha` (free, 1M context) |
| **LLM (local/offline)** | LM Studio |
| **Vector DB** | ChromaDB (persistent, 5 collections) |
| **Embeddings** | `gemini-embedding-001` — **3072 dims** (verified) |
| **Live Data (Local)** | NTES Direct — enquiry.indianrail.gov.in |
| **Live Data (Cloud)** | RapidAPI — irctc-indian-railway-pnr-status |
| **Frontend** | Vanilla HTML5, CSS3, JavaScript |
| **Deployment** | Render.com |

---

## Key Features

| Feature | Description |
|---|---|
| **Hybrid Retrieval** | Vector + Keyword + Metadata fused via RRF |
| **Intent Classifier v2** | 11 fine-grained categories |
| **Smart Context Builder** | 6 query-type strategies with char budgets |
| **LLM Flexibility** | Gemini Flash / OpenRouter ox-alpha / LM Studio — switch via `LLM_PROVIDER`. Embeddings always use `gemini-embedding-001`. |
| **Auto LLM Badge** | Sidebar badge auto-updates from `/health` — colour-coded per provider (blue=Gemini, purple=OpenRouter, amber=LM Studio) |
| **SSE Streaming** | Real-time word-by-word rendering |
| **Route Trimming** | 80%+ token reduction for route docs |
| **PNR Support** | Live booking status |
| **Multi-modal** | Image/PDF upload with Gemini Vision |
| **Rate Limiting** | 15 req/min per IP |
| **Response Cache** | 10-min TTL for STATIC queries |

---

## Knowledge Base

| Collection | Docs | Content |
|---|---|---|
| trains | 12,813 | Train info, zones, schedules |
| stations | 9,956 | Station codes, names, AKAs |
| train_routes | 12,738 | Full per-stop arr/dep/halt times |
| railway_rules | 183 | Rules, cancellation, luggage |
| references | 90 | Ticket classes, service tax |
| **Total** | **35,780** | |

---

## Local Setup

```powershell
# 1. Clone
git clone https://github.com/Prasanth0544/Railway_Rag.git
cd Railway_Rag

# 2. Install
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Configure .env (see section below)

# 4. Build ChromaDB (one-time)
.venv\Scripts\python scripts/create_embeddings.py --skip-routes
.venv\Scripts\python scripts/embed_routes.py

# 5. Start server
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open: **http://127.0.0.1:8000/web/index.html**

---

## Environment Configuration

```env
# LLM Provider — gemini | openrouter | lmstudio
LLM_PROVIDER=gemini

# Google Gemini (ALWAYS required — used for gemini-embedding-001 embeddings)
GOOGLE_API_KEY=your-key-from-aistudio.google.com
GEMINI_MODEL=gemini-3.6-flash

# OpenRouter (when LLM_PROVIDER=openrouter)
OPENROUTER_API_KEY=your-openrouter-key
OPENROUTER_MODEL=stealth/ox-alpha

# LM Studio (when LLM_PROVIDER=lmstudio)
LOCAL_API_BASE=http://localhost:1234/v1
LOCAL_MODEL_NAME=google/gemma-2-9b

# Optional
RAPIDAPI_KEY=your-key   # For live train data on Render cloud
DATA_COLLECTIONS_DIR=path/to/csv_files
```

> `GOOGLE_API_KEY` is **always required** even when using OpenRouter or LM Studio,
> because `gemini-embedding-001` (embeddings) always uses the Gemini API.

### Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini`, `openrouter`, or `lmstudio` |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Gemini model (LLM_PROVIDER=gemini) |
| `OPENROUTER_API_KEY` | — | OpenRouter API key (LLM_PROVIDER=openrouter) |
| `OPENROUTER_MODEL` | `stealth/ox-alpha` | OpenRouter model ID (1M ctx, free) |
| `LOCAL_API_BASE` | `http://localhost:1234/v1` | LM Studio URL |
| `LOCAL_MODEL_NAME` | `google/gemma-2-9b` | LM Studio model |
| `RAPIDAPI_KEY` | — | Live train data for Render cloud |
| `MALLOC_ARENA_MAX` | — | Set 2 on Linux to save ~30-50MB RAM |
| `USE_LOCAL_EMBEDDINGS` | *removed* | Removed; `gemini-embedding-001` used exclusively |

---

## Cloud Deployment (Render.com)

Set in Render dashboard → Environment:

| Key | Required | Notes |
|-----|----------|-------|
| `GOOGLE_API_KEY` | Always | For `gemini-embedding-001` embeddings |
| `OPENROUTER_API_KEY` | If LLM_PROVIDER=openrouter | From openrouter.ai |
| `RAPIDAPI_KEY` | Recommended | For live train data from cloud IPs |

`render.yaml` sets:
- `LLM_PROVIDER=openrouter`
- `OPENROUTER_MODEL=stealth/ox-alpha`
- `MALLOC_ARENA_MAX=2`

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | LLM provider/model info, collection stats |
| `POST` | `/ask` | Standard RAG query |
| `POST` | `/ask/smart` | SSE streaming (intent-aware) |
| `POST` | `/ask/upload` | Multi-modal (image/PDF) |
| `GET` | `/trains/{train_no}` | Train details |
| `GET` | `/stations/{code}` | Station details |
| `GET` | `/admin/stats` | Query analytics |
| `POST` | `/clear-cache` | Clear cache + session history |

---

## License

Educational and portfolio purposes.

---

> Built with FastAPI, LangChain, ChromaDB, Google Gemini, OpenRouter, and RapidAPI.
> Live at: **https://railway-rag.onrender.com**
