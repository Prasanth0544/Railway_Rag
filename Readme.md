# 🚂 Railway RAG Assistant

> **AI-Powered Indian Railways Information System**
> Hybrid RAG · FastAPI · LangChain · ChromaDB · MongoDB Atlas · Google Gemini · Live APIs

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-orange)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-purple)](https://www.trychroma.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?logo=mongodb)](https://www.mongodb.com/atlas)
[![Gemini](https://img.shields.io/badge/Gemini-Flash-blue?logo=google)](https://aistudio.google.com)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-ox--alpha-purple)](https://openrouter.ai)
[![Render](https://img.shields.io/badge/Deployed-Render.com-46E3B7?logo=render)](https://railway-rag.onrender.com)

---

## What is This?

A **production-ready Hybrid RAG** assistant for Indian Railways. Ask any question in plain English and get accurate, grounded answers backed by a 35,780-document knowledge base.

**Intelligence comes from:**
- **35,780+ indexed documents** in ChromaDB (5 collections)
- **12,738 train route docs** with full per-stop arr/dep/halt schedule times
- **Switchable LLM** — Gemini Flash (default), OpenRouter `stealth/ox-alpha`, or LM Studio (local)
- **3072-dimensional embeddings** via `gemini-embedding-001`
- **MongoDB Atlas** for persistent analytics, query logs, and user feedback

> **Embedding model verified:** `gemini-embedding-001` outputs **3072 dimensions** (confirmed via live API).
> `text-embedding-004` returns HTTP 404 — do NOT use it.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10, FastAPI, Uvicorn |
| **RAG Framework** | LangChain (LCEL) |
| **LLM (default)** | Google Gemini Flash (`gemini-3.6-flash`) |
| **LLM (cloud alt)** | OpenRouter `stealth/ox-alpha` (free, 1M context) |
| **LLM (local/offline)** | LM Studio |
| **Vector DB** | ChromaDB (persistent, 5 collections, local) |
| **Embeddings** | `gemini-embedding-001` — **3072 dims** (verified) |
| **Analytics DB** | MongoDB Atlas — query logs + feedback (cloud) |
| **Live Data (Local)** | NTES Direct — enquiry.indianrail.gov.in |
| **Live Data (Cloud)** | RapidAPI — irctc-indian-railway-pnr-status |
| **Frontend** | Vanilla HTML5, CSS3, JavaScript (SSE streaming) |
| **Deployment** | Render.com |

---

## Key Features

| Feature | Description |
|---|---|
| **Hybrid Retrieval** | Vector + Keyword + Metadata fused via RRF |
| **Intent Classifier v2** | 11 fine-grained intent categories |
| **Smart Context Builder** | 6 query-type strategies with char budgets |
| **LLM Flexibility** | Gemini Flash / OpenRouter ox-alpha / LM Studio — switch via `LLM_PROVIDER` |
| **SSE Streaming** | Real-time word-by-word answer rendering |
| **User Feedback** | 👍/👎 rating + optional text comment, stored in MongoDB Atlas |
| **Query Logger** | Every query logged to MongoDB `query_logs` collection |
| **Admin Dashboard** | Live analytics at `/web/admin.html` — top queries, intent distribution, feedback stats, system health |
| **System Health Panel** | Per-service status: ChromaDB · Gemini API · OpenRouter · RapidAPI · RAG Chain · MongoDB Atlas |
| **Follow-up Chips** | Smart contextual suggested queries below every answer |
| **Stream Leak Fix** | `cleanAnswerText()` strips LLM-appended chip labels from the answer bubble |
| **Route Trimming** | 80%+ token reduction for route documents |
| **PNR Support** | Live booking status via PNR number |
| **Multi-modal** | Image/PDF upload with Gemini Vision |
| **Rate Limiting** | 15 req/min per IP |
| **Response Cache** | 10-min TTL for static queries |
| **Dark / Light Theme** | System-aware theme with toggle |

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

## MongoDB Atlas Schema

Database: **`Railway_Rag`**

### `query_logs` collection
```json
{
  "ts": "2026-08-25T17:30:00Z",
  "question": "What is the luggage allowance for Sleeper class?",
  "intent": "GENERAL_INFO",
  "train_no": null,
  "response_time_ms": 1240,
  "used_live_api": false,
  "error": false,
  "hallucination_flag": false
}
```

### `feedback` collection
```json
{
  "ts": "2026-08-25T17:31:00Z",
  "question": "What are the cancellation charges?",
  "answer_preview": "A flat minimum cancellation charge...",
  "rating": "up",
  "comment": "Very helpful!",
  "session_id": "anon"
}
```

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

# 4. Build ChromaDB (one-time, takes ~20 min)
.venv\Scripts\python scripts/create_embeddings.py --skip-routes
.venv\Scripts\python scripts/embed_routes.py

# 5. Start server
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open: **http://127.0.0.1:8000/web/index.html**
Admin: **http://127.0.0.1:8000/web/admin.html**

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

# MongoDB Atlas — query logs + feedback
MONGO_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority

# Optional — Live data on Render cloud
RAPIDAPI_KEY=your-key

# Local CSV path (dev only — not needed on Render)
DATA_COLLECTIONS_DIR=path/to/csv_files
```

> `GOOGLE_API_KEY` is **always required** even when using OpenRouter or LM Studio,
> because `gemini-embedding-001` (embeddings) always uses the Gemini API.

### Configuration Reference

| Variable | Required | Description |
|---|---|---|
| `LLM_PROVIDER` | ✅ | `gemini`, `openrouter`, or `lmstudio` |
| `GOOGLE_API_KEY` | ✅ Always | For LLM + embeddings when using Gemini |
| `GEMINI_MODEL` | ✅ | Default: `gemini-3.6-flash` |
| `MONGO_URI` | ✅ | MongoDB Atlas connection string |
| `OPENROUTER_API_KEY` | If openrouter | From openrouter.ai |
| `OPENROUTER_MODEL` | If openrouter | Default: `stealth/ox-alpha` |
| `LOCAL_API_BASE` | If lmstudio | LM Studio server URL |
| `RAPIDAPI_KEY` | Recommended | Live train status on Render cloud |
| `MALLOC_ARENA_MAX` | Render only | Set `2` to save ~30–50 MB RAM |

---

## Cloud Deployment (Render.com)

Add these in **Render Dashboard → Environment**:

| Key | Value |
|---|---|
| `LLM_PROVIDER` | `gemini` |
| `GOOGLE_API_KEY` | your Gemini key |
| `GEMINI_MODEL` | `gemini-3.6-flash` |
| `MONGO_URI` | your Atlas connection string |
| `OPENROUTER_API_KEY` | your OpenRouter key |
| `OPENROUTER_MODEL` | `stealth/ox-alpha` |
| `RAPIDAPI_KEY` | your RapidAPI key |
| `MALLOC_ARENA_MAX` | `2` |

> **Note:** `DATA_COLLECTIONS_DIR` is a local Windows path — do NOT add it to Render. ChromaDB data must be committed to the repo or stored in a persistent disk.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | LLM provider/model info, collection stats |
| `GET` | `/api/health` | Per-service status: ChromaDB · Gemini · OpenRouter · RapidAPI · RAG Chain · MongoDB |
| `POST` | `/ask` | Standard RAG query |
| `POST` | `/ask/smart` | SSE streaming (intent-aware, recommended) |
| `POST` | `/ask/upload` | Multi-modal (image/PDF) |
| `GET` | `/trains/{train_no}` | Train details |
| `GET` | `/stations/{code}` | Station details |
| `GET` | `/admin/stats` | Query analytics from MongoDB |
| `GET` | `/feedback/summary` | Feedback stats from MongoDB |
| `POST` | `/feedback` | Submit thumbs-up/down + comment |
| `POST` | `/clear-cache` | Clear response cache + session history |

Full interactive docs at: **http://localhost:8000/docs**

---

## Architecture

```
User Query
    │
    ▼
Intent Classifier (11 categories)
    │
    ▼
Smart Context Builder ──► ChromaDB (5 collections, 35k docs)
    │                            ↕ gemini-embedding-001 (3072d)
    ▼
LLM (Gemini / OpenRouter / LM Studio)
    │
    ▼
SSE Stream ──► Frontend (index.html)
    │
    ├──► MongoDB Atlas
    │      ├── query_logs
    │      └── feedback
    │
    └──► Admin Dashboard (admin.html)
```

---

## License

Educational and portfolio purposes.

---

> Built with FastAPI · LangChain · ChromaDB · MongoDB Atlas · Google Gemini · OpenRouter · RapidAPI
> Live at: **https://railway-rag.onrender.com**
