# 🚂 Railway RAG Assistant

> **AI-Powered Indian Railways Information System**
> Hybrid RAG · FastAPI · LangChain · ChromaDB · Google Gemini · Live APIs

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-orange)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-purple)](https://www.trychroma.com)
[![Gemini](https://img.shields.io/badge/Gemini-3.1_Flash_Lite-blue?logo=google)](https://aistudio.google.com)
[![Render](https://img.shields.io/badge/Deployed-Render.com-46E3B7?logo=render)](https://railway-rag.onrender.com)

---

## 📋 What is This?

A **production-ready Hybrid RAG (Retrieval-Augmented Generation)** assistant for Indian Railways. Ask any question in plain English — train schedules, live running status, PNR status, cancellation rules, luggage policies, station info — and get grounded, accurate answers.

**No model training required.** Intelligence comes from:
- **35,383+ indexed railway documents** stored in ChromaDB (5 collections)
- **Multi-strategy hybrid retrieval** (vector + keyword + metadata)
- **Live APIs** for real-time train status and PNR
- **Gemini 3.1 Flash-lite** for natural language generation
- **3072-dimensional Gemini embeddings** (`gemini-embedding-001`)

---

## 🏗️ Architecture

```
User Question
      │
      ▼
 Intent Classifier (STATIC / LIVE / HYBRID / PNR)
      │
      ├─── STATIC ──► ChromaDB Hybrid Retriever
      │                 ├── Vector Search (semantic, 3072-dim Gemini embeddings)
      │                 ├── Keyword Search ($contains)
      │                 └── Metadata Lookup (train_no, station_code)
      │
      ├─── LIVE ────► Live Train Status (tiered by environment)
      │                 ├── Local:  NTES Direct (enquiry.indianrail.gov.in)
      │                 ├── Cloud:  RapidAPI (irctc-indian-railway-pnr-status)
      │                 └── Fallback: erail.in (schedule data)
      │
      ├─── PNR ─────► PNR Status API
      │
      └─── HYBRID ──► Both ChromaDB + Live APIs combined
                           │
                           ▼
                    Context → Gemini 3.1 Flash-lite
                           │
                           ▼
                   SSE Streaming Response
                   + Sources Panel (chips)
```

---

## ✨ Key Features

### 🧠 Backend Intelligence
| Feature | Description |
|---|---|
| **Hybrid Retrieval** | Vector (semantic) + Keyword ($contains) + Metadata (exact train/station match) combined |
| **Intent Classifier** | Keyword + regex rules classify STATIC / LIVE / HYBRID / PNR with confidence scores |
| **Fuzzy Station Resolver** | Handles typos & phonetic variants (e.g. "Santhamagulur" → correct station) using difflib |
| **Route Trimming** | Condenses 70-stop schedules to origin→target→destination (reduces tokens by ~90%) |
| **Multi-turn Memory** | Keeps last 5 Q&A pairs per session for contextual follow-up questions |
| **PNR Support** | Detects 10-digit PNR in query and fetches live booking + passenger status |
| **Multi-modal Uploads** | Upload ticket images/PDFs — Gemini Vision extracts and answers questions |
| **LLM Flexibility** | Gemini 3.1 Flash-lite (cloud) **or** LM Studio local model (fully offline) |

### 🚦 Live Train Data (Smart Provider Selection)
| Environment | Primary Source | Fallback |
|---|---|---|
| **Local (your PC)** | NTES Direct (indianrail.gov.in) — no API key needed | erail.in |
| **Cloud (Render)** | RapidAPI Indian Railways — RAPIDAPI_KEY required | erail.in |

> **Why two sources?** NTES blocks cloud server IPs (Render/Singapore IP ranges are flagged). RapidAPI provides reliable live data from cloud. Locally, NTES works perfectly on residential IPs.

Live output example:
```
=== LIVE TRAIN STATUS (Source: RapidAPI | Fetched: 2026-07-26T23:31 IST) ===
Train: 12622 - Tamil Nadu Express
Current Location: Kosi Kalan
Journey Progress: 12% of route completed
Position Detail: Departed from Kosi Kalan at 22:24 26-Jul
Delay: 6 MINUTES LATE

Recently Passed Stations (last 3):
  [PASSED] New Delhi | Departed: 21:05 | On Time
  [PASSED] Kosi Kalan | Departed: 22:24 | 6 min late

Upcoming Stations (next 5):
  [NEXT] Agra Cantt | Scheduled Arrival: 23:25 | Expected delay: 6 min late
  [NEXT] Gwalior Jn | Scheduled Arrival: 01:13 | Expected delay: 6 min late
```

### 🎨 Frontend
| Feature | Description |
|---|---|
| **SSE Streaming** | Real-time word-by-word answer rendering via Server-Sent Events |
| **Right-side Source Panel** | Grouped, clickable source chips (Trains, Routes, Rules, Stations) with relevance scores |
| **File Upload** | Drag-and-drop or attach image/PDF for multi-modal Q&A |
| **Voice Input** | Mic button for speech-to-text queries |
| **Dark/Light Theme** | Toggle with localStorage persistence |
| **RAG Pipeline Sidebar** | Live stats — total docs, LLM model, collection sizes, pipeline flow visualization |
| **Example Chips** | Quick-access query suggestions in a scrollable single-line row |
| **Follow-up Chips** | AI-suggested follow-up questions after each answer |

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10, FastAPI, Uvicorn |
| **RAG Framework** | LangChain (LCEL) |
| **LLM** | Google Gemini 3.1 Flash-lite / LM Studio (local) |
| **Vector DB** | ChromaDB (persistent, 5 collections) |
| **Embeddings** | `gemini-embedding-001` (3072-dim, cloud) |
| **Live Data (Local)** | NTES API — enquiry.indianrail.gov.in |
| **Live Data (Cloud)** | RapidAPI — irctc-indian-railway-pnr-status.p.rapidapi.com |
| **PNR Data** | erail.in + RailYatri scrapers |
| **Frontend** | Vanilla HTML5, CSS3, JavaScript — no framework |
| **Deployment** | Render.com (render.yaml configured) |

---

## 📁 Project Structure

```
Railway RAG Assistant/
├── app/
│   ├── __init__.py
│   ├── config.py            # Pydantic settings — env vars with validation
│   ├── intent.py            # Intent classifier (STATIC/LIVE/HYBRID/PNR)
│   ├── logger.py            # Structured logging setup
│   ├── main.py              # FastAPI app — REST + SSE streaming endpoints
│   ├── ntes_client.py       # Live train status — NTES (local) + RapidAPI (cloud) + erail.in
│   ├── pnr_client.py        # PNR API client for live booking status
│   ├── rag.py               # RAG chain — system prompt + LLM switching
│   └── retriever.py         # Hybrid retriever — vector + keyword + metadata + fuzzy
├── scripts/
│   ├── create_embeddings.py  # Build ChromaDB — all collections (rules/trains/stations)
│   ├── embed_routes.py       # Dedicated: train_routes collection (compact stop_codes)
│   ├── embed_local.py        # Future: Mumbai Metro/Local schedules (~26k docs)
│   ├── preprocess.py         # CSV → LangChain Documents with station linking
│   └── test_keys.py          # Gemini API key quota diagnostic tool
├── web/
│   ├── index.html            # Main UI with sidebar, chips, chat area
│   ├── styles.css            # Design system (dark mode, glassmorphism, grid)
│   ├── app.js                # SSE reader, source chips, markdown renderer
│   └── assets/
│       ├── marked.min.js     # Markdown renderer (local, no CDN)
│       └── railway-network.svg
├── data/
│   └── railway_rules.csv     # 183 curated railway rules and regulations
├── render.yaml               # Render.com deployment config (auto-deploy)
├── build.sh                  # Build script for Render
├── .env.example              # Template for environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Knowledge Base (v1 — Current)

> 5 ChromaDB collections · **35,383 documents** · 3072-dim Gemini embeddings

| Collection | Documents | Source | Content |
|---|---|---|---|
| **trains** | 12,813 | `train_info.csv` | Train numbers, names, types, zones, schedules, duration |
| **stations** | 9,956 | `station_info.csv` + zones + AKA | Station codes, names, alternate names, WiFi, zones |
| **train_routes** | 12,341 | `train_routes.csv` (stop_codes) | Which trains stop at which stations — compact route docs |
| **railway_rules** | 183 | `railway_rules.csv` | Booking, cancellation, luggage, penalties, concessions |
| **references** | 90 | `ticket_classes.csv` + `service_tax.csv` | Ticket classes, service tax tables |
| **Total** | **35,383** | | |

### 🔮 v2 Roadmap (Future Expansion — +270k docs)

| Collection | Est. Docs | Content |
|---|---|---|
| `train_schedules` | ~232,489 | Per-stop arrival/departure/platform/halt for every train |
| `coach_positions` | ~12,444 | Coach layout & reversal stations per train |
| `local_schedules` | ~25,826 | Mumbai Metro/Local trip schedules |
| `platform_info` | ~114 | Platform directions at stations |
| **v2 Total** | **~306,256** | |

> v2 embeddings are scripted and ready — will be enabled after v1 deployment is validated.

---

## 🚀 Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Prasanth0544/Railway_Rag.git
cd Railway_Rag
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
```
Edit `.env` and add your values:
```env
GOOGLE_API_KEY=your-gemini-api-key       # Get free at aistudio.google.com
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-3.1-flash-lite
USE_LOCAL_EMBEDDINGS=false               # Uses Gemini cloud embeddings
DATA_COLLECTIONS_DIR=path/to/your/csv_files

# Optional — only needed for cloud deployments (NTES is blocked on cloud IPs)
# RAPIDAPI_KEY=your-rapidapi-key         # Get free at rapidapi.com
```

> **Multiple API Keys:** Add `GOOGLE_API_KEY_1`, `GOOGLE_API_KEY_2`, ... for automatic key rotation when daily quota (1000 req/day) is exhausted.

### 5. Build the Vector Database (one-time)

```powershell
# Step 1: Embed railway_rules, trains, stations, references
.venv\Scripts\python scripts/create_embeddings.py --skip-routes

# Step 2: Embed train_routes (dedicated script with key rotation)
.venv\Scripts\python scripts/embed_routes.py
```

> Embedding uses `gemini-embedding-001` (cloud). With 1 API key: ~4–5 hours for all 35k docs.
> With 13 keys rotating automatically: completes in one session.

### 6. Start the Server
```powershell
# Windows
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
```bash
# macOS / Linux
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 7. Open the Web UI
Navigate to: **http://127.0.0.1:8000/web/index.html**

API Docs available at: **http://127.0.0.1:8000/docs**

---

## ☁️ Cloud Deployment (Render.com)

This project includes a `render.yaml` — Render auto-detects and configures everything.

### Steps

1. Go to [render.com](https://render.com) → **New Web Service** → Connect GitHub → `Railway_Rag`
2. Render reads `render.yaml` automatically — no manual config needed
3. In **Environment** tab, set these secrets:

| Key | Value | Required |
|-----|-------|----------|
| `GOOGLE_API_KEY` | Your Gemini API key from [aistudio.google.com](https://aistudio.google.com) | ✅ Required |
| `RAPIDAPI_KEY` | Your key from [rapidapi.com](https://rapidapi.com) → subscribe to **IRCTC Indian Railway PNR Status** (free tier) | ⭐ For live train data |

4. Deploy → get `https://railway-rag.onrender.com`

### Why RapidAPI is needed on Render

Render's Singapore server IPs are blocked by NTES (enquiry.indianrail.gov.in). The code auto-detects cloud environment and routes live train queries to RapidAPI instead. Without `RAPIDAPI_KEY`, it falls back to erail.in (schedule data only, no live GPS position).

### render.yaml config summary
```yaml
startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 75
envVars:
  - GOOGLE_API_KEY       # Set manually in dashboard
  - RAPIDAPI_KEY         # Set manually in dashboard (optional but recommended)
  - PYTHONDONTWRITEBYTECODE: 1
  - MALLOC_ARENA_MAX: 2  # Reduces glibc memory arena overhead (~30-50MB saved)
  - HF_HUB_OFFLINE: 1   # Blocks HuggingFace downloads
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root redirect |
| `GET` | `/health` | System health, LLM info, collection stats |
| `POST` | `/ask` | Standard RAG query (non-streaming) |
| `POST` | `/ask/smart` | SSE streaming query |
| `POST` | `/ask/upload` | Multi-modal query with image/PDF |
| `GET` | `/trains` | List trains (paginated) |
| `GET` | `/stations` | List stations (paginated) |
| `GET` | `/rules` | List all railway rules |
| `GET` | `/trains/{train_no}` | Get specific train details |
| `GET` | `/stations/{station_code}` | Get specific station details |

---

## 💬 Example Queries

```bash
# Live running status
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "where is 12622 train now?"}'

# Trains between stations
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which trains run from LTT to NDLS?"}'

# PNR check
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Check PNR 8101234567"}'

# Railway rules
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the cancellation charges for Sleeper class?"}'

# Fuzzy station name (handles typos)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What trains stop at Santhamagulur?"}'
```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | Primary Gemini API key from [aistudio.google.com](https://aistudio.google.com) |
| `GOOGLE_API_KEY_1` … `_N` | — | Additional keys for embedding rotation (optional) |
| `RAPIDAPI_KEY` | — | RapidAPI key for live train data on cloud (subscribe to IRCTC Indian Railway PNR Status — free tier at [rapidapi.com](https://rapidapi.com)) |
| `LLM_PROVIDER` | `gemini` | `gemini` or `lmstudio` (offline) |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | Gemini model name |
| `USE_LOCAL_EMBEDDINGS` | `false` | `true` = offline sentence-transformers, `false` = Gemini cloud |
| `DATA_COLLECTIONS_DIR` | — | Path to CSV data files directory |
| `LOCAL_API_BASE` | `http://localhost:1234/v1` | LM Studio server URL |
| `HF_HUB_OFFLINE` | `0` | Set to `1` to block all HuggingFace network calls |
| `MALLOC_ARENA_MAX` | — | Set to `2` on Linux to reduce glibc memory usage (recommended for Render free tier) |

---

## 🔍 How Live Train Data Works

```
Query: "where is 12622 now?"
         │
         ▼
    Cloud detected? (PORT env var set by Render)
         │
    YES  │  NO (local dev)
         │       │
         ▼       ▼
    RapidAPI   NTES Direct
    HTTP 200   (Fast, no API key)
         │       │
         └───────┘
               │
         Parse response:
         - current_station
         - delay_minutes (from most recent departed stop)
         - progress_percent
         - Last 3 passed stations + next 5 upcoming
               │
               ▼
         Format for LLM → Gemini generates response
```

**Timeout strategy:**
- RapidAPI/erail.in: `(8s connect, 15s read)` — fails fast if slow
- NTES: `(5s connect, 10s read)` — tight because it either works instantly or is blocked

**5-minute cache:** Same train queried twice within 5 min returns cached data (saves API quota).

---

## 📄 License

This project is for **educational and portfolio purposes**.

---

> Built with ❤️ using FastAPI, LangChain, ChromaDB, Google Gemini, and RapidAPI Indian Railways.
> Live at: **https://railway-rag.onrender.com**