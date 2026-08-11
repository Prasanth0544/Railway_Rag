# 🚂 Railway RAG Assistant

> **AI-Powered Indian Railways Information System**
> Hybrid RAG · FastAPI · LangChain · ChromaDB · Google Gemini · Live APIs

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-orange)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-purple)](https://www.trychroma.com)
[![Gemini](https://img.shields.io/badge/Gemini-Flash-blue?logo=google)](https://aistudio.google.com)
[![Render](https://img.shields.io/badge/Deployed-Render.com-46E3B7?logo=render)](https://railway-rag.onrender.com)
[![Total LOC](https://img.shields.io/badge/Lines_of_Code-13.4k-blue?logo=python)](#-codebase-statistics-loc)

---

## 📋 What is This?

A **production-ready Hybrid RAG (Retrieval-Augmented Generation)** assistant for Indian Railways. Ask any question in plain English — train schedules, live running status, PNR status, cancellation rules, luggage policies, station info — and get grounded, accurate answers.

**No model training required.** Intelligence comes from:
- **35,383+ indexed railway documents** stored in ChromaDB (5 collections)
- **12,728 train route docs** with full per-stop arr/dep/halt schedule times
- **Multi-strategy hybrid retrieval** (vector + keyword + metadata + intent-aware routing)
- **Smart context builder** — query-type-aware LLM context (6 strategies)
- **Live APIs** for real-time train status and PNR
- **Gemini Flash** for natural language generation
- **3072-dimensional Gemini embeddings** (`gemini-embedding-001`)

---

## 🏗️ Architecture

```
User Question
      │
      ▼
 Intent Classifier v2 (11 fine-grained categories)
 STATIC / LIVE / HYBRID / PNR + category:
   BETWEEN_STATIONS | SCHEDULE_QUERY | CANCELLATION_RULES
   STATION_INFO | COACH_QUERY | LIVE_STATUS | PNR_STATUS ...
      │
      ├─── STATIC / HYBRID ──► Intent-Aware ChromaDB Retriever
      │                          Step 1: Exact train number lookup
      │                          Step 2: Station name resolution (11,354 AKAs)
      │                          Step 3: Collection routing via intent_category
      │                                  BETWEEN_STATIONS → [train_routes]
      │                                  CANCELLATION_RULES → [railway_rules, references]
      │                                  STATION_INFO → [stations]
      │                          Step 4: Keyword scan ($contains)
      │                          Step 5: Semantic vector search
      │                          Step 6: Reciprocal Rank Fusion (RRF) merge
      │                          Step 7: Direction validation (A→B not B→A)
      │                          Step 8: Route doc trimming (over 80% token reduction)
      │
      ├─── LIVE ────────────► Live Train Status (tiered by environment)
      │                          Local:  NTES Direct (enquiry.indianrail.gov.in)
      │                          Cloud:  RapidAPI (irctc-indian-railway-pnr-status)
      │                          Fallback: erail.in (schedule data)
      │
      ├─── PNR ─────────────► PNR Status API
      │
      └─── All paths ───────► Smart Context Builder
                                  train_number  → 15,000 char budget
                                  route_search  → 12,000 char budget
                                  rules_refund  → 15,000 char budget
                                  station_info  →  8,000 char budget
                                  class_amenity → 10,000 char budget
                                  default       → 18,000 char budget
                                       │
                                       ▼
                               Gemini Flash (LLM)
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
| **Hybrid Retrieval** | Vector (semantic) + Keyword ($contains) + Metadata (exact train/station match) fused via **Reciprocal Rank Fusion (RRF)** |
| **Intent Classifier v2** | 11 fine-grained categories (BETWEEN_STATIONS, SCHEDULE_QUERY, CANCELLATION_RULES, etc.) with confidence scores |
| **Intent-Aware Retriever** | `intent_category` from classifier directly sets ChromaDB collection routing — no redundant keyword re-detection |
| **Smart Context Builder** | 6 query-type strategies with per-type char budgets — sends only what Gemini needs, no wasted tokens |
| **Schedule Times in Routes** | 10,239 route docs have full `arr HH:MM dep HH:MM (Xmin)` per stop — LLM can answer "what time does 12727 reach BZA?" |
| **Fuzzy Station Resolver** | Handles typos & phonetic variants (e.g. "Santhamagulur" → correct station) using difflib — 11,354 station names/AKAs |
| **Route Trimming** | Condenses 70-stop schedules to origin→queried stops→destination (reduces tokens by over 80%) |
| **Direction Validation** | Validates A→B direction in train route docs — drops wrong-direction trains from results |
| **Multi-turn Memory** | Keeps last 5 Q&A pairs per session for contextual follow-up questions |
| **PNR Support** | Detects 10-digit PNR in query and fetches live booking + passenger status |
| **Multi-modal Uploads** | Upload ticket images/PDFs — Gemini Vision extracts and answers questions |
| **LLM Flexibility** | Gemini Flash (cloud) **or** LM Studio local model (fully offline) |
| **Rate Limiting** | 15 requests/min per client IP — protects Gemini API quota |
| **Response Caching** | 10-minute TTL cache for STATIC queries — reduces API calls for repeated questions |
| **Hallucination Detection** | Post-validation flags fabricated train numbers not present in retrieved context |
| **Analytics Logging** | Every query logged to JSONL with intent, response time, retrieval stats, and **context strategy used** |
| **Admin Stats** | `/admin/stats` endpoint — query counts, top questions, error rates, intent distribution, context strategy breakdown |

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
| **Typing Stage Indicator** | Animated labels: "🔍 Classifying → 📚 Searching → ✨ Generating" with fade transitions |
| **Live Train Visualization** | Visual station timeline with progress bar, passed/upcoming dots, delay badge, pulsing current station |
| **Auto-Suggestions** | 20 popular queries filtered as you type (2+ chars), keyboard navigation (↑↓Enter) |
| **Right-side Source Panel** | Grouped, clickable source chips (Trains, Routes, Rules, Stations) with relevance scores |
| **File Upload** | Drag-and-drop or attach image/PDF for multi-modal Q&A |
| **Voice Input** | Mic button for speech-to-text queries |
| **Dark/Light Theme** | Toggle with localStorage persistence |
| **RAG Pipeline Sidebar** | Live stats — total docs, LLM model, collection sizes, pipeline flow visualization |
| **Example Chips** | Quick-access query suggestions in a scrollable single-line row |
| **Follow-up Chips** | AI-suggested follow-up questions after each answer |

### 📊 Quality & Evaluation
| Metric | Score |
|---|:---:|
| **Intent Classification Accuracy** | 96.7% (29/30 test queries) |
| **Retrieval Recall (≥50% keyword hit)** | 95.5% (21/22 STATIC queries) |
| **Overall Keyword Hit Rate** | 83.9% |

Run the evaluation benchmark:
```bash
python tests/evaluate_rag.py --skip-hallucination -v
```

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10, FastAPI, Uvicorn |
| **RAG Framework** | LangChain (LCEL) |
| **LLM** | Google Gemini Flash / LM Studio (local) |
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
│   ├── intent.py            # Intent classifier v2 — 11 fine-grained categories
│   ├── logger.py            # Structured logging setup
│   ├── main.py              # FastAPI app — REST + SSE streaming endpoints
│   ├── ntes_client.py       # Live train status — NTES (local) + RapidAPI (cloud) + erail.in
│   ├── pnr_client.py        # PNR API client for live booking status
│   ├── rag.py               # RAG chain — smart context builder + LLM switching
│   └── retriever.py         # Hybrid retriever — vector + keyword + intent-aware routing
├── scripts/
│   ├── create_embeddings.py  # Build ChromaDB — all collections (rules/trains/stations)
│   ├── embed_routes.py       # Dedicated: train_routes with schedule times + key rotation
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
├── docs/
│   └── smart_context_builder.md  # Design doc for query-type-aware context strategy
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

## 📊 Knowledge Base

> 5 ChromaDB collections · **35,383 documents** · 3072-dim Gemini embeddings

| Collection | Documents | Source | Content |
|---|---|---|---|
| **trains** | 12,813 | `train_info.csv` | Train numbers, names, types, zones, schedules, duration |
| **stations** | 9,956 | `station_info.csv` + zones + AKA | Station codes, names, alternate names, WiFi, zones |
| **train_routes** | 12,728 | `train_route_decoded.csv` + `train_routes.csv` | Full per-stop arr/dep/halt times for 10,239 trains; stop-codes only for 2,489 |
| **railway_rules** | 183 | `railway_rules.csv` | Booking, cancellation, luggage, penalties, concessions |
| **references** | 90 | `ticket_classes.csv` + `service_tax.csv` | Ticket classes, service tax tables |
| **Total** | **35,770** | | |

### Route Document Format (train_routes)

Each route doc contains the full schedule with timing information:

```
Train 12727 — Hyderabad Godavari SF Express (Daily). From VSKP to HYB. 21 stops, 707.0 km.
VSKP dep 17:20 | DVD arr 17:45 dep 17:47 (2min) | BZA arr 23:15 dep 23:30 (15min) | SC arr 05:45 dep 05:50 (5min) | HYB arr 06:15 [last].
```

- **10,239 trains** have full `arr HH:MM dep HH:MM (Xmin halt)` per stop ✅
- **2,489 trains** have stop-code only format (fallback from `train_routes.csv`)
- Avg document length: **620 characters**

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
GEMINI_MODEL=gemini-2.5-flash
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

# Step 2: Embed train_routes with full schedule times (checkpoint-resumable)
.venv\Scripts\python scripts/embed_routes.py
```

> Embedding uses `gemini-embedding-001` (cloud). With 1 API key (~1000 req/day free quota): runs over multiple days using checkpoint resume. With multiple rotating keys: completes in one session.

### 6. Start the Server
```powershell
# Windows
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
```bash
# macOS / Linux
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
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
| `POST` | `/ask/smart` | SSE streaming query (intent-aware, smart context) |
| `POST` | `/ask/upload` | Multi-modal query with image/PDF |
| `GET` | `/trains` | List trains (paginated) |
| `GET` | `/stations` | List stations (paginated) |
| `GET` | `/rules` | List all railway rules |
| `GET` | `/trains/{train_no}` | Get specific train details |
| `GET` | `/stations/{station_code}` | Get specific station details |
| `GET` | `/admin/stats` | Query analytics — counts, intent distribution, context strategies |
| `POST` | `/clear-cache` | Clear response cache + session history (dev/debug) |

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

# Schedule with times (requires embedded route docs)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What time does 12727 reach Vijayawada?"}'

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
| `RAPIDAPI_KEY` | — | RapidAPI key for live train data on cloud |
| `LLM_PROVIDER` | `gemini` | `gemini` or `lmstudio` (offline) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name |
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

## 📊 Codebase Statistics (LOC)

This project adheres to industry-standard code measurement guidelines, separating Source Lines of Code (SLOC), comments, and structural whitespace across all layers.

| Layer | Files | Source Code (SLOC) | Comments | Blank | Total LOC |
|---|---|---|---|---|---|
| **Scripts & Pipelines** | 8 | 1,850 | 420 | 480 | **2,750** |
| **Backend (Core App)** | 10 | 2,900 | 680 | 640 | **4,220** |
| **Frontend (UI Web)** | 4 | 2,742 | 151 | 372 | **3,265** |
| **Documentation** | 3 | 680 | 0 | 210 | **890** |
| **Config & Infrastructure** | 9 | 191 | 35 | 45 | **271** |
| **Frontend (Vendor Assets)** | 1 | 58 | 9 | 2 | **69** |
| **TOTAL** | **35** | **8,421** | **1,295** | **1,749** | **11,465** |

### 🛠️ Running the LOC Counter

```bash
# Print formatted console summary table
python scripts/count_loc.py

# Export markdown format for documentation
python scripts/count_loc.py --format markdown

# Export JSON for CI/CD pipelines
python scripts/count_loc.py --format json
```

---

## 📄 License

This project is for **educational and portfolio purposes**.

---

> Built with ❤️ using FastAPI, LangChain, ChromaDB, Google Gemini, and RapidAPI Indian Railways.
> Live at: **https://railway-rag.onrender.com**