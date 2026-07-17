# 🚂 Railway RAG Assistant

> **AI-Powered Indian Railways Information System**
> Hybrid RAG · FastAPI · LangChain · ChromaDB · Google Gemini · Live APIs

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-orange)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-purple)](https://www.trychroma.com)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-blue?logo=google)](https://aistudio.google.com)

---

## 📋 What is This?

A **production-ready Hybrid RAG (Retrieval-Augmented Generation)** assistant for Indian Railways. Ask any question in plain English — train schedules, live running status, PNR status, cancellation rules, luggage policies, station info — and get grounded, accurate answers.

**No model training required.** Intelligence comes from:
- **33,200+ indexed railway documents** stored in ChromaDB
- **Multi-strategy hybrid retrieval** (vector + keyword + metadata)
- **Live APIs** for real-time train status and PNR
- **Gemini 2.5 Flash** for natural language generation

---

## 🏗️ Architecture

```
User Question
      │
      ▼
 Intent Classifier (STATIC / LIVE / HYBRID / PNR)
      │
      ├─── STATIC ──► ChromaDB Hybrid Retriever
      │                 ├── Vector Search (semantic)
      │                 ├── Keyword Search ($contains)
      │                 └── Metadata Lookup (train_no, station_code)
      │
      ├─── LIVE ────► NTES API (real-time running status)
      │
      ├─── PNR ─────► PNR Status API
      │
      └─── HYBRID ──► Both ChromaDB + NTES API combined
                           │
                           ▼
                    Context → Gemini 2.5 Flash
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
| **LLM Flexibility** | Gemini 2.5 Flash (cloud) **or** LM Studio local model (fully offline) |

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
| **Backend** | Python 3.11, FastAPI, Uvicorn |
| **RAG Framework** | LangChain (LCEL) |
| **LLM** | Google Gemini 2.5 Flash / LM Studio (local) |
| **Vector DB** | ChromaDB (persistent, 5 collections) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (offline) |
| **Live Data** | NTES API (train status) + PNR API |
| **Frontend** | Vanilla HTML5, CSS3, JavaScript — no framework |
| **Deployment** | Docker + docker-compose / Render.com / Railway.app |

---

## 📁 Project Structure

```
Railway RAG Assistant/
├── app/
│   ├── __init__.py
│   ├── config.py           # Pydantic settings — env vars with validation
│   ├── intent.py           # Intent classifier (STATIC/LIVE/HYBRID/PNR)
│   ├── logger.py           # Structured logging setup
│   ├── main.py             # FastAPI app — REST + SSE streaming endpoints
│   ├── ntes_client.py      # NTES API client for live train running status
│   ├── pnr_client.py       # PNR API client for live booking status
│   ├── rag.py              # RAG chain — system prompt + LLM switching
│   └── retriever.py        # Hybrid retriever — vector + keyword + metadata + fuzzy
├── scripts/
│   ├── create_embeddings.py # Build ChromaDB from CSV data (run once)
│   ├── preprocess.py        # CSV → LangChain Documents with station linking
│   └── test_bza_hyd.py      # Test script for BZA–HYD route retrieval
├── web/
│   ├── index.html           # Main UI with sidebar, chips, chat area
│   ├── styles.css           # Design system (dark mode, glassmorphism, grid)
│   ├── app.js               # SSE reader, source chips, markdown renderer
│   └── assets/
│       ├── marked.min.js    # Markdown renderer (local, no CDN)
│       └── railway-network.svg
├── data/
│   └── railway_rules.csv    # 183 curated railway rules and regulations
├── Dockerfile               # Container image for deployment
├── docker-compose.yml       # Compose config with volume mounts
├── .env.example             # Template for environment variables
├── .gitignore
├── requirements.txt
└── Readme.md
```

---

## 📊 Knowledge Base

| Collection | Documents | Content |
|---|---|---|
| **Trains** | 12,813 | Train numbers, names, types, zones, schedules |
| **Stations** | 9,956 | Station codes, names, AKA variants, zones |
| **Train Routes** | 10,158 | Stop-level schedules with running frequency |
| **Railway Rules** | 183 | Booking, cancellation, luggage, penalties, concessions |
| **References** | 90 | Ticket classes, service tax tables |
| **Total** | **33,200** | |

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
GOOGLE_API_KEY=your-gemini-api-key     # Get free at aistudio.google.com
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
USE_LOCAL_EMBEDDINGS=true
DATA_COLLECTIONS_DIR=path/to/your/csv_files
```

### 5. Build the Vector Database (one-time, ~5–10 minutes)
```powershell
# Windows
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"
.venv\Scripts\python scripts/create_embeddings.py
```
```bash
# macOS / Linux
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/create_embeddings.py
```

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

## 🐳 Docker (Recommended)

### Run locally with Docker
```bash
# 1. Build and start
docker compose up --build

# 2. Open browser
# http://localhost:8000/web/index.html
```

> **Note:** Add `GOOGLE_API_KEY=your_key` inside `docker-compose.yml` under `environment`, or create a `.env` file — Docker Compose will pick it up automatically.

---

## ☁️ Cloud Deployment

Your app serves **both backend API and frontend** from the same server — no separate hosting needed.

### Option A — Render.com (Free, Easiest)

1. Go to [render.com](https://render.com) → New Web Service → Connect GitHub → `Railway_Rag`
2. **Build Command:** `pip install -r requirements.txt`
3. **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. **Add Environment Variable:** `GOOGLE_API_KEY=your_key`
5. **Add Disk** → Mount at `/opt/render/project/src/chroma_db` (for ChromaDB persistence)
6. Deploy → get `https://your-app.onrender.com`
7. Run the ingestion once via Render Shell: `python scripts/create_embeddings.py`

### Option B — Railway.app (Docker auto-detected)

1. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
2. Railway auto-detects the `Dockerfile` and builds it
3. Add variable: `GOOGLE_API_KEY=your_key`
4. Add a Volume → mount to `/app/chroma_db`
5. Deploy → get `https://your-app.up.railway.app`

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root redirect |
| `GET` | `/health` | System health, LLM info, collection stats |
| `POST` | `/ask` | Standard RAG query (non-streaming) |
| `POST` | `/ask/stream` | SSE streaming query |
| `POST` | `/ask/upload` | Multi-modal query with image/PDF |
| `GET` | `/trains` | List trains (paginated) |
| `GET` | `/stations` | List stations (paginated) |
| `GET` | `/rules` | List all railway rules |
| `GET` | `/trains/{train_no}` | Get specific train details |
| `GET` | `/stations/{station_code}` | Get specific station details |

---

## 💬 Example Queries

```bash
# Train route
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the route of train 12727?"}'

# Live status
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the running status of train 12728?"}'

# PNR check
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Check PNR 8101234567"}'

# Cancellation rules
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the cancellation charges for Sleeper class?"}'

# Fuzzy station name
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What trains stop at Santhamagulur?"}'
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | Gemini API key from [aistudio.google.com](https://aistudio.google.com) |
| `LLM_PROVIDER` | `gemini` | `gemini` or `lmstudio` (offline) |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name |
| `USE_LOCAL_EMBEDDINGS` | `true` | Use offline sentence-transformers |
| `DATA_COLLECTIONS_DIR` | — | Path to CSV data files |
| `LOCAL_API_BASE` | `http://localhost:1234/v1` | LM Studio server URL |
| `HF_HUB_OFFLINE` | `1` | Prevent HuggingFace network calls |

---

## 📄 License

This project is for **educational and portfolio purposes**.

---

> Built with ❤️ using FastAPI, LangChain, ChromaDB, and Google Gemini.