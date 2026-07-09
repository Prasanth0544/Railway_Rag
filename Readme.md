# 🚂 Railway RAG Assistant

> **AI-Powered Indian Railway Information Retrieval System**
> Hybrid RAG with FastAPI, LangChain, ChromaDB & Google Gemini

---

## 📋 About

A **Hybrid Retrieval-Augmented Generation (RAG)** system that answers natural-language questions about Indian Railways — trains, stations, routes, schedules, rules, and regulations — using multi-strategy retrieval (vector search + keyword search + metadata lookup) and context-grounded answer generation via Google Gemini.

**No model training or fine-tuning required.** Intelligence comes from retrieval-augmented generation: 33,200+ railway documents are embedded into ChromaDB, searched at query time using a hybrid pipeline, and fed to Gemini for accurate, human-readable answers.

---

## 🏗️ Architecture

```
                         User Question
                              │
                              ▼
                   POST /ask/stream (FastAPI)
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
            Query Rewriting        Train Number
          (Fuzzy Station Resolver)   Detection
                    │                    │
                    ▼                    ▼
             Intent-Based Routing    Exact Metadata
          (transit vs rules intent)    Lookup
                    │                    │
         ┌──────────┼──────────┐        │
         ▼          ▼          ▼        │
    Keyword     Vector     Metadata     │
    Search      Search     Filter       │
   ($contains)  (cosine)  (train_no)    │
         └──────────┼──────────┘        │
                    ▼                   │
              Merge & Deduplicate ◄─────┘
                    │
                    ▼
         Route Schedule Trimming
          (context optimization)
                    │
                    ▼
          Prompt Template + Gemini 2.5 Flash
                    │
                    ▼
           Streamed JSON Answer (SSE)
```

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **LLM** | Google Gemini 2.5 Flash (cloud) / LM Studio (local, offline) |
| **RAG Framework** | LangChain (LCEL) |
| **Vector Database** | ChromaDB (persistent, 5 collections) |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 (offline) / Gemini Embeddings (cloud) |
| **Frontend** | HTML5, CSS3, JavaScript (no framework) |
| **Data** | 12,813 trains, 9,956 stations, 10,158 routes, 183 rules, 90 references |

---

## 🧠 Key Features

### Hybrid Retrieval Pipeline
- **Fuzzy Station Resolver**: Handles typos and phonetic variations (e.g. "Santhamagulur" → "Santamagulur (SAB)") using `difflib` with 0.8 cutoff
- **Train Number Detection**: Extracts 5-digit train numbers from queries and does direct metadata lookups with 1.0 relevance
- **Intent-Based Routing**: Classifies queries as transit or rules intent, routing to only relevant collections
- **Keyword Substring Search**: Uses ChromaDB `$contains` for exact station name matching in route documents
- **Vector Semantic Search**: Cosine similarity search for meaning-based retrieval
- **Dynamic Route Trimming**: Condenses 70-stop route schedules to origin → target → destination, reducing context by ~90%
- **Running Frequency Enrichment**: Route documents include "Runs on: Daily/Mon,Thu,Sat" from train_info cross-reference

### Web Dashboard
- **SSE Token Streaming**: Real-time word-by-word answer generation
- **Source Checklist**: Retrieved documents with type badges (Train, Route, Station, Rule) and similarity scores
- **RAG Statistics Strip**: Response time, average score, LLM engine, embedding model
- **System Info Sidebar**: Live collection stats, pipeline flow visualization
- **Dark/Light Mode**: Toggle theme with persistence

---

## 📁 Project Structure

```
Railway RAG Assistant/
├── app/
│   ├── main.py                 # FastAPI application & endpoints (REST + SSE streaming)
│   ├── rag.py                  # RAG chain with LLM provider switching (Gemini / LM Studio)
│   └── retriever.py            # Hybrid retriever (vector + keyword + metadata + fuzzy)
├── scripts/
│   ├── preprocess.py           # CSV → LangChain Documents (station linking, route enrichment)
│   └── create_embeddings.py    # Documents → ChromaDB vectors (batched, with retry)
├── web/
│   ├── index.html              # Dashboard UI with sidebar, pipeline flow, source checklist
│   ├── styles.css              # Design system (dark mode, glassmorphism, animations)
│   └── app.js                  # SSE stream reader, markdown renderer, health check
├── data/
│   └── railway_rules.csv       # 183 curated railway rules & regulations
├── chroma_db/                  # Persistent vector store (auto-generated, git-ignored)
├── .env                        # API keys & configuration (git-ignored)
├── .gitignore
├── requirements.txt
├── COMMANDS.md                 # Complete command reference (setup, test, embed, run)
├── project_summary.md          # Detailed project summary
└── Readme.md                   # This file
```

---

## 🚀 Setup & Installation

### 1. Clone & Navigate
```bash
git clone https://github.com/Prasanth0544/Railway_Rag.git
cd Railway_Rag
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com), then update `.env`:
```env
GOOGLE_API_KEY=your-actual-api-key-here
LLM_PROVIDER=gemini                    # or "lmstudio" for offline
USE_LOCAL_EMBEDDINGS=true              # offline embeddings (recommended)
DATA_COLLECTIONS_DIR=path/to/your/csv  # where train_info.csv, station_info.csv live
```

### 5. Create Embeddings (one-time, ~5 minutes)
```powershell
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"
.venv\Scripts\python scripts/create_embeddings.py
```

### 6. Start the Backend API Server
```powershell
$env:HF_HUB_OFFLINE="1"; $env:TRANSFORMERS_OFFLINE="1"
.venv\Scripts\python -m uvicorn app.main:app --reload
```
API running at **http://localhost:8000** | Swagger docs at **http://localhost:8000/docs**

### 7. Start the Web UI
```bash
python -m http.server 3000 --directory web
```
Open **http://localhost:3000** in your browser.

---

## 📡 API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/ask` | Standard RAG query endpoint |
| `POST` | `/ask/stream` | Real-time SSE streaming endpoint |
| `GET` | `/health` | System health, LLM info, collection stats |
| `GET` | `/trains` | List trains (paginated) |
| `GET` | `/stations` | List stations (paginated) |
| `GET` | `/rules` | List all railway rules |
| `GET` | `/trains/{train_no}` | Get specific train details |
| `GET` | `/stations/{station_code}` | Get specific station details |

---

## 💬 Example Queries

```bash
# Train route with full schedule
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the route of train 12727?"}'

# Station stop query (hybrid search)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What trains stop daily at Vinukonda?"}'

# Fuzzy station name (typo handling)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What trains stop at Santhamagulur?"}'

# Rules query (intent routing)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the cancellation charges for AC tickets?"}'
```

---

## 📊 Dataset Coverage

| Collection | Documents | Source |
|-----------|----------|-------|
| **Trains** | 12,813 | train_info.csv (numbers, names, types, schedules, zones) |
| **Stations** | 9,956 | station_info.csv + station_zones.csv + station_aka_info.csv (linked) |
| **Train Routes** | 10,158 | train_route_decoded.csv (stop-level schedules with running frequency) |
| **Railway Rules** | 183 | railway_rules.csv (booking, cancellation, luggage, penalties, concessions) |
| **References** | 90 | ticket_classes.csv + service_tax.csv |
| **Total** | **33,200** | |

---

## ⚙️ Configuration Modes

| Mode | LLM | Embeddings | Internet? |
|------|-----|-----------|-----------|
| **Cloud** | Google Gemini 2.5 Flash | sentence-transformers (offline) | Yes (for LLM only) |
| **Fully Offline** | LM Studio (Gemma 2 9B) | sentence-transformers (offline) | No |

---

## 🛠️ Development

### Resume Description
> Built a Hybrid RAG system using LangChain, ChromaDB, FastAPI, and Google Gemini that answers Indian Railways queries across 33,200+ documents. Implemented multi-strategy retrieval (vector + keyword + metadata), fuzzy station name resolution, intent-based collection routing, and real-time SSE streaming — achieving accurate retrieval across 12k trains, 10k stations, and 10k routes without model fine-tuning.

---

## 📄 License

This project is for educational and portfolio purposes.