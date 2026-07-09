# 🚂 Railway RAG Assistant — Project Summary

This document provides a comprehensive technical summary of the **Railway RAG Assistant** project, detailing the architecture, dataset ingestion pipelines, hybrid retrieval strategies, RAG orchestration, API endpoints, and the web dashboard.

---

## 🧭 Project Goal

The **Railway RAG Assistant** is a **Hybrid Retrieval-Augmented Generation (RAG)** system built to answer natural-language queries about Indian Railways (trains, stations, schedules, routes, rules, and general information) using multi-strategy retrieval backed by **Google Gemini** (or local offline LLMs via LM Studio).

Rather than fine-tuning a model, this RAG approach retrieves relevant context from a persistent **ChromaDB** vector store using a combination of vector search, keyword matching, and metadata lookups, then feeds it to the LLM to compile accurate, hallucination-free, and structured responses.

---

## 🏗️ Architecture & Query Flow

```
                         🧑 User Question
                              │
                              ▼
               ⚡ FastAPI Server (POST /ask/stream)
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
          Step 1: Train Number    Step 2: Fuzzy Station
          Detection (regex)       Resolution (difflib)
                    │                    │
                    ▼                    ▼
          Exact Metadata          Query Rewriting
          Lookup (score 1.0)      (append canonical name + code)
                    │                    │
                    └─────────┬──────────┘
                              ▼
                   Step 3: Intent-Based Routing
                (transit vs rules collection filter)
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
         Step 4:         Step 5:        Step 5:
         Keyword         Vector         (same)
         $contains       Similarity
         Search          Search
               └──────────────┬──────────────┘
                              ▼
                   Step 6: Merge & Deduplicate
                              │
                              ▼
                   Step 7: Route Schedule Trimming
                  (origin → target → destination only)
                              │
                              ▼
                 📝 Prompt Template (rag.py)
                              │
                              ▼
              🤖 Google Gemini 2.5 Flash (or LM Studio)
                              │
                              ▼
                   ✅ Streamed JSON Response (SSE)
```

---

## 📊 Dataset Ingestion & Preprocessing

The system processes large datasets exported from MongoDB (erail APK format) along with curated railway rules. A key innovation is the **cross-dataset station lookup table** that links all station datasets by `station_code` before embedding, and the **frequency enrichment** that cross-references running days from train_info into route documents.

### Station Linking Pipeline

Before any embedding, `preprocess.py` calls `build_station_lookup()` which merges three datasets:

| Dataset | Records | Role |
|---|---|---|
| `station_info.csv` | 9,956 | Canonical names, GPS coords, WiFi |
| `station_zones.csv` | 10,760 | Railway zone per station (NR, SCR, SR, etc.) |
| `station_aka_info.csv` | 775 | Alternate names (e.g. "Bezawada" → BZA, "Madras Central" → MAS) |

The resulting lookup is **shared as a singleton** by both the station document builder and the route document builder.

### Frequency Enrichment Pipeline

Route documents are enriched with running frequency data:
- `preprocess.py` builds a `frequency_lookup` dictionary from `train_info.csv` mapping each train number to its `running_days_text`
- Each route document's text now includes `"Runs on: Daily"` or `"Runs on: Mon, Thu, Sat"`, and the metadata includes a `runs_on` field
- This enables the LLM to correctly distinguish daily vs weekly trains when answering station stop queries

### Data Ingestion Statistics

| Collection | Documents | Source | Content |
|---|---|---|---|
| Railway Rules | 183 | `data/railway_rules.csv` | Booking rules, cancellation charges, luggage, penalties, concessions, department roles |
| Train Information | 12,813 | `train_info.csv` | Train numbers, names, types, sources, destinations, duration, operating days, speed type, zone |
| Station Documents | 9,956 | Linked (info + zones + AKA) | Canonical name, city, zone, GPS, WiFi, all alternate names for fuzzy matching |
| Train Routes | 10,158 | `train_route_decoded.csv` | Full station name sequence + per-stop schedule (arrival, departure, halt, distance) + running frequency |
| References | 90 | `ticket_classes.csv` + `service_tax.csv` | Ticket class details and service tax info |
| **Total** | **33,200** | | |

---

## 🧠 Hybrid Retrieval Strategies

The `UnifiedRetriever` in `retriever.py` implements a 7-step retrieval pipeline that combines multiple strategies:

### Step 1: Train Number Detection
- Regex extracts 5-digit train numbers from the query (e.g. `"train 67239 stopping stations"` → `67239`)
- Direct metadata lookup in `trains` and `train_routes` collections (`where={"train_no": "67239"}`)
- Matched documents are assigned a relevance score of `1.0` (highest priority)

### Step 2: Fuzzy Station Name Resolution
- Uses `difflib.get_close_matches` (cutoff `0.8`) to match misspelled/phonetic station names
- Exact match with word boundaries is tried first (longest match wins)
- Stop words filter prevents common English words from matching short station codes (e.g. "the" ≠ station code `THE`)
- If matched, the search query is rewritten to append the canonical name and code

### Step 3: Intent-Based Collection Routing
- Transit keywords (`stop`, `route`, `timetable`, `departure`, etc.) → route to `trains`, `stations`, `train_routes`
- Rules keywords (`cancel`, `refund`, `luggage`, `penalty`, etc.) → route to `railway_rules`, `references`
- Reduces irrelevant noise by skipping unrelated collections entirely

### Step 4: Keyword Substring Search (Hybrid)
- When a station is resolved, uses ChromaDB's `$contains` operator to find all route documents containing the station name
- Assigns `0.95` relevance score to pin these results near the top
- Solves the "haystack dilution" problem where a station name is a tiny fraction of a 70-stop route document

### Step 5: Vector Semantic Search
- Standard cosine similarity search using `all-MiniLM-L6-v2` embeddings (384 dimensions)
- Retrieves `PER_COLLECTION_K=5` results per active collection
- Applies a `0.20` relevance score threshold to filter noise

### Step 6: Merge & Deduplicate
- Combines exact matches (Step 1), keyword matches (Step 4), and semantic matches (Step 5)
- Deduplicates by first 80 characters of content
- Priority order: exact (1.0) → keyword (0.95) → semantic (0.2–0.9)

### Step 7: Route Schedule Trimming
- For station-based queries, condenses long route schedules to only 3 stops: origin, target station, and terminal
- Reduces context size by ~90%, preventing LLM context window overflow
- Skipped for train number queries (user wants the full schedule)

### Dynamic Limit Expansion
- When a station or train number is detected, the retrieval limit expands from `top_k` (default 5) to `20`
- Ensures all matching trains at a station are returned (e.g. Vinukonda has 14 trains)

---

## 📁 Project Structure & Codebase Components

```
Railway RAG Assistant/
├── .env                        # Environment configurations (API keys & provider settings)
├── .gitignore                  # Git ignore file (excludes virtual environment and database)
├── requirements.txt            # Python dependencies (FastAPI, LangChain, ChromaDB, Pandas)
├── Readme.md                   # Setup instructions and user guide
├── project_summary.md          # [THIS FILE] Detailed project summary
├── COMMANDS.md                 # Complete command reference (750+ lines)
├── data/
│   └── railway_rules.csv       # Hand-curated CSV of 183 railway rules
├── scripts/
│   ├── preprocess.py           # Ingests CSVs, builds station lookup, enriches routes with frequency
│   └── create_embeddings.py    # Generates embeddings and populates ChromaDB (batched, with retry)
├── app/
│   ├── main.py                 # FastAPI application with REST + SSE streaming endpoints
│   ├── retriever.py            # Hybrid retriever (7-step pipeline: vector + keyword + metadata + fuzzy)
│   └── rag.py                  # LangChain RAG chain with Gemini/LM Studio provider switching
└── web/
    ├── index.html              # Dashboard with system info, pipeline flow, source checklist
    ├── styles.css              # Design system (dark mode, glassmorphism, micro-animations)
    └── app.js                  # SSE stream reader, markdown renderer, health polling
```

### Code Walkthrough

1. **`scripts/preprocess.py`**
   * Dynamically loads data from `.csv` files stored in a configurable directory (`DATA_COLLECTIONS_DIR`).
   * Builds a **cross-dataset station lookup** by merging `station_info`, `station_zones`, and `station_aka_info` by station code.
   * Builds a **frequency lookup** from `train_info.csv` to enrich route documents with running days.
   * Standardizes text inputs into readable natural-language templates optimized for embedding search.

2. **`scripts/create_embeddings.py`**
   * Configures embeddings via `sentence-transformers/all-MiniLM-L6-v2` (offline) or Google Gemini (cloud).
   * Iterates through documents in batches of 256 (with auto-retry for rate limits) and saves vectors to `chroma_db/`.
   * Supports CLI switches: `--skip-routes`, `--rules-only`, `--trains-only`, `--routes-only`.

3. **`app/retriever.py`**
   * Implements the **7-step hybrid retrieval pipeline** (train number detection → fuzzy resolution → intent routing → keyword search → vector search → merge → trimming).
   * Maintains a singleton station resolver with 20,395 name/AKA entries for fuzzy matching.
   * Dynamically expands retrieval limits when station or train number queries are detected.

4. **`app/rag.py`**
   * Orchestrates the Retrieve → Augment → Generate pipeline.
   * Connects to **Google Gemini 2.5 Flash** (cloud) or **LM Studio** (local, OpenAI-compatible).
   * Enforces strict system prompting: no hallucination, cite train numbers, include station codes, format clearly.

5. **`app/main.py`**
   * FastAPI application with CORS middleware and lifespan-managed RAG chain initialization.
   * `POST /ask` — Standard RAG query returning JSON with answer + sources.
   * `POST /ask/stream` — SSE streaming endpoint sending metadata, tokens, and done events.
   * Data endpoints: `GET /trains`, `GET /stations`, `GET /rules` with pagination.
   * `GET /health` — System health with LLM provider, model info, and per-collection document counts.

6. **`web/` Front-End Web Client**
   * **`index.html`**: Responsive two-panel dashboard with input area, template question chips, sidebar system monitor, and animated RAG pipeline flowchart.
   * **`styles.css`**: Custom design tokens, dark mode default (light mode togglable), glassmorphism cards, micro-animations, and Inter font from Google Fonts.
   * **`app.js`**: Health check on init, SSE stream reader for `/ask/stream`, real-time token rendering, source checklist with type badges and scores, RAG statistics strip.

---

## ⚙️ Configuration & Modes

| Mode | LLM Provider | Embedding Provider | Internet Required? | Setup |
|---|---|---|---|---|
| **Cloud (Default)** | Google Gemini 2.5 Flash | sentence-transformers (offline) | Yes (for LLM calls only) | `GOOGLE_API_KEY` in `.env` |
| **Fully Offline** | LM Studio (Gemma 2 9B) | sentence-transformers (offline) | No | `USE_LOCAL_EMBEDDINGS=true` + `LLM_PROVIDER=lmstudio` |

---

## 🚦 Current Status

- [x] **Project Scaffolding**: `.env`, `.gitignore`, `requirements.txt`, `COMMANDS.md`
- [x] **Data Pipelines**: Station linking, route enrichment with frequency, reference data loading
- [x] **Embedding Pipeline**: 33,200 documents embedded across 5 ChromaDB collections
- [x] **Hybrid Retriever**: 7-step pipeline (train number + fuzzy + intent + keyword + vector + dedup + trimming)
- [x] **RAG Chain**: Gemini/LM Studio dual-provider support with streaming
- [x] **FastAPI Backend**: REST + SSE streaming endpoints, data browsing, health checks
- [x] **Web Dashboard**: Dark mode, SSE streaming, source checklist, pipeline flow, stats strip
- [x] **Pushed to GitHub**: https://github.com/Prasanth0544/Railway_Rag
