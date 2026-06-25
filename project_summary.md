# 🚂 Railway RAG Assistant — Project Summary

This document provides a comprehensive summary of the **Railway RAG Assistant** project, detailing the architecture, the dataset ingestion pipelines, the RAG orchestration, API endpoints, current development status, and next steps.

---

## 🧭 Project Goal

The **Railway RAG Assistant** is a Retrieval-Augmented Generation (RAG) system built to answer natural-language queries about Indian Railways (trains, stations, schedules, routes, rules, and general information) using a semantic vector search backed by **Google Gemini** (or local offline LLMs).

Rather than fine-tuning a model, this RAG approach extracts relevant context from a persistent **ChromaDB** vector store and feeds it to the LLM to compile accurate, hallucination-free, and structured responses.

---

## 🏗️ Architecture & Query Flow

```
                     🧑 User Question
                            │
                            ▼
              ⚡ FastAPI Server (POST /ask)
                            │
                            ▼
           🔍 Unified Retriever (retriever.py)
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
      [railway_rules]    [trains]      [stations]   ... (5 collections)
            └───────────────┬───────────────┘
                            │ (Merge & Rank by Score)
                            ▼
                Top-K Context Documents
                            │
                            ▼
           📝 Prompt Template (rag.py Prompt)
                            │
                            ▼
         🤖 Google Gemini API (or Local LM Studio)
                            │
                            ▼
                 ✅ JSON-Structured Response
```

---

## 📊 Dataset Ingestion & Preprocessing

The system processes large datasets exported from MongoDB (erail APK format) along with reference railway rules.

### Data Ingestion Statistics:
* **Railway Rules (`data/railway_rules.csv`)**: **183** rule documents covering booking rules (Tatkal, quotas), cancellation charges, luggage allowances, penalties, concessions, and department responsibilities.
* **Train Information (`train_info.csv`)**: **12,813** train documents processed containing train numbers, names, types, sources/destinations, duration, operating days, and stops.
* **Station Information (`station_info.csv`)**: **11,354** station documents, enriched with regional railway zones by merging `station_zones.csv`, and including coordinates, cities, and WiFi availability.
* **Train Routes (`train_route_decoded.csv`)**: Routes parsed from nested JSON arrays detailing station stop sequences, arrival/departure schedules, and cumulative distance.
* **Reference Data**: Ticket classes (`ticket_classes.csv`) and service taxes (`service_tax.csv`).

---

## 📁 Project Structure & Codebase Components

```
railway-rag-assistant/
├── .env                        # Environment configurations (API keys & provider settings)
├── .gitignore                  # Git ignore file (excludes virtual environment and database)
├── requirements.txt            # Python dependencies (FastAPI, LangChain, ChromaDB, Pandas)
├── Readme.md                   # Setup instructions and user guide
├── project_summary.md          # [THIS FILE] Project summary
├── data/
│   └── railway_rules.csv       # Hand-curated CSV of 183 railway rules
├── scripts/
│   ├── preprocess.py           # Ingests, parses, and converts CSVs to LangChain Documents
│   └── create_embeddings.py    # Generates Gemini embeddings and populates ChromaDB
├── app/
│   ├── main.py                 # FastAPI application and endpoint definitions
│   ├── retriever.py            # Unified retriever querying across all collections
│   └── rag.py                  # LangChain RAG pipeline supporting cloud/local LLM execution
└── web/
    ├── index.html              # Main HTML markup with System Info & RAG Pipeline Flow
    ├── styles.css              # Custom styling with dark mode, animations, and layouts
    └── app.js                  # Frontend logic handling Fetch-based SSE streaming and rendering
```

### Script & Code Walkthrough:

1. **`scripts/preprocess.py`**
   * Dynamically loads data from `.csv` files stored in a configurable directory (`DATA_COLLECTIONS_DIR`).
   * Combines datasets, merges station zone details, and decodes JSON-based routes.
   * Standardizes text inputs into readable natural-language templates optimized for embedding search.

2. **`scripts/create_embeddings.py`**
   * Configures embeddings via **Google Generative AI** (`embedding-001`) or locally (`sentence-transformers/all-MiniLM-L6-v2`).
   * Iterates through documents in batches (respecting Gemini API rate limits) and saves vectors to a local persistent directory (`chroma_db/`).
   * Supports command-line switches like `--skip-routes`, `--rules-only`, or `--trains-only` to facilitate modular/fast rebuilding.

3. **`app/retriever.py`**
   * Connects to local persistent **ChromaDB**.
   * Implements a **Unified Retriever** which runs concurrent queries against active vector database collections (`railway_rules`, `trains`, `stations`, `train_routes`, `references`), merging search results and sorting them globally by relevance score.
   * Can switch to local offline embeddings (`sentence-transformers/all-MiniLM-L6-v2`) if offline mode is activated.

4. **`app/rag.py`**
   * Orchestrates the final Prompt-to-LLM pipeline.
   * Connects to either **Google Gemini API** (cloud-based `gemini-1.5-flash`) or a local **LM Studio** instance (compatible with offline testing on Ryzens/RTX GPUs).
   * Enforces rules via system prompting (no hallucinations, strict formatting, citing source train/station details).

5. **`app/main.py`**
   * Implements a FastAPI application serving:
     * `POST /ask` — REST query path executing the RAG pipeline.
     * `POST /ask/stream` — Real-time streaming RAG endpoint transmitting answer tokens via Server-Sent Events (SSE).
     * `GET /trains` & `GET /stations` — Paginated details directly from source datasets.
     * `GET /rules` — Lists ingested railway rules.
     * `GET /trains/{train_no}` & `GET /stations/{station_code}` — Specific key-based lookups.
     * `GET /health` — Verifies API running state, LLM details, and collection stats.

6. **`web/` Front-End Web Client**
   * **`index.html`**: A responsive, two-panel dashboard built on semantic HTML5. Includes an input text area, template question chips, sidebar system status monitor, and visual RAG flowchart.
   * **`styles.css`**: Design tokens for custom themes (dark mode default, light mode togglable), animations (pulsing pipeline flows, loading typing bubbles), and clean key-value listings.
   * **`app.js`**: Handles:
     * System health check query on initialization to fill sidebar stats.
     * Stream reader reading from `/ask/stream`.
     * Real-time rendering of tokens, retrieved sources checklists with score filters, and RAG execution statistics.

---

## ⚙️ Configuration & Modes

The project is highly flexible, supporting two primary execution strategies configured via `.env`:

| Mode | LLM Provider | Embedding Provider | Internet Required? | Setup |
|---|---|---|---|---|
| **Cloud (Default)** | Google Gemini (`gemini-1.5-flash`) | Google Gemini (`embedding-001`) | **Yes** | Requires `GOOGLE_API_KEY` in `.env` |
| **Offline (Local)** | Local server (LM Studio e.g. Gemma 2 9B) | HuggingFace sentence-transformers | **No** | Requires `USE_LOCAL_EMBEDDINGS=true` and `LLM_PROVIDER=lmstudio` |

---

## 🚦 Current Status & Verification Checklist

- [x] **Project Scaffolding**: Setup `requirements.txt`, `.gitignore`, and `.env` template.
- [x] **Data Pipelines**: Finished robust data cleaning scripts in `preprocess.py` targeting the 24k+ total source documents.
- [x] **Core Backend & Chain**: Fully coded retriever system (`retriever.py`), RAG generation loop (`rag.py`), and FastAPI routes (`main.py`).
- [x] **Local Test Validation**: Syntactical analysis verified for all python modules.
- [x] **Dependency Installation**: Completed successfully.
- [x] **Vector Database Creation**: Vector embeddings generated for rules, trains, stations, and routes in local ChromaDB.
- [x] **FastAPI Server Launch**: Back-end running on `http://localhost:8000` supporting Server-Sent Events (SSE).
- [x] **Responsive Web Dashboard**: Frontend UI integrated with streaming answers, dynamic system settings card, checklist sources, and visual pipeline flow.

