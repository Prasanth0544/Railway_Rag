# 🚂 Railway RAG Assistant

> **AI-Powered Indian Railway Information Retrieval System**
> Using Retrieval-Augmented Generation with FastAPI, LangChain, ChromaDB & Gemini API

---

## 📋 About

A Retrieval-Augmented Generation (RAG) system that answers natural-language questions about Indian Railways — trains, stations, routes, schedules, rules, and regulations — by semantically searching a vector database and generating contextual answers via Google Gemini.

**No model training required.** Intelligence comes from retrieval-augmented generation: railway data is embedded into ChromaDB, semantically searched at query time, and fed to Gemini for polished, human-readable answers.

---

## 🏗️ Architecture

```
User Question
      ↓
POST /ask (FastAPI)
      ↓
Unified Retriever (retriever.py)
      ↓
ChromaDB (3 collections: trains, stations, railway_rules)
      ↓
Top-5 Relevant Documents + Question
      ↓
Prompt Template + Gemini 1.5 Flash (rag.py)
      ↓
Structured JSON Answer
```

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend** | Python, FastAPI |
| **LLM** | Google Gemini 1.5 Flash |
| **RAG Framework** | LangChain (LCEL) |
| **Vector Database** | ChromaDB (persistent) |
| **Embeddings** | Gemini Embeddings (`models/embedding-001`) |
| **Data** | Trains, Stations, Railway Rules (CSV) |

---

## 📁 Project Structure

```
railway-rag-assistant/
├── data/
│   ├── trains.csv              # 30 trains dataset
│   ├── stations.csv            # 48 stations dataset
│   └── railway_rules.csv       # 80+ railway rules & regulations
├── scripts/
│   ├── preprocess.py           # CSV → natural-language documents
│   └── create_embeddings.py    # Documents → ChromaDB vectors
├── chroma_db/                  # Persistent vector store (auto-generated)
├── app/
│   ├── main.py                 # FastAPI application & endpoints
│   ├── rag.py                  # RAG chain (LCEL pipeline)
│   └── retriever.py            # Unified ChromaDB retriever
├── .env                        # API keys (git-ignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Installation

### 1. Clone & Navigate
```bash
git clone <your-repo-url>
cd railway-rag-assistant
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com), then update `.env`:
```env
GOOGLE_API_KEY=your-actual-api-key-here
```

### 5. Create Embeddings
```bash
python scripts/create_embeddings.py
```
This will generate embeddings for all 3 datasets and store them in `chroma_db/`.

### 6. Start the Server
```bash
uvicorn app.main:app --reload
```
Server runs at **http://localhost:8000**

---

## 📡 API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/ask` | **Main RAG query endpoint** |
| `GET` | `/trains` | List all trains |
| `GET` | `/stations` | List all stations |
| `GET` | `/rules` | List all railway rules |
| `GET` | `/trains/{train_no}` | Get specific train details |
| `GET` | `/stations/{station_code}` | Get specific station details |

### Interactive Docs
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 💬 Example Queries

### Train Search
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which trains run between Vijayawada and Hyderabad?"}'
```

### Train Lookup
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about train 12727"}'
```

### Station Query
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which station comes after Eluru?"}'
```

### Rules Query
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the cancellation charges for AC tickets?"}'
```

### Response Format
```json
{
  "question": "Which trains stop at Rajahmundry?",
  "answer": "The following trains stop at Rajahmundry: ...",
  "sources": [
    {
      "type": "train",
      "relevance_score": 0.8542,
      "train_no": "12727",
      "train_name": "Godavari Express"
    }
  ],
  "num_documents_retrieved": 5
}
```

---

## 📊 Dataset Coverage

### Trains (30 entries)
- Major routes: Hyderabad ↔ Visakhapatnam, Hyderabad ↔ Chennai, Secunderabad ↔ New Delhi
- Types: Rajdhani, SuperFast, Express, Passenger
- Includes: Godavari Express, Telangana Express, Konark Express, Charminar Express, etc.

### Stations (48 entries)
- Covering AP, Telangana, Tamil Nadu, Kerala, Maharashtra, Delhi, Odisha, West Bengal
- Includes station codes, GPS coordinates, zones, and divisions

### Railway Rules (80+ entries)
- **Booking Rules**: ARP, Tatkal, quotas, ticket types
- **Cancellation & Refund**: Time-based charges, TDR, auto-cancellation
- **Travel Classes**: 1A, 2A, 3A, 3E, SL, CC, EC, 2S, GN
- **Luggage Rules**: Free allowance, size restrictions, prohibited items
- **Department Roles**: Station Master, TTE, RPF, Guard, Loco Pilot
- **Concessions**: Senior citizens, students, freedom fighters
- **Station Amenities**: Waiting room, retiring room, cloak room, catering
- **Penalties**: Ticketless travel, chain pulling, smoking

---

## 🛠️ Development

### Resume Description
> Developed a Retrieval-Augmented Generation (RAG) system using LangChain, ChromaDB, FastAPI, and Gemini API to answer railway-related queries from railway schedule, station, and rules datasets through semantic search and vector retrieval.

---

## 📄 License

This project is for educational and portfolio purposes.