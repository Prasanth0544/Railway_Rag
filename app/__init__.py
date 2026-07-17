"""
Railway RAG Assistant — Application Package

Core modules:
  - main.py       FastAPI application with all endpoints
  - rag.py        RAG chain (retrieval + LLM generation)
  - retriever.py  Unified multi-strategy ChromaDB retriever
  - intent.py     Intent classification & query rewriting
  - ntes_client.py  Live train status (NTES scraper)
  - pnr_client.py   PNR status (ConfirmTkt / RailYatri)
"""
