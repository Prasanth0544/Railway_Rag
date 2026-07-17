"""
rag.py — RAG Chain with LLM Provider Switching

Supports:
  1. Google Gemini API (cloud — default for deployment)
  2. LM Studio local server (localhost:1234 — free, offline, for development)

Set LLM_PROVIDER in .env to switch:
  LLM_PROVIDER=gemini      → uses GOOGLE_API_KEY
  LLM_PROVIDER=lmstudio    → uses LOCAL_API_BASE (no key needed)
"""

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from app.retriever import get_unified_retriever


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Indian Railways assistant. Answer questions 
about Indian Railways using the context provided below.

RULES:
1. Answer based on the provided context. Extract as much useful information as possible.
2. If the context has PARTIAL information, use it and say what you found. Never say 
   "I don't have enough information" if ANY relevant data is present in the context.
3. Only say information is unavailable if the context is completely empty or says "No relevant documents found."
4. Format responses clearly with rich detail:
   - Use train numbers AND names when referring to trains (e.g. "12727 — Godavari SF Express")
   - Include departure/arrival times when available
   - Mention station codes alongside names (e.g. "Vijayawada (BZA)")
   - For routes, list ALL stations in order — do not abbreviate or cut off the list
   - For rules, cite the specific category and rule title
5. If multiple trains match a query between two stations, list ALL of them with their numbers, names, and timing.
6. Be thorough. If the user asks about stops or schedules, give the complete list from the context.
7. For simple yes/no or status queries, be concise (2-3 sentences). For schedule/route queries, be complete.

CONTEXT:
{context}
"""

HUMAN_PROMPT = "{question}"


# ─────────────────────────────────────────────
# LLM FACTORY — Gemini or LM Studio
# ─────────────────────────────────────────────

def get_llm():
    """
    Return the configured LLM.

    LLM_PROVIDER=gemini   → ChatGoogleGenerativeAI (gemini-1.5-flash)
    LLM_PROVIDER=lmstudio → ChatOpenAI pointing at http://localhost:1234
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()

    if provider == "lmstudio":
        from langchain_openai import ChatOpenAI
        base_url = os.getenv("LOCAL_API_BASE", "http://localhost:1234/v1")
        model    = os.getenv("LOCAL_MODEL_NAME", "local-model")

        print(f"🖥️  LLM: LM Studio @ {base_url} (model: {model})")
        return ChatOpenAI(
            base_url=base_url,
            api_key="lm-studio",         # LM Studio ignores this but OpenAI client needs it
            model=model,
            temperature=0.3,
            max_tokens=1024,
        )

    else:  # Default: Gemini
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key    = os.getenv("GOOGLE_API_KEY", "")
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        print(f"☁️  LLM: Google Gemini ({model_name})")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
            max_output_tokens=2048,
        )


# ─────────────────────────────────────────────
# DOCUMENT FORMATTER & SOURCE EXTRACTOR
# ─────────────────────────────────────────────

def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a context string for the LLM prompt."""
    if not docs:
        return "No relevant documents found."

    parts = []
    for i, doc in enumerate(docs, 1):
        collection = doc.metadata.get("collection", "unknown")
        score      = doc.metadata.get("relevance_score", "N/A")
        parts.append(
            f"[Doc {i} | {collection} | relevance: {score}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(parts)


def get_sources(docs: list[Document]) -> list[dict]:
    """Extract structured source metadata from retrieved documents."""
    sources = []
    for doc in docs:
        src = {
            "type"            : doc.metadata.get("source_type", "unknown"),
            "relevance_score" : doc.metadata.get("relevance_score", 0.0),
        }

        stype = doc.metadata.get("source_type", "")
        if stype == "train":
            src["train_no"]   = doc.metadata.get("train_no", "")
            src["train_name"] = doc.metadata.get("train_name", "")
        elif stype == "train_route":
            src["train_no"]   = doc.metadata.get("train_no", "")
            src["train_name"] = doc.metadata.get("train_name", "")
        elif stype == "station":
            src["station_code"] = doc.metadata.get("station_code", "")
            src["station_name"] = doc.metadata.get("station_name", "")
        elif stype == "rule":
            src["category"]   = doc.metadata.get("category", "")
            src["rule_title"] = doc.metadata.get("rule_title", "")
        elif stype == "reference":
            src["ref_type"]   = doc.metadata.get("ref_type", "")

        sources.append(src)
    return sources


# ─────────────────────────────────────────────
# RAG CHAIN
# ─────────────────────────────────────────────

class RAGChain:
    """
    Retrieval-Augmented Generation pipeline.

    Retrieve → Format Context → LLM → Structured Response
    """

    def __init__(self):
        self.retriever = get_unified_retriever(top_k=10)
        self.llm       = get_llm()
        self.prompt    = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human",  HUMAN_PROMPT),
        ])
        self.parser = StrOutputParser()
        print("✅ RAG Chain initialized")

    def invoke(self, question: str) -> dict:
        """
        Full RAG pipeline:
          1. Retrieve relevant documents from ChromaDB
          2. Format into context string
          3. Send to LLM with prompt
          4. Return structured response with sources
        """
        # Step 1: Retrieve
        docs = self.retriever.retrieve(question)

        # Step 2: Format context
        context = format_docs(docs)

        # Step 3: Generate answer
        chain  = self.prompt | self.llm | self.parser
        answer = chain.invoke({"context": context, "question": question})

        # Step 4: Extract sources
        sources = get_sources(docs)

        return {
            "question"               : question,
            "answer"                 : answer,
            "sources"                : sources,
            "num_documents_retrieved": len(docs),
        }


# ─────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────

_rag_chain: RAGChain | None = None


def get_rag_chain() -> RAGChain:
    """Get or create the singleton RAG chain instance."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain
