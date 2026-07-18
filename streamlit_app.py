import os
import sys
import time
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

# ── Page Config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Railway RAG Assistant",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load API key from Streamlit secrets
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ.setdefault("LLM_PROVIDER", "gemini")
os.environ.setdefault("GEMINI_MODEL", "gemini-3.1-flash-lite")
os.environ.setdefault("USE_LOCAL_EMBEDDINGS", "false")

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%); min-height: 100vh; }
[data-testid="stSidebar"] { background: rgba(255,255,255,0.03) !important; border-right: 1px solid rgba(255,255,255,0.08) !important; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stMetric"] { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 10px 14px; margin: 4px 0; }
[data-testid="stMetricValue"] { font-weight: 700 !important; font-size: 1.4rem !important; color: #a78bfa !important; }
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: #94a3b8 !important; }
.stButton > button { background: rgba(167,139,250,0.1) !important; border: 1px solid rgba(167,139,250,0.3) !important; border-radius: 8px !important; color: #c4b5fd !important; font-size: 0.8rem !important; transition: all 0.2s ease !important; width: 100%; }
.stButton > button:hover { background: rgba(167,139,250,0.25) !important; border-color: #a78bfa !important; transform: translateX(2px); }
[data-testid="stChatInput"] input { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 12px !important; color: #f1f5f9 !important; }
[data-testid="stChatMessage"] { border-radius: 14px; margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.06); background: rgba(255,255,255,0.03); }
[data-testid="stExpander"] { background: rgba(255,255,255,0.03) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 10px !important; }
.stSpinner > div { border-top-color: #a78bfa !important; }
.main-title { font-size: 2.2rem; font-weight: 700; background: linear-gradient(135deg, #a78bfa, #60a5fa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.25rem; }
.main-subtitle { color: #64748b; font-size: 0.95rem; margin-bottom: 1.5rem; }
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1.5rem; }
.chip { background: rgba(167,139,250,0.1); border: 1px solid rgba(167,139,250,0.25); color: #c4b5fd; border-radius: 20px; padding: 4px 14px; font-size: 0.78rem; }
</style>
""", unsafe_allow_html=True)


# ── Load RAG Chain ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading RAG pipeline (first time ~60s)…")
def load_rag_chain():
    from app.rag import get_rag_chain
    return get_rag_chain()


def smart_query(question: str) -> dict:
    """
    Full smart pipeline mirroring /ask/smart:
      STATIC  → ChromaDB retrieval + LLM
      LIVE    → NTES live API + LLM
      HYBRID  → NTES + ChromaDB + LLM
      PNR     → PNR API + LLM
    """
    from app.intent import classify_intent
    from app.ntes_client import get_train_running_status, format_live_status_for_llm
    from app.pnr_client import get_pnr_status, format_pnr_status_for_llm
    from app.rag import get_sources, format_docs, SYSTEM_PROMPT, HUMAN_PROMPT
    from langchain_core.prompts import ChatPromptTemplate

    chain = load_rag_chain()
    t0 = time.time()

    # 1. Classify intent
    intent_res = classify_intent(question)
    intent   = intent_res["intent"]
    train_no = intent_res.get("train_no")
    is_pnr   = intent_res.get("is_pnr", False)
    pnr_val  = intent_res.get("pnr")

    docs, live_context, sources, warnings = [], "", [], []

    if is_pnr and pnr_val:
        try:
            pnr_status = get_pnr_status(pnr_val)
            if pnr_status.get("success"):
                live_context = format_pnr_status_for_llm(pnr_status)
                sources.insert(0, {"type": "pnr_status", "relevance_score": 1.0,
                                   "pnr": pnr_val, "train_name": pnr_status.get("train_name", "")})
            else:
                warnings.append(f"PNR fetch failed: {pnr_status.get('error')}")
                live_context = f"⚠️ PNR STATUS UNAVAILABLE: {pnr_status.get('error')}"
        except Exception as exc:
            warnings.append(str(exc))
    else:
        needs_static = intent in ("STATIC", "HYBRID")
        needs_live   = intent in ("LIVE", "HYBRID") and train_no

        if needs_static:
            q = f"{question} train {train_no}" if (train_no and train_no not in question) else question
            docs    = chain.retriever.retrieve(q)
            sources = get_sources(docs)

        if needs_live:
            try:
                status = get_train_running_status(train_no)
                if status.get("success"):
                    live_context = format_live_status_for_llm(status)
                    sources.insert(0, {
                        "type": "live_status", "relevance_score": 1.0,
                        "train_no": train_no, "train_name": status.get("train_name", ""),
                        "current_station": status.get("current_station", ""),
                        "source": status.get("source", "NTES"),
                    })
                else:
                    warnings.append(f"Live status unavailable: {status.get('error')}")
                    live_context = f"⚠️ LIVE STATUS UNAVAILABLE for train {train_no}. Using timetable only."
            except Exception as exc:
                warnings.append(str(exc))
                live_context = f"⚠️ LIVE STATUS ERROR: {exc}"

        if needs_live and not live_context:
            live_context = f"⚠️ LIVE STATUS UNAVAILABLE for train {train_no}."

    # Build context and generate answer
    static_ctx = format_docs(docs) if docs else ""
    full_ctx   = "\n\n".join(filter(None, [live_context, static_ctx])) or "No relevant data found."

    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
    answer = (prompt | chain.llm | chain.parser).invoke({"context": full_ctx, "question": question})

    return {
        "answer": answer,
        "sources": sources,
        "intent": intent,
        "train_no": train_no,
        "num_documents_retrieved": len(docs),
        "warnings": warnings,
        "elapsed": f"{round(time.time() - t0, 2)}s",
    }


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚂 Railway RAG")
    st.markdown("<span style='color:#64748b;font-size:0.85rem'>Indian Railways AI Assistant</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📊 Knowledge Base")
    c1, c2 = st.columns(2)
    c1.metric("Trains", "12,813"); c2.metric("Stations", "9,956")
    c1.metric("Routes", "10,158"); c2.metric("Rules", "183")
    st.markdown("---")
    st.markdown("### ⚡ Quick Questions")
    for q in ["Vijayawada → Hyderabad trains", "Stops of train 12727",
               "Cancellation charges AC", "Sleeper class luggage limit", "TTE duties on train"]:
        if st.button(q, key=f"btn_{q}", use_container_width=True):
            st.session_state["pending_query"] = q
    st.markdown("---")
    st.markdown("<span style='color:#475569;font-size:0.75rem'>Powered by Gemini · ChromaDB · LangChain</span>", unsafe_allow_html=True)


# ── Main Chat Area ───────────────────────────────────────────────────
st.markdown('<div class="main-title">🚂 Railway Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Ask about trains, routes, stations & rules</div>', unsafe_allow_html=True)
st.markdown("""<div class="chip-row">
  <span class="chip">Vijayawada → Hyderabad</span>
  <span class="chip">Stops of train 12727</span>
  <span class="chip">Cancellation charges</span>
  <span class="chip">Sleeper luggage limit</span>
  <span class="chip">Trains via RJY</span>
</div>""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            badge = f" · **{msg.get('intent','')}**" if msg.get("intent") else ""
            with st.expander(f"🔍 {len(msg['sources'])} sources · {msg.get('elapsed','')}{badge}"):
                for src in msg["sources"]:
                    t     = src.get("type", "Source")
                    name  = src.get("train_name") or src.get("station_name") or ""
                    score = src.get("relevance_score", 0.0)
                    cs    = f" · now at **{src['current_station']}**" if src.get("current_station") else ""
                    st.markdown(f"`{t}` **{name}**{cs} — relevance: `{score:.2f}`")
        for w in msg.get("warnings", []):
            st.warning(w)

query = st.chat_input("Ask about train 12727, live status, PNR, cancellation rules…")
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("*Classifying intent & retrieving data…*")
        try:
            result   = smart_query(query)
            answer   = result["answer"]
            sources  = result["sources"]
            intent   = result["intent"]
            elapsed  = result["elapsed"]
            n_docs   = result["num_documents_retrieved"]
            warnings = result["warnings"]

            placeholder.markdown(answer)

            if sources:
                with st.expander(f"🔍 {len(sources)} sources · {n_docs} docs · {elapsed} · **{intent}**"):
                    for src in sources:
                        t     = src.get("type", "Source")
                        name  = src.get("train_name") or src.get("station_name") or ""
                        score = src.get("relevance_score", 0.0)
                        cs    = f" · now at **{src['current_station']}**" if src.get("current_station") else ""
                        st.markdown(f"`{t}` **{name}**{cs} — relevance: `{score:.2f}`")

            for w in warnings:
                st.warning(w)

            st.caption(f"⚡ {elapsed} · {n_docs} docs · {intent} · Gemini + ChromaDB")

            st.session_state.messages.append({
                "role": "assistant", "content": answer,
                "sources": sources, "elapsed": elapsed,
                "intent": intent, "warnings": warnings,
            })
        except Exception as e:
            placeholder.error(f"❌ {e}")
            st.info("💡 Check GOOGLE_API_KEY in Streamlit secrets.")
