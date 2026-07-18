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

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.03) !important;
    border-right: 1px solid rgba(255,255,255,0.08) !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* Hide default header */
[data-testid="stHeader"] { background: transparent !important; }

/* Metric cards in sidebar */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 10px 14px;
    margin: 4px 0;
}
[data-testid="stMetricValue"] { font-weight: 700 !important; font-size: 1.4rem !important; color: #a78bfa !important; }
[data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: #94a3b8 !important; }

/* Quick question buttons */
.stButton > button {
    background: rgba(167, 139, 250, 0.1) !important;
    border: 1px solid rgba(167, 139, 250, 0.3) !important;
    border-radius: 8px !important;
    color: #c4b5fd !important;
    font-size: 0.8rem !important;
    padding: 6px 12px !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background: rgba(167, 139, 250, 0.25) !important;
    border-color: #a78bfa !important;
    transform: translateX(2px);
}

/* Chat input */
[data-testid="stChatInput"] input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    border-radius: 14px;
    margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.06);
    background: rgba(255,255,255,0.03);
}

/* User message */
[data-testid="stChatMessage"][data-testid*="user"] {
    background: rgba(167, 139, 250, 0.08) !important;
    border-color: rgba(167, 139, 250, 0.2) !important;
}

/* Spinner */
.stSpinner > div { border-top-color: #a78bfa !important; }

/* Expander */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
}

/* Title area */
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
}
.main-subtitle {
    color: #64748b;
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1.5rem; }
.chip {
    background: rgba(167,139,250,0.1);
    border: 1px solid rgba(167,139,250,0.25);
    color: #c4b5fd;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)


# ── Load RAG Chain ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading RAG pipeline (first time ~60s)…")
def load_rag_chain():
    from app.rag import get_rag_chain
    return get_rag_chain()


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚂 Railway RAG")
    st.markdown("<span style='color:#64748b;font-size:0.85rem'>Indian Railways AI Assistant</span>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📊 Knowledge Base")
    c1, c2 = st.columns(2)
    c1.metric("Trains", "12,813")
    c2.metric("Stations", "9,956")
    c1.metric("Routes", "10,158")
    c2.metric("Rules", "183")

    st.markdown("---")
    st.markdown("### ⚡ Quick Questions")
    quick_qs = [
        "Vijayawada → Hyderabad trains",
        "Stops of train 12727",
        "Cancellation charges AC",
        "Sleeper class luggage limit",
        "TTE duties on train",
    ]
    for q in quick_qs:
        if st.button(q, key=f"btn_{q}", use_container_width=True):
            st.session_state["pending_query"] = q

    st.markdown("---")
    st.markdown("<span style='color:#475569;font-size:0.75rem'>Powered by Gemini · ChromaDB · LangChain</span>", unsafe_allow_html=True)


# ── Main Chat Area ───────────────────────────────────────────────────
st.markdown('<div class="main-title">🚂 Railway Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Ask about trains, routes, stations & rules</div>', unsafe_allow_html=True)

# Suggestion chips (visual only)
st.markdown("""
<div class="chip-row">
  <span class="chip">Vijayawada → Hyderabad</span>
  <span class="chip">Stops of train 12727</span>
  <span class="chip">Cancellation charges</span>
  <span class="chip">Sleeper luggage limit</span>
  <span class="chip">Trains via RJY</span>
</div>
""", unsafe_allow_html=True)

# Chat history init
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"🔍 {len(msg['sources'])} sources  ·  {msg.get('elapsed', '')}"):
                for src in msg["sources"]:
                    t   = src.get("type", "Source")
                    name = src.get("train_name") or src.get("station_name") or src.get("rule_title") or ""
                    score = src.get("relevance_score", 0.0)
                    st.markdown(f"`{t}` **{name}** — relevance: `{score:.2f}`")

# Get query
query = st.chat_input("Ask about train 12727, cancellation rules, Vijayawada station…")
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

# Handle query
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("*Retrieving documents…*")
        try:
            chain  = load_rag_chain()
            t0     = time.time()
            result = chain.invoke(query)
            elapsed = f"{round(time.time() - t0, 2)}s"

            answer  = result.get("answer", "No answer generated.")
            sources = result.get("sources", [])
            n_docs  = result.get("num_documents_retrieved", 0)

            placeholder.markdown(answer)

            if sources:
                with st.expander(f"🔍 {len(sources)} sources retrieved · {n_docs} docs · {elapsed}"):
                    for src in sources:
                        t    = src.get("type", "Source")
                        name = src.get("train_name") or src.get("station_name") or src.get("rule_title") or ""
                        score = src.get("relevance_score", 0.0)
                        st.markdown(f"`{t}` **{name}** — relevance: `{score:.2f}`")

            st.caption(f"⚡ {elapsed} · {n_docs} documents · Gemini + ChromaDB")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "elapsed": elapsed,
            })
        except Exception as e:
            placeholder.error(f"❌ {e}")
            st.info("💡 Make sure `GOOGLE_API_KEY` is set in Streamlit secrets.")
