import os
import sys
import time
import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# --- Page Config ---
st.set_page_config(
    page_title="Railway RAG Assistant 🚂",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load API key from Streamlit secrets
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ.setdefault("LLM_PROVIDER", "gemini")
os.environ.setdefault("GEMINI_MODEL", "gemini-3.1-flash-lite")
os.environ.setdefault("USE_LOCAL_EMBEDDINGS", "true")


@st.cache_resource(show_spinner="Loading RAG pipeline (first time ~60s)...")
def load_rag_chain():
    """Load the RAG chain once and cache it for all users."""
    from app.rag import get_rag_chain
    return get_rag_chain()


# ─── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Indian-railway-logo.svg/120px-Indian-railway-logo.svg.png", width=80)
    st.title("Railway RAG")
    st.caption("Indian Railways AI Assistant")
    st.markdown("---")

    st.subheader("📊 Knowledge Base")
    col1, col2 = st.columns(2)
    col1.metric("Trains", "12,813")
    col2.metric("Stations", "9,956")
    col1.metric("Routes", "10,158")
    col2.metric("Rules", "183")

    st.markdown("---")
    st.subheader("⚡ Quick Questions")
    examples = [
        "Vijayawada → Hyderabad trains",
        "Stops of train 12727",
        "Cancellation charges AC",
        "Sleeper class luggage limit",
        "TTE duties",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["pending_query"] = ex

    st.markdown("---")
    st.caption("Powered by Gemini · ChromaDB · LangChain")


# ─── Main Chat Area ───────────────────────────────────────────────────
st.title("🚂 Railway Assistant")
st.markdown("*Ask about trains, routes, stations & rules*")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} sources retrieved"):
                for src in msg["sources"]:
                    src_type = src.get("type", "Source")
                    name = src.get("train_name") or src.get("station_name") or src.get("rule_title") or ""
                    score = src.get("relevance_score", 0.0)
                    st.markdown(f"- **{src_type}** {name} — relevance: `{score:.2f}`")

# Get query — either from chat input or from sidebar button
query = st.chat_input("Ask about train 12727, cancellation rules, Vijayawada station...")
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

# Handle new query
if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving documents and generating answer..."):
            try:
                chain = load_rag_chain()
                t0 = time.time()
                result = chain.invoke(query)
                elapsed = round(time.time() - t0, 2)

                answer  = result.get("answer", "No answer generated.")
                sources = result.get("sources", [])
                n_docs  = result.get("num_documents_retrieved", 0)

                st.markdown(answer)

                if sources:
                    with st.expander(f"📎 {len(sources)} sources retrieved · {n_docs} docs · {elapsed}s"):
                        for src in sources:
                            src_type = src.get("type", "Source")
                            name = src.get("train_name") or src.get("station_name") or src.get("rule_title") or ""
                            score = src.get("relevance_score", 0.0)
                            st.markdown(f"- **{src_type}** {name} — relevance: `{score:.2f}`")

                st.caption(f"⚡ Generated in {elapsed}s · {n_docs} documents retrieved")

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.info("Make sure GOOGLE_API_KEY is set in Streamlit secrets.")
