"""
ChatSolveAI — Streamlit frontend.

Calls the FastAPI backend (api/main.py) for all AI work.
Session state is persisted in MongoDB via the API.

Run standalone (API must be running separately):
    streamlit run app.py

Run via Docker (recommended):
    docker compose up --build
"""

import json
import os
import time
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # picks up .env when running locally outside Docker

# ── Config ────────────────────────────────────────────────────────────────────
# API_URL can come from either a plain env var (local / Docker) or Streamlit
# secrets (Streamlit Community Cloud). Streamlit secrets take precedence.
_SECRET_API_URL = None
try:
    _SECRET_API_URL = st.secrets.get("API_URL")  # raises if no secrets.toml
except Exception:
    pass

API_URL = (_SECRET_API_URL or os.getenv("API_URL") or "https://chatsolve-api.onrender.com").rstrip("/")

# Cold-start-friendly health check. Render/HF free tiers can take 20-40s to
# wake from sleep — a 3s timeout misreports "unreachable" during that window.
HEALTH_TIMEOUT_S = int(os.getenv("API_HEALTH_TIMEOUT", "20"))
HEALTH_RETRIES   = int(os.getenv("API_HEALTH_RETRIES", "2"))

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChatSolveAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; }

    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
        color: white;
        margin-bottom: 6px;
    }
    .tag {
        font-size: 0.72rem;
        font-weight: 500;
        padding: 2px 8px;
        border-radius: 8px;
        display: inline-block;
    }
    .tag-rag      { background: #e8f5e9; color: #1b5e20; }
    .tag-fallback { background: #fbe9e7; color: #bf360c; }

    .src-card {
        background: #f8f9fa;
        border-left: 3px solid #90caf9;
        padding: 6px 10px;
        border-radius: 4px;
        margin-bottom: 6px;
        font-size: 0.82rem;
        color: #37474f;
    }
    .src-card.top { border-left-color: #1976d2; background: #e3f2fd; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []   # [{role, content, sources}]

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []


# ── API helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=15)
def api_health() -> bool:
    """
    Probe /health with a longer timeout + a few retries so that a cold-booting
    backend (free-tier Render / HF Space) is not misreported as 'unreachable'.
    Cached briefly so sidebar + input gate do not each fire their own request.
    """
    for attempt in range(HEALTH_RETRIES + 1):
        try:
            r = requests.get(f"{API_URL}/health", timeout=HEALTH_TIMEOUT_S)
            if r.ok:
                return True
        except requests.RequestException:
            pass
        if attempt < HEALTH_RETRIES:
            time.sleep(2)
    return False


def stream_response(query: str):
    """
    Generator that yields text tokens from the FastAPI SSE streaming endpoint.
    Designed to be passed directly to st.write_stream().
    """
    payload = {"session_id": st.session_state.session_id, "query": query}
    sources: list[str] = []

    with requests.post(
        f"{API_URL}/chat/stream",
        json=payload,
        stream=True,
        timeout=60,
    ) as r:
        r.raise_for_status()
        for raw_line in r.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                yield data.get("token", "")
            except json.JSONDecodeError:
                pass

    st.session_state.last_sources = sources


def fetch_analytics() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/analytics", timeout=5)
        return r.json() if r.ok else None
    except Exception:
        return None


def reset_session():
    sid = st.session_state.session_id
    try:
        requests.delete(f"{API_URL}/chat/session/{sid}", timeout=5)
    except Exception:
        pass
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages   = []
    st.session_state.last_sources = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/chatbot.png", width=64)
    st.title("ChatSolveAI")
    st.caption("LangChain RAG · FastAPI · MongoDB")
    st.divider()

    # API status indicator
    healthy = api_health()
    if healthy:
        st.success("API connected", icon="✅")
    else:
        st.error(f"API unreachable at {API_URL}", icon="🔴")
        st.info(
            "The backend may be cold-starting (free-tier hosts sleep after idle). "
            "Wait ~30s and click **Refresh**. To start locally:\n"
            "```\nuvicorn api.main:app --port 8000\n```"
        )

    st.caption(f"Session: `{st.session_state.session_id[:8]}…`")
    st.divider()

    # Source documents from last assistant turn
    if st.session_state.last_sources:
        st.subheader("Retrieved sources")
        for i, src in enumerate(st.session_state.last_sources[:3]):
            css = "top" if i == 0 else ""
            label = ["1st", "2nd", "3rd"][i]
            st.markdown(
                f'<div class="src-card {css}"><strong>{label}</strong> — {src[:100]}…</div>',
                unsafe_allow_html=True,
            )
        st.divider()

    # Analytics panel
    if healthy:
        analytics = fetch_analytics()
        if analytics:
            st.subheader("Usage analytics")
            col1, col2 = st.columns(2)
            col1.metric("Sessions", analytics.get("total_sessions", 0))
            col2.metric("Queries",  analytics.get("total_queries",  0))
            col1.metric("Today",    analytics.get("queries_today",  0))
            col2.metric("Avg turns", analytics.get("avg_session_length", 0))

            tops = analytics.get("top_questions", [])
            if tops:
                st.caption("Top questions")
                for item in tops[:5]:
                    st.markdown(f"- {item['question'][:50]}… `×{item['count']}`")
        st.divider()

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 New chat", use_container_width=True):
            reset_session()
            st.rerun()
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            api_health.clear()
            st.rerun()

    st.divider()
    st.markdown(
        "<small>LangChain · FAISS · GPT-3.5-turbo<br>"
        "MongoDB · FastAPI · Docker</small>",
        unsafe_allow_html=True,
    )


# ── Main chat UI ──────────────────────────────────────────────────────────────
st.markdown("## 💬 ChatSolveAI — Customer Support")
st.caption(
    "Powered by LangChain RAG + GPT-3.5-turbo. "
    "Retrieves from the knowledge base; falls back to generation for novel queries."
)

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Ask about orders, billing, account, or technical issues…"):

    if not healthy:
        st.error("Cannot send message — API is not reachable.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

    with st.chat_message("assistant"):
        full_response = st.write_stream(stream_response(prompt))

    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_response,
        "sources": st.session_state.last_sources,
    })
