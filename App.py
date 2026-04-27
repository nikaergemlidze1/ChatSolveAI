"""
ChatSolveAI — Streamlit frontend (v2.1).

Calls the FastAPI backend for all AI work. Session state is persisted
via the backend's MongoDB store.

run standalone:   streamlit run App.py
run via Docker:   docker compose up --build
"""

from __future__ import annotations

import json, os, time, uuid
from datetime import datetime

import requests, streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Config ──────────────────────────────────────────────────────
_SECRET_API_URL = _SECRET_API_KEY = None
try:
    _SECRET_API_URL = st.secrets.get("API_URL")
    _SECRET_API_KEY = st.secrets.get("API_KEY")
except Exception:
    pass

API_URL = (_SECRET_API_URL or os.getenv("API_URL") or "https://Nikollass-chatsolveai-api.hf.space").rstrip("/")
API_KEY = _SECRET_API_KEY or os.getenv("API_KEY") or ""

HEALTH_TIMEOUT_S = int(os.getenv("API_HEALTH_TIMEOUT", "20"))
HEALTH_RETRIES   = int(os.getenv("API_HEALTH_RETRIES", "2"))
USE_STREAMING    = os.getenv("USE_STREAMING", "true").lower() not in {"0", "false", "no"}


def _api_headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def _session_id_from_url() -> str | None:
    try:
        sid = st.query_params.get("sid")
    except Exception:
        return None
    if isinstance(sid, list):
        sid = sid[0] if sid else None
    sid = str(sid or "").strip()
    if not sid or len(sid) > 128:
        return None
    return sid


def _sync_session_url() -> None:
    try:
        if st.query_params.get("sid") != st.session_state.session_id:
            st.query_params["sid"] = st.session_state.session_id
    except Exception:
        pass


def _adopt_url_session() -> None:
    url_session_id = _session_id_from_url()
    if not url_session_id or url_session_id == st.session_state.get("session_id"):
        return
    st.session_state["session_id"] = url_session_id
    st.session_state["conv_id"]    = str(uuid.uuid4())[:8]
    st.session_state["messages"]   = []
    st.session_state["last_sources"] = []
    st.session_state["last_meta"]  = {}
    st.session_state["pending_query"] = None
    st.session_state["pending_append_user"] = True
    st.session_state["history_loaded_for"] = None


USER_AVATAR      = "🧑"
ASSISTANT_AVATAR = "🤖"

INTENT_META = {
    "billing":   {"label": "Billing",   "emoji": "💳", "color": "#FF9800"},
    "account":   {"label": "Account",   "emoji": "🔐", "color": "#2196F3"},
    "shipping":  {"label": "Shipping",  "emoji": "📦", "color": "#4CAF50"},
    "technical": {"label": "Technical", "emoji": "🛠️", "color": "#9C27B0"},
    "general":   {"label": "General",   "emoji": "💬", "color": "#607D8B"},
}

EXAMPLE_QUESTIONS = [
    ("🔐", "How do I reset my password?"),
    ("📦", "Where is my order?"),
    ("💳", "How do I get a refund?"),
    ("🚫", "How do I cancel my subscription?"),
]

# ── Page setup ──────────────────────────────────────────────────
st.set_page_config(page_title="ChatSolveAI", page_icon="🤖", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""<style>
:root { --accent: #4F8BF9; --accent-2: #8E6BFF; }
.main .block-container { padding-top: 1rem; max-width: 980px; }
.hero-title {
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; font-weight: 800; font-size: 2rem; margin-bottom: 0.2rem;
}
.hero-sub { color: #9ea3b0; margin-bottom: 1.2rem; }
.pill {
    display: inline-flex; align-items: center; gap: 4px; padding: 2px 10px;
    border-radius: 999px; font-size: 0.72rem; font-weight: 600;
    background: rgba(79,139,249,0.12); color: var(--accent); margin-right: 6px;
}
.pill-green  { background: rgba(76,175,80,0.15);  color: #81c784; }
.pill-amber  { background: rgba(255,152,0,0.15);  color: #ffb74d; }
.pill-red    { background: rgba(244,67,54,0.15);  color: #e57373; }
.pill-purple { background: rgba(156,39,176,0.15); color: #ba68c8; }
.meter-wrap { background: rgba(255,255,255,0.06); border-radius: 6px; height: 6px;
              width: 100%; overflow: hidden; margin-top: 4px; }
.meter-fill { height: 100%; background: linear-gradient(90deg, #4CAF50, #81C784); }
/* … other styles omitted for brevity … */
</style>""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────
def _init_state():
    url_session_id = _session_id_from_url()
    defaults = {
        "session_id":   url_session_id or str(uuid.uuid4()),
        "conv_id":      str(uuid.uuid4())[:8],
        "messages":     [],
        "last_sources": [],
        "last_meta":    {},
        "pending_query": None,
        "pending_append_user": True,
        "history_loaded_for": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    st.session_state.pop("followups", None)

_init_state()
_adopt_url_session()
_sync_session_url()

# … All the same functions: api_health, call_chat, call_chat_stream,
# call_history, call_suggest, call_feedback, confidence_class,
# render_meta, render_sources, build_transcript_md, submit_query,
# _perform_full_reset …

# ── Sidebar (only on the main App page) ───────────────────────────
if st.session_state.get("_current_page", "app") == "app":
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/chatbot.png", width=64)
        st.title("ChatSolveAI")
        st.caption(
            "LangChain · FAISS · GPT-3.5-turbo  \n"
            "MongoDB · FastAPI · Docker · HF Spaces"
        )
        st.divider()

        healthy = api_health()
        if healthy:
            st.success("API connected", icon="✅")
        else:
            st.error(f"API unreachable at {API_URL}", icon="🔴")
            st.info(
                "Cold‑start may be needed — wait ~30 s and try again."
            )

        st.caption(f"Session: `{st.session_state.session_id[:8]}…`")
        st.divider()

        if st.button("🗑 New chat", key="btn_new_chat", use_container_width=True,
                     help="Clears the conversation and starts a fresh session."):
            _perform_full_reset()
            st.rerun()

        if st.session_state.messages:
            st.download_button(
                "⬇️ Export chat (.md)",
                data=build_transcript_md(),
                file_name=f"chatsolveai_{st.session_state.session_id[:8]}.md",
                mime="text/markdown",
                use_container_width=True,
            )
else:
    # Admin dashboard gets an absolutely empty sidebar
    st.sidebar.empty()

# ── Main chat UI ────────────────────────────────────────────────
st.markdown('<div class="hero-title">💬 ChatSolveAI — Customer Support</div>',
            unsafe_allow_html=True)
st.markdown('<p class="hero-sub">LangChain RAG · GPT‑3.5‑turbo · MongoDB · FastAPI …</p>',
            unsafe_allow_html=True)

if st.session_state.pending_query:
    if healthy:
        q = st.session_state.pending_query
        append_user = bool(st.session_state.get("pending_append_user", True))
        st.session_state.pending_query = None
        st.session_state.pending_append_user = True
        if submit_query(q, append_user=append_user):
            st.rerun()
    else:
        st.warning("Backend cold‑starting … try again in ~30s.")
        st.session_state.pending_query = None
        st.session_state.pending_append_user = True

if not st.session_state.messages:
    st.markdown("**👋 Try one of these to get started:**")
    cols = st.columns(2)
    _conv = st.session_state.conv_id
    for i, (emoji, q) in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 2]:
            st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
            st.button(f"{emoji}   {q}", key=f"chip_{_conv}_{i}",
                      use_container_width=True, on_click=_queue_query, args=(q,))
            st.markdown('</div>', unsafe_allow_html=True)
else:
    msgs     = st.session_state.messages
    last_idx = len(msgs) - 1
    conv_id  = st.session_state.conv_id

    history_container = st.container(height=900)
    with history_container:
        for idx, msg in enumerate(msgs):
            with st.chat_message(msg["role"],
                                 avatar=USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    render_meta(msg.get("meta", {}))
                    render_sources(msg.get("sources", []))
                    fb_key = f"fb_{conv_id}_{idx}"
                    if st.session_state.get(fb_key) is None:
                        c1, c2, _ = st.columns([1, 1, 8])
                        with c1:
                            if st.button("👍", key=f"up_{conv_id}_{idx}"):
                                _record_feedback(idx, "up")
                                st.rerun()
                        with c2:
                            if st.button("👎", key=f"down_{conv_id}_{idx}"):
                                _record_feedback(idx, "down")
                                st.rerun()
                    else:
                        rating = st.session_state[fb_key]
                        st.caption(f"You rated: {'👍' if rating == 'up' else '👎'}")
                        if rating == "down":
                            if st.button("Regenerate", key=f"regen_{conv_id}_{idx}"):
                                _queue_regenerate(idx)
                                del st.session_state[fb_key]
                                st.rerun()

    # Follow‑up chips
    chips_container = st.container()
    with chips_container:
        if msgs[-1]["role"] == "assistant":
            followups = msgs[-1].get("followups") or []
            if followups:
                st.markdown('<div style="margin-top:14px; font-weight:600; '
                            'color:#c9b8ff;">💡 Suggested follow‑ups</div>',
                            unsafe_allow_html=True)
                cols = st.columns(len(followups))
                fu_clicked = None
                for i, q in enumerate(followups):
                    with cols[i]:
                        st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
                        if st.button(q, key=f"fu_{conv_id}_{last_idx}_{i}",
                                     use_container_width=True):
                            fu_clicked = q
                        st.markdown('</div>', unsafe_allow_html=True)
                if fu_clicked:
                    st.session_state.pending_query = fu_clicked
                    st.rerun()

# ── Chat input ──────────────────────────────────────────────────
if prompt := st.chat_input("Ask about orders, billing, account, or technical issues…"):
    if not healthy:
        st.error("Cannot send message — API unreachable.")
        st.stop()
    _queue_query(prompt)
    st.rerun()