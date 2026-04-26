"""
ChatSolveAI — Streamlit frontend (v2.1).

Calls the FastAPI backend (Hugging Face Space or local) for all AI work.
Session state is persisted in MongoDB via the API.

Run standalone (API must be running separately):
    streamlit run app.py

Run via Docker (recommended):
    docker compose up --build
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
import urllib.parse
import uuid
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# ── Config ────────────────────────────────────────────────────────────────────

_SECRET_API_URL = None
try:
    _SECRET_API_URL = st.secrets.get("API_URL")
except Exception:
    pass

API_URL = (_SECRET_API_URL or os.getenv("API_URL") or "https://Nikollass-chatsolveai-api.hf.space").rstrip("/")

HEALTH_TIMEOUT_S = int(os.getenv("API_HEALTH_TIMEOUT", "20"))
HEALTH_RETRIES   = int(os.getenv("API_HEALTH_RETRIES", "2"))

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


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ChatSolveAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {
    --accent: #4F8BF9;
    --accent-2: #8E6BFF;
    --bg-card: rgba(255,255,255,0.03);
    --border: rgba(255,255,255,0.08);
}
.main .block-container { padding-top: 1rem; max-width: 980px; }

/* Hero gradient on the title */
.hero-title {
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
    font-size: 2rem;
    margin-bottom: 0.2rem;
}
.hero-sub { color: #9ea3b0; margin-bottom: 1.2rem; }

/* Intent pill */
.pill {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 2px 10px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600;
    background: rgba(79,139,249,0.12);
    color: var(--accent);
    margin-right: 6px;
}
.pill-green    { background: rgba(76,175,80,0.15);  color: #81c784; }
.pill-amber    { background: rgba(255,152,0,0.15);  color: #ffb74d; }
.pill-red      { background: rgba(244,67,54,0.15);  color: #e57373; }
.pill-purple   { background: rgba(156,39,176,0.15); color: #ba68c8; }

/* Confidence meter bar */
.meter-wrap {
    background: rgba(255,255,255,0.06); border-radius: 6px;
    height: 6px; width: 100%; overflow: hidden; margin-top: 4px;
}
.meter-fill { height: 100%; background: linear-gradient(90deg, #4CAF50, #81C784); }

/* Source cards */
.src-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid rgba(79,139,249,0.8);
    padding: 8px 12px; border-radius: 6px;
    margin-bottom: 8px;
    font-size: 0.82rem; line-height: 1.4;
    color: #d4d7dd;
}
.src-card.top { border-left-color: #66bb6a; background: rgba(102,187,106,0.08); }
.src-card .src-meta { font-size: 0.68rem; color: #7a8190; margin-top: 4px; }

/* Example chips */
.chip-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 20px; }
.chip-btn button {
    width: 100% !important;
    background: rgba(79,139,249,0.08) !important;
    border: 1px solid var(--border) !important;
    color: #d4d7dd !important;
    text-align: left !important;
    justify-content: flex-start !important;
    height: auto !important;
    padding: 10px 14px !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.15s ease;
}
.chip-btn button:hover {
    background: rgba(79,139,249,0.15) !important;
    border-color: var(--accent) !important;
}

/* Follow-up suggestion chips — now rendered as buttons */
.fu-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 8px;
}
.fu-chip-link {
    display: inline-block;
    background: rgba(142,107,255,0.10);
    border: 1px solid rgba(142,107,255,0.35);
    color: #c9b8ff !important;
    font-size: 0.82rem;
    padding: 6px 14px;
    border-radius: 999px;
    text-decoration: none !important;
    transition: background 0.15s ease, border-color 0.15s ease;
}
.fu-chip-link:hover {
    background: rgba(142,107,255,0.22);
    border-color: rgba(142,107,255,0.6);
}

/* Feedback buttons */
.fb-row { display: flex; gap: 6px; margin-top: 6px; }

/* Typing indicator */
@keyframes blink { 0%,100% { opacity: 0.2; } 50% { opacity: 1; } }
.typing span {
    display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent); margin-right: 4px; animation: blink 1.2s infinite;
}
.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }

/* Hide Streamlit chrome we don't need */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "session_id":   str(uuid.uuid4()),
        "conv_id":      str(uuid.uuid4())[:8],
        "messages":     [],
        "last_sources": [],
        "last_meta":    {},
        "pending_query": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    st.session_state.pop("followups", None)

_init_state()


# ── API helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=15, show_spinner=False)
def api_health() -> bool:
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


def call_chat(query: str) -> dict | None:
    """Blocking call — returns the full enriched response dict."""
    last_err = None
    for attempt in range(2):
        try:
            r = requests.post(
                f"{API_URL}/chat",
                json={"session_id": st.session_state.session_id, "query": query},
                timeout=60,
            )
            if r.ok:
                return r.json()
            if 500 <= r.status_code < 600 and attempt == 0:
                last_err = f"{r.status_code}: {r.text[:120]}"
                time.sleep(3)
                continue
            st.error(f"API error {r.status_code}: {r.text[:200]}")
            return None
        except requests.RequestException as e:
            last_err = str(e)
            if attempt == 0:
                time.sleep(3)
                continue
    st.error(f"Network error after retry: {last_err}")
    return None


def call_suggest(last_answer: str) -> list[str]:
    """Best-effort follow-up suggestions; failures are silent."""
    try:
        r = requests.post(
            f"{API_URL}/suggest",
            json={"last_answer": last_answer, "n": 3},
            timeout=25,
        )
        if r.ok:
            return r.json().get("suggestions", [])
    except requests.RequestException:
        pass
    return []


def call_feedback(query: str, answer: str, rating: str) -> bool:
    try:
        r = requests.post(
            f"{API_URL}/feedback",
            json={
                "session_id": st.session_state.session_id,
                "query":      query,
                "answer":     answer,
                "rating":     rating,
            },
            timeout=10,
        )
        return r.ok
    except requests.RequestException:
        return False


def _fire_and_forget_delete(sid: str) -> None:
    """Background-thread DELETE — server LRU handles the slot anyway."""
    def _go():
        try:
            requests.delete(f"{API_URL}/chat/session/{sid}", timeout=5)
        except Exception:
            pass
    threading.Thread(target=_go, daemon=True).start()


# ── Callbacks ─────────────────────────────────────────────────────────────────


def _queue_query(query: str):
    """Button-click callback — queue a query and let Streamlit auto-rerun."""
    if not query:
        return
    st.session_state.pending_query = query


def _record_feedback(idx: int, rating: str):
    """Feedback callback."""
    msgs = st.session_state.messages
    if idx <= 0 or idx >= len(msgs):
        return
    try:
        call_feedback(msgs[idx - 1]["content"], msgs[idx]["content"], rating)
    except Exception:
        pass
    conv_id = st.session_state.get("conv_id", "")
    st.session_state[f"fb_{conv_id}_{idx}"] = rating


# ── Helpers ───────────────────────────────────────────────────────────────────

def confidence_class(c: float) -> str:
    if c >= 0.75: return "pill-green"
    if c >= 0.5:  return "pill-amber"
    return "pill-red"


def render_meta(meta: dict):
    """Intent pill + confidence meter + latency."""
    if not meta:
        return
    intent = meta.get("intent", "general")
    info   = INTENT_META.get(intent, INTENT_META["general"])
    conf   = float(meta.get("confidence", 0.0))
    lat    = int(meta.get("latency_ms", 0))
    pill_class = confidence_class(conf)

    st.markdown(
        f'<div style="margin-top:8px; display:flex; gap:6px; flex-wrap:wrap; align-items:center;">'
        f'  <span class="pill">{info["emoji"]} {info["label"]}</span>'
        f'  <span class="pill {pill_class}">{int(conf*100)}% confidence</span>'
        f'  <span class="pill pill-purple">⚡ {lat} ms</span>'
        f'</div>'
        f'<div class="meter-wrap"><div class="meter-fill" style="width:{conf*100:.0f}%"></div></div>',
        unsafe_allow_html=True,
    )


def _similarity_from_l2(l2: float) -> float:
    """Convert FAISS L2 distance → cosine similarity in [0,1]."""
    return max(0.0, min(1.0, 1.0 - (l2 ** 2) / 2.0))


def render_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for i, src in enumerate(sources[:4]):
            cls = "top" if i == 0 else ""
            meta = src.get("metadata", {})
            meta_bits = []
            if src.get("score") is not None:
                sim = _similarity_from_l2(float(src["score"]))
                meta_bits.append(f'similarity: {sim:.2f}')
            if meta.get("topic"):
                meta_bits.append(f'topic: {meta["topic"]}')
            if meta.get("source_query"):
                meta_bits.append(f'matched q: {meta["source_query"][:60]}')
            meta_line = " · ".join(meta_bits) or "retrieved"
            rank = ["1st", "2nd", "3rd", "4th"][i]
            st.markdown(
                f'<div class="src-card {cls}">'
                f'<strong>{rank}</strong> — {src["content"][:220]}'
                f'{"…" if len(src["content"]) > 220 else ""}'
                f'<div class="src-meta">{meta_line}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def build_transcript_md() -> str:
    """Export the conversation as Markdown for download."""
    lines = [
        f"# ChatSolveAI Conversation",
        f"_Session `{st.session_state.session_id}` · "
        f"exported {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
    ]
    for m in st.session_state.messages:
        who = "**🧑 You**" if m["role"] == "user" else "**🤖 ChatSolveAI**"
        lines.append(f"{who}\n\n{m['content']}\n")
        if m["role"] == "assistant" and m.get("meta"):
            meta = m["meta"]
            lines.append(
                f"> intent: {meta.get('intent','general')} · "
                f"confidence: {int(float(meta.get('confidence',0))*100)}% · "
                f"latency: {meta.get('latency_ms',0)} ms\n"
            )
    return "\n".join(lines)


def submit_query(query: str):
    """Central entry point — always used whether query came from chip or input."""
    if not query or not str(query).strip():
        return
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking…"):
        result = call_chat(query)

    if not result:
        if (
            st.session_state.messages
            and st.session_state.messages[-1]["role"] == "user"
            and st.session_state.messages[-1]["content"] == query
        ):
            st.session_state.messages.pop()
        return

    answer     = result["answer"]
    sources    = result.get("source_documents", [])
    meta = {
        "intent":     result.get("intent", "general"),
        "confidence": result.get("confidence", 0.0),
        "latency_ms": result.get("latency_ms", 0),
        "condensed_query": result.get("condensed_query", query),
    }

    suggestions = call_suggest(answer)

    st.session_state.messages.append({
        "role":     "assistant",
        "content":  answer,
        "sources":  sources,
        "meta":     meta,
        "followups": suggestions,
    })
    st.session_state.last_sources = sources
    st.session_state.last_meta    = meta


# ── Reset handler (applied before any UI) ─────────────────────────────────────

def _perform_full_reset():
    old_sid = st.session_state.get("session_id", "")

    # Clear URL params and caches
    try:
        st.query_params.clear()
    except Exception:
        pass
    api_health.clear()

    # New conversation identity
    st.session_state["session_id"]    = str(uuid.uuid4())
    st.session_state["conv_id"]       = str(uuid.uuid4())[:8]
    st.session_state["messages"]      = []
    st.session_state["last_sources"]  = []
    st.session_state["last_meta"]     = {}
    st.session_state["pending_query"] = None
    st.session_state.pop("followups", None)

    # Remove stale widget keys from the old conversation
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith(("fb_", "up_", "down_", "fu_", "chip_", "followup_")):
            del st.session_state[key]

    if old_sid:
        _fire_and_forget_delete(old_sid)


# Reset is now triggered directly from the New-chat button handler
# (see sidebar below). No flag-based indirection needed — calling
# `_perform_full_reset()` then `st.rerun()` from inside the button
# block is the same pattern FinSight AI uses for its "Clear Chat"
# button, which works reliably on Streamlit Cloud.


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/chatbot.png", width=64)
    st.title("ChatSolveAI")
    # Tech stack lives at the top of the sidebar (single rendering site).
    # Previously this was duplicated as a sidebar footer too — Streamlit
    # Cloud was rendering the bottom block twice after `st.rerun()`
    # regardless of element type (`st.markdown` with `<small><br></small>`,
    # then `st.caption` with markdown soft-break — both leaked). Single
    # caption at the top, no bottom footer = no possible duplication.
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
            "The backend may be cold-starting (free-tier hosts sleep after idle). "
            "Wait ~30s and try again. To start locally:\n"
            "```\nuvicorn api.main:app --port 8000\n```"
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


# ── Main chat UI ──────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">💬 ChatSolveAI — Customer Support</div>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">LangChain RAG · GPT-3.5-turbo · MongoDB · FastAPI on Hugging Face. '
    'Retrieves from a curated knowledge base and falls back to generation for novel queries.</p>',
    unsafe_allow_html=True,
)

# Process any pending query FIRST — this lives outside the chat layout
# so the new turn ends up in `st.session_state.messages` *before* we
# decide which branch (empty state vs history) to render.
if st.session_state.pending_query:
    if healthy:
        q = st.session_state.pending_query
        st.session_state.pending_query = None
        submit_query(q)
    else:
        st.warning(
            "Backend is waking up and didn't answer in time. "
            "Wait ~30s and try again."
        )
        st.session_state.pending_query = None

# Mutually-exclusive empty-state vs history rendering — same pattern
# FinSight AI uses for its Strategy Copilot tab. Critical detail:
# when there are no messages, the history container is NEVER created.
# When New chat sets messages = [] and reruns, the entire chat
# container disappears from the element tree, so Streamlit cannot
# leave stale `st.chat_message` DOM behind — there is no parent
# element for those children to live under anymore.
if not st.session_state.messages:
    # Empty state — example chips only.
    st.markdown("**👋 Try one of these to get started:**")
    cols = st.columns(2)
    _conv = st.session_state.conv_id
    for i, (emoji, q) in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 2]:
            st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
            st.button(
                f"{emoji}   {q}",
                key=f"chip_{_conv}_{i}",
                use_container_width=True,
                on_click=_queue_query,
                args=(q,),
            )
            st.markdown('</div>', unsafe_allow_html=True)
else:
    # History state — fixed-height scrollable container holds the
    # chat_message bubbles. The container only exists in this branch,
    # so the empty-state branch is incapable of inheriting any DOM
    # from a previous frame.
    msgs     = st.session_state.messages
    last_idx = len(msgs) - 1
    conv_id  = st.session_state.conv_id

    history_height = min(640, max(280, len(msgs) * 130))
    history_container = st.container(height=history_height)
    with history_container:
        for idx, msg in enumerate(msgs):
            avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    render_meta(msg.get("meta", {}))
                    render_sources(msg.get("sources", []))

                    # Feedback buttons (stay inside the message).
                    fb_key = f"fb_{conv_id}_{idx}"
                    if st.session_state.get(fb_key) is None:
                        c1, c2, _ = st.columns([1, 1, 8])
                        with c1:
                            if st.button("👍", key=f"up_{conv_id}_{idx}", help="Good answer"):
                                _record_feedback(idx, "up")
                                st.rerun()
                        with c2:
                            if st.button("👎", key=f"down_{conv_id}_{idx}", help="Needs work"):
                                _record_feedback(idx, "down")
                                st.rerun()
                    else:
                        rating = st.session_state[fb_key]
                        st.caption(
                            f"You rated this answer: {'👍' if rating == 'up' else '👎'}"
                        )

    # Follow-up suggestion buttons live OUTSIDE the scrollable history
    # so the input bar doesn't get pushed off-screen, but they're
    # wrapped in their OWN dedicated `st.container()`. Why a wrapper:
    # `st.button` widgets emitted at the bare `else:` scope leaked
    # past the empty-state branch on Streamlit Cloud — the chip pills
    # from the previous conversation stayed painted alongside the
    # freshly rendered example chips after **New chat**. The fix is
    # the same trick the history container uses: when the `else:`
    # branch isn't taken, the wrapper container is never created,
    # so its parent element is gone and Streamlit removes the entire
    # chip subtree in one atomic step (the diff path that worked
    # for `history_container` in PR #25 also works here).
    chips_container = st.container()
    with chips_container:
        if msgs[-1]["role"] == "assistant":
            followups = (msgs[-1].get("followups") or [])
            if followups:
                st.markdown(
                    '<div style="margin-top:14px; font-weight:600; '
                    'color:#c9b8ff;">💡 Suggested follow-ups</div>',
                    unsafe_allow_html=True,
                )
                cols = st.columns(len(followups))
                fu_clicked = None
                for i, q in enumerate(followups):
                    with cols[i]:
                        st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
                        if st.button(
                            q,
                            key=f"fu_{conv_id}_{last_idx}_{i}",
                            use_container_width=True,
                        ):
                            fu_clicked = q
                        st.markdown('</div>', unsafe_allow_html=True)
                if fu_clicked:
                    st.session_state.pending_query = fu_clicked
                    st.rerun()

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about orders, billing, account, or technical issues…"):
    if not healthy:
        st.error("Cannot send message — API is not reachable.")
        st.stop()
    _queue_query(prompt)
    st.rerun()