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

/* Follow-up suggestion chips (st.button widgets — small, pill-shaped) */
.followup-chip button {
    background: rgba(142,107,255,0.10) !important;
    border: 1px solid rgba(142,107,255,0.35) !important;
    color: #c9b8ff !important;
    font-size: 0.82rem !important;
    padding: 4px 10px !important;
    height: auto !important;
    border-radius: 999px !important;
}
.followup-chip button:hover { background: rgba(142,107,255,0.22) !important; }

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
        "conv_id":      str(uuid.uuid4())[:8],   # bumps on every "New chat"
        "messages":     [],
        "last_sources": [],
        "last_meta":    {},
        "pending_query": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # Drop legacy global 'followups' if present from older session.
    # Follow-up chips are now stored per assistant message.
    st.session_state.pop("followups", None)

_init_state()


# ── API helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=15)
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
    """Blocking call — returns the full enriched response dict.

    Two attempts: HF Spaces free tier sleeps after ~5 min idle, so the first
    request after a long pause can hit a TCP reset / 503 while the container
    starts. A single retry with a short pause covers that case cleanly.
    """
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
            # 5xx during cold-start → retry once
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
    """Best-effort follow-up suggestions; failures are silent (chips just don't show)."""
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


@st.cache_data(ttl=20)
def fetch_analytics() -> dict | None:
    """Sidebar live analytics. Cached briefly so fast reruns don't hammer the API."""
    try:
        r = requests.get(f"{API_URL}/analytics", timeout=8)
        return r.json() if r.ok else None
    except Exception:
        return None




def _fire_and_forget_delete(sid: str) -> None:
    """Background-thread DELETE — server LRU handles the slot anyway."""
    def _go():
        try:
            requests.delete(f"{API_URL}/chat/session/{sid}", timeout=5)
        except Exception:
            pass
    threading.Thread(target=_go, daemon=True).start()


def _do_full_reset():
    """
    On_click callback for "New chat". Resets every conversation key in
    session_state and bumps ``conv_id``. The chat zone is rendered inside
    ``st.container(key=f"chatzone_{conv_id}")`` — when ``conv_id`` changes,
    Streamlit assigns a new identity to the container and rebuilds its
    entire subtree from scratch instead of diffing element-by-element
    against the previous frame. That guarantees both stale chat_message
    DOM nodes (main Q/A bubbles) AND stale follow-up suggestion chips
    are evicted in one atomic swap, with no browser reload required.
    """
    old_sid = st.session_state.get("session_id", "")

    try:
        fetch_analytics.clear()
        api_health.clear()
    except Exception:
        pass

    # Re-assign every conversation key. The conv_id bump is what forces
    # the keyed chat container to swap identity on the next render.
    st.session_state["session_id"]    = str(uuid.uuid4())
    st.session_state["conv_id"]       = str(uuid.uuid4())[:8]
    st.session_state["messages"]      = []
    st.session_state["last_sources"]  = []
    st.session_state["last_meta"]     = {}
    st.session_state["pending_query"] = None
    st.session_state.pop("followups", None)

    # Sweep stale widget click-state from the previous conversation.
    stale_prefixes = ("fb_", "up_", "down_", "fu_", "chip_", "followup_")
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith(stale_prefixes):
            try:
                del st.session_state[key]
            except KeyError:
                pass

    # Server-side cleanup on a daemon thread (instant, doesn't block).
    if old_sid:
        _fire_and_forget_delete(old_sid)


def _refresh_ui():
    """
    Sidebar Refresh callback — clears caches so the next run re-checks the API
    and reloads analytics. Also nukes any parked ``pending_query`` so a
    half-finished cold-start click cannot resurrect after the reload.
    """
    try:
        api_health.clear()
        fetch_analytics.clear()
    except Exception:
        pass
    st.session_state.pending_query = None


def _queue_query(query: str):
    """Button-click callback — queue a query and let Streamlit auto-rerun."""
    if not query:
        return
    st.session_state.pending_query = query


# "New chat" reset is handled directly inside ``_do_full_reset`` (an
# on_click callback). No signal/iframe/reload step needed — the keyed
# chat container below (see chat_holder) does the DOM-rebuild work.


def _record_feedback(idx: int, rating: str):
    """
    Feedback callback — one click is enough because Streamlit auto-reruns
    after a callback finishes, and the next run sees ``fb_<conv_id>_<idx>``
    already set and renders the 'You rated…' caption instead of the buttons.
    """
    msgs = st.session_state.messages
    if idx <= 0 or idx >= len(msgs):
        return
    try:
        call_feedback(msgs[idx - 1]["content"], msgs[idx]["content"], rating)
    except Exception:
        # Never block the UI on feedback — rating still records locally.
        pass
    conv_id = st.session_state.get("conv_id", "")
    st.session_state[f"fb_{conv_id}_{idx}"] = rating


# ── Helpers ───────────────────────────────────────────────────────────────────

def confidence_class(c: float) -> str:
    if c >= 0.75: return "pill-green"
    if c >= 0.5:  return "pill-amber"
    return "pill-red"


def render_meta(meta: dict):
    """Intent pill + confidence meter + latency (below an assistant message)."""
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
    """Convert FAISS L2 distance (normalized vectors) → cosine similarity in [0,1]."""
    # text-embedding-3-small returns unit vectors, so ||a-b||² = 2 - 2·cos(θ)
    # ⇒ cos = 1 - L2²/2.  Clamp for display.
    return max(0.0, min(1.0, 1.0 - (l2 ** 2) / 2.0))


def render_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for i, src in enumerate(sources[:4]):
            cls = "top" if i == 0 else ""
            meta = src.get("metadata", {})
            meta_bits = []
            # Prefer real FAISS similarity; fall back gracefully
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
        # Roll back the orphan user turn so the transcript doesn't end on
        # an unanswered message — user can click the same chip / retype and
        # retry cleanly once the backend is awake.
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

    # Fire follow-up suggestions inline so they're stored ON the message.
    # That way each assistant message owns its chips — clearing messages
    # (e.g. via "New chat") removes the chips with them, and there's no
    # global followups state that can leak into the next session.
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


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/chatbot.png", width=64)
    st.title("ChatSolveAI")
    st.caption("LangChain RAG · FastAPI · MongoDB")
    st.divider()

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

    if healthy:
        analytics = fetch_analytics()
        if analytics:
            st.subheader("📊 Live analytics")
            col1, col2 = st.columns(2)
            col1.metric("Sessions", analytics.get("total_sessions", 0))
            col2.metric("Queries",  analytics.get("total_queries",  0))
            col1.metric("Today",    analytics.get("queries_today",  0))
            col2.metric("Avg turns", analytics.get("avg_session_length", 0))

            tops = analytics.get("top_questions", [])
            if tops:
                st.caption("🔥 Top questions")
                for item in tops[:5]:
                    st.markdown(f"- {item['question'][:50]}… `×{item['count']}`")
        st.divider()

    # Controls — on_click callbacks. Streamlit guarantees the callback
    # runs and the next render sees the new state; we deliberately do
    # NOT call st.rerun() inside the callback because doing so from a
    # handler that is already in a rerun cycle can cause Streamlit Cloud
    # to commit the new frame on top of the previous DOM (ghost frame).
    col1, col2 = st.columns(2)
    with col1:
        st.button(
            "🗑 New chat",
            key="btn_new_chat",
            use_container_width=True,
            on_click=_do_full_reset,
            help="Clears the conversation and starts a fresh session.",
        )
    with col2:
        st.button(
            "🔄 Refresh",
            key="btn_refresh",
            use_container_width=True,
            on_click=_refresh_ui,
            help="Re-checks the API and reloads sidebar analytics.",
        )

    # Export
    if st.session_state.messages:
        st.download_button(
            "⬇️ Export chat (.md)",
            data=build_transcript_md(),
            file_name=f"chatsolveai_{st.session_state.session_id[:8]}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.divider()
    st.markdown(
        "<small>LangChain · FAISS · GPT-3.5-turbo<br>"
        "MongoDB · FastAPI · Docker · HF Spaces</small>",
        unsafe_allow_html=True,
    )


# ── Main chat UI ──────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">💬 ChatSolveAI — Customer Support</div>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">LangChain RAG · GPT-3.5-turbo · MongoDB · FastAPI on Hugging Face. '
    'Retrieves from a curated knowledge base and falls back to generation for novel queries.</p>',
    unsafe_allow_html=True,
)

# All chat content (pending-query handler, example chips, message
# history, follow-up chips) renders inside a SINGLE st.container whose
# key is bound to ``conv_id``. When "New chat" bumps ``conv_id``, the
# container key changes — Streamlit assigns a new identity and rebuilds
# the entire subtree from scratch instead of diffing against the
# previous frame. That guarantees both stale chat_message DOM nodes
# (main Q/A bubbles) AND stale follow-up suggestion chips are evicted
# in one atomic swap, with no browser reload required.
chat_holder = st.container(key=f"chatzone_{st.session_state.conv_id}")

with chat_holder:
    # 1. Handle a queued query FIRST — before any chip / history render.
    if st.session_state.pending_query:
        if healthy:
            q = st.session_state.pending_query
            st.session_state.pending_query = None
            submit_query(q)
        else:
            st.warning(
                "Backend is waking up and didn't answer in time. "
                "Click **🔄 Refresh** in the sidebar in a few seconds, then retry."
            )
            st.session_state.pending_query = None

    # 2. Example-question chips (only on fresh conversations).
    if not st.session_state.messages:
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

    # 3. Conversation history. Widget keys carry conv_id so a fresh
    #    conversation cannot inherit click-state from a previous one.
    last_idx = len(st.session_state.messages) - 1
    conv_id  = st.session_state.conv_id
    for idx, msg in enumerate(st.session_state.messages):
        avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_meta(msg.get("meta", {}))
                render_sources(msg.get("sources", []))

                # Feedback row — one click thanks to on_click callback.
                fb_key = f"fb_{conv_id}_{idx}"
                if st.session_state.get(fb_key) is None:
                    c1, c2, _ = st.columns([1, 1, 8])
                    c1.button(
                        "👍",
                        key=f"up_{conv_id}_{idx}",
                        help="Good answer",
                        on_click=_record_feedback,
                        args=(idx, "up"),
                    )
                    c2.button(
                        "👎",
                        key=f"down_{conv_id}_{idx}",
                        help="Needs work",
                        on_click=_record_feedback,
                        args=(idx, "down"),
                    )
                else:
                    rating = st.session_state[fb_key]
                    st.caption(
                        f"You rated this answer: {'👍' if rating == 'up' else '👎'}"
                    )

    # 4. Follow-up chips — rendered into a DEDICATED st.empty() slot
    #    that lives OUTSIDE the message loop (but still inside the
    #    chat container). st.empty() is Streamlit's purpose-built
    #    primitive for atomic content swap: writing into it replaces
    #    the prior content; not writing leaves it empty (and any
    #    prior content is removed). This isolates chip rendering
    #    from the per-message loop, where the conditional-element
    #    diff was leaving stale chip DOM behind on Streamlit Cloud.
    #
    #    Chips are real ``st.button`` widgets so click handling stays
    #    inside Streamlit's normal callback flow — clicking a chip
    #    appends a new turn to the conversation; it does NOT clear
    #    the conversation. Only "New chat" clears.
    chips_slot = st.empty()
    if (
        st.session_state.messages
        and st.session_state.messages[-1].get("role") == "assistant"
    ):
        last_msg  = st.session_state.messages[-1]
        followups = last_msg.get("followups") or []
        if followups:
            with chips_slot.container():
                st.markdown(
                    '<div style="margin-top:14px; font-weight:600; '
                    'color:#c9b8ff;">💡 Suggested follow-ups</div>',
                    unsafe_allow_html=True,
                )
                cols = st.columns(len(followups))
                for i, q in enumerate(followups):
                    with cols[i]:
                        st.markdown(
                            '<div class="followup-chip">',
                            unsafe_allow_html=True,
                        )
                        st.button(
                            q,
                            key=f"fu_{st.session_state.conv_id}_{i}",
                            on_click=_queue_query,
                            args=(q,),
                        )
                        st.markdown('</div>', unsafe_allow_html=True)


# Input box (guarded behind health). Lives outside the chat holder so
# the input bar position doesn't jump between renders.
if prompt := st.chat_input("Ask about orders, billing, account, or technical issues…"):
    if not healthy:
        st.error("Cannot send message — API is not reachable.")
        st.stop()
    _queue_query(prompt)
    st.rerun()