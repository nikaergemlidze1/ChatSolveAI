"""
ChatSolveAI — Streamlit frontend (v3.0) — single page with sidebar nav.

Replaces the previous multipage layout (App.py + pages/1_Admin_Dashboard.py).
Streamlit's pages/ directory caused DOM remnants to persist across page swaps;
folding both views into one script eliminates that class of bug entirely.
"""

from __future__ import annotations
import json, os, time, uuid
from datetime import datetime
import requests, streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════
_SECRET_API_URL = _SECRET_API_KEY = _SECRET_ADMIN_PW = None
try:
    _SECRET_API_URL = st.secrets.get("API_URL")
    _SECRET_API_KEY = st.secrets.get("API_KEY")
    _SECRET_ADMIN_PW = st.secrets.get("ADMIN_PASSWORD")
except: pass

API_URL = (_SECRET_API_URL or os.getenv("API_URL") or "https://Nikollass-chatsolveai-api.hf.space").rstrip("/")
API_KEY = _SECRET_API_KEY or os.getenv("API_KEY") or ""
ADMIN_PASSWORD = _SECRET_ADMIN_PW or os.getenv("ADMIN_PASSWORD") or ""
HEALTH_TIMEOUT_S = int(os.getenv("API_HEALTH_TIMEOUT","20"))
HEALTH_RETRIES   = int(os.getenv("API_HEALTH_RETRIES","2"))
USE_STREAMING    = os.getenv("USE_STREAMING","true").lower() not in {"0","false","no"}

def _api_headers(): return {"X-API-Key": API_KEY} if API_KEY else {}

# ══════════════════════════════════════════════
# Page setup
# ══════════════════════════════════════════════
st.set_page_config(page_title="ChatSolveAI", page_icon="🤖", layout="wide",
                   initial_sidebar_state="expanded")

# ══════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════
st.markdown("""<style>
:root {--accent:#4F8BF9;--accent-2:#8E6BFF;}
.hero-title{background:linear-gradient(90deg,var(--accent)0%,var(--accent-2)100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;font-weight:800;font-size:2rem;margin-bottom:0.2rem}
.hero-sub{color:#9ea3b0;margin-bottom:1.2rem}
.pill{display:inline-flex;align-items:center;gap:4px;padding:2px 10px;border-radius:999px;font-size:0.72rem;font-weight:600;background:rgba(79,139,249,.12);color:var(--accent);margin-right:6px}
.pill-green{background:rgba(76,175,80,.15);color:#81c784}
.pill-amber{background:rgba(255,152,0,.15);color:#ffb74d}
.pill-red{background:rgba(244,67,54,.15);color:#e57373}
.pill-purple{background:rgba(156,39,176,.15);color:#ba68c8}
.meter-wrap{background:rgba(255,255,255,.06);border-radius:6px;height:6px;overflow:hidden;margin-top:4px}
.meter-fill{height:100%;background:linear-gradient(90deg,#4CAF50,#81C784)}
.src-card{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-left:3px solid rgba(79,139,249,.8);padding:8px 12px;border-radius:6px;margin-bottom:8px;font-size:.82rem;color:#d4d7dd}
.src-card.top{border-left-color:#66bb6a;background:rgba(102,187,106,.08)}
.src-meta{font-size:.68rem;color:#7a8190;margin-top:4px}
.chip-btn button{width:100%!important;background:rgba(79,139,249,.08)!important;border:1px solid rgba(255,255,255,.08)!important;color:#d4d7dd!important;text-align:left!important;padding:10px 14px!important;border-radius:10px!important}
.chip-btn button:hover{background:rgba(79,139,249,.15)!important;border-color:#4F8BF9!important}
#MainMenu,footer{visibility:hidden}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# Session state helpers (chat only)
# ══════════════════════════════════════════════
def _session_id_from_url():
    try: sid = st.query_params.get("sid")
    except: return None
    if isinstance(sid, list): sid = sid[0] if sid else None
    return str(sid).strip() if sid and len(str(sid))<=128 else None

def _sync_session_url():
    try:
        if st.query_params.get("sid") != st.session_state.session_id:
            st.query_params["sid"] = st.session_state.session_id
    except: pass

def _adopt_url_session():
    url_id = _session_id_from_url()
    if not url_id or url_id == st.session_state.get("session_id"): return
    st.session_state["session_id"] = url_id
    st.session_state["conv_id"] = str(uuid.uuid4())[:8]
    st.session_state["messages"] = []
    st.session_state["last_sources"] = []
    st.session_state["last_meta"] = {}
    st.session_state["pending_query"] = None
    st.session_state["pending_append_user"] = True

USER_AVATAR = "🧑"
ASSISTANT_AVATAR = "🤖"
INTENT_META = {
    "billing":{"label":"Billing","emoji":"💳"},"account":{"label":"Account","emoji":"🔐"},
    "shipping":{"label":"Shipping","emoji":"📦"},"technical":{"label":"Technical","emoji":"🛠️"},
    "general":{"label":"General","emoji":"💬"}
}
EXAMPLE_QUESTIONS = [
    ("🔐","How do I reset my password?"), ("📦","Where is my order?"),
    ("💳","How do I get a refund?"), ("🚫","How do I cancel my subscription?")
]

# Topical groupings for the start-page quick-question UI.
# Each category expands into 3 representative questions on click.
TOPIC_CATEGORIES = [
    ("🔐", "Account", [
        "How do I reset my password?",
        "How do I unlock my account?",
        "How do I enable two-factor authentication?",
    ]),
    ("📦", "Orders", [
        "Where is my order?",
        "How long does delivery take?",
        "Can I cancel my order after it has shipped?",
    ]),
    ("💳", "Refunds", [
        "How do I get a refund?",
        "What is your refund policy?",
        "How long does a refund take to process?",
    ]),
    ("🚫", "Subscription", [
        "How do I cancel my subscription?",
        "How do I update my payment method?",
        "Where can I view my billing history?",
    ]),
]

def _init_state():
    url_id = _session_id_from_url()
    for k,v in {"session_id":url_id or str(uuid.uuid4()),"conv_id":str(uuid.uuid4())[:8],
                "messages":[],"last_sources":[],"last_meta":{},"pending_query":None,
                "pending_append_user":True,"history_loaded_for":None}.items():
        if k not in st.session_state: st.session_state[k] = v
    st.session_state.pop("followups",None)

# ══════════════════════════════════════════════
# API helpers
# ══════════════════════════════════════════════
@st.cache_data(ttl=15,show_spinner=False)
def api_health():
    for _ in range(HEALTH_RETRIES+1):
        try:
            r = requests.get(f"{API_URL}/health",timeout=HEALTH_TIMEOUT_S)
            if r.ok: return True
        except: pass
        time.sleep(2)
    return False

def call_chat(query):
    for attempt in range(2):
        try:
            r = requests.post(f"{API_URL}/chat",json={"session_id":st.session_state.session_id,"query":query},headers=_api_headers(),timeout=60)
            if r.ok: return r.json()
            if 500<=r.status_code<600 and attempt==0: time.sleep(3); continue
        except: time.sleep(3)
    return None

def call_chat_stream(query, output_box=None):
    parts,final,err = [],[],None
    try:
        with requests.post(f"{API_URL}/chat/stream",json={"session_id":st.session_state.session_id,"query":query},headers=_api_headers(),timeout=90,stream=True) as r:
            if not r.ok: return None
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"): continue
                payload = line.removeprefix("data:").strip()
                if payload=="[DONE]": break
                try: ev = json.loads(payload)
                except: continue
                if "token" in ev:
                    parts.append(ev["token"])
                    if output_box: output_box.markdown("".join(parts))
                elif ev.get("event")=="final": final = ev
    except Exception as e: err = str(e)
    if final:
        final.setdefault("answer","".join(parts)); return final
    if parts:
        return {"answer":"".join(parts),"source_documents":[],"confidence":0.0,
                "condensed_query":query,"intent":"general","latency_ms":0}
    if err: st.error(f"Stream error: {err}")
    return None

def call_suggest(answer):
    try:
        r = requests.post(f"{API_URL}/suggest",json={"last_answer":answer,"n":3},headers=_api_headers(),timeout=25)
        if r.ok: return r.json().get("suggestions",[])
    except: pass
    return []

def call_feedback(q,a,rating):
    try:
        r = requests.post(f"{API_URL}/feedback",json={"session_id":st.session_state.session_id,"query":q,"answer":a,"rating":rating},headers=_api_headers(),timeout=10)
        return r.ok
    except: return False

@st.cache_data(ttl=30, show_spinner=False)
def _fetch_admin(path):
    r = requests.get(f"{API_URL}{path}", headers=_api_headers(), timeout=20)
    r.raise_for_status()
    return r.json()

# ══════════════════════════════════════════════
# Chat callbacks & render helpers
# ══════════════════════════════════════════════
def _queue_query(q):
    if q: st.session_state.pending_query = q; st.session_state.pending_append_user = True

def _queue_regenerate(idx):
    msgs = st.session_state.messages
    if 0<idx<len(msgs) and msgs[idx]["role"]=="assistant" and msgs[idx-1]["role"]=="user":
        query = msgs[idx-1]["content"]; del msgs[idx]
        st.session_state.pending_query = query; st.session_state.pending_append_user = False

def _record_feedback(idx,rating):
    msgs = st.session_state.messages
    if idx<=0 or idx>=len(msgs): return
    call_feedback(msgs[idx-1]["content"],msgs[idx]["content"],rating)
    st.session_state[f"fb_{st.session_state.conv_id}_{idx}"] = rating

def confidence_class(c):
    if c>=0.75: return "pill-green"
    if c>=0.5: return "pill-amber"
    return "pill-red"

def render_meta(meta):
    if not meta: return
    intent = meta.get("intent","general")
    info = INTENT_META.get(intent,INTENT_META["general"])
    conf = float(meta.get("confidence",0))
    lat = int(meta.get("latency_ms",0))
    pc = confidence_class(conf)
    pills = (f'<span class="pill">{info["emoji"]} {info["label"]}</span>'
             f'<span class="pill {pc}">{int(conf*100)}% confidence</span>'
             f'<span class="pill pill-purple">⚡ {lat} ms</span>')
    st.markdown(f'<div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap;">{pills}</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="meter-wrap"><div class="meter-fill" style="width:{conf*100:.0f}%"></div></div>',unsafe_allow_html=True)

def _similarity_from_l2(l2): return max(0,min(1,1-l2**2/2))

def render_sources(sources):
    if not sources: return
    with st.expander(f"📚 Sources ({len(sources)})",expanded=False):
        for i,src in enumerate(sources[:4]):
            cls = "top" if i==0 else ""
            meta = src.get("metadata",{})
            bits = []
            if src.get("score") is not None: bits.append(f'similarity:{_similarity_from_l2(float(src["score"])):.2f}')
            if meta.get("topic"): bits.append(f'topic:{meta["topic"]}')
            if meta.get("source_query"): bits.append(f'matched q:{meta["source_query"][:60]}')
            line = " · ".join(bits) or "retrieved"
            rank = ["1st","2nd","3rd","4th"][i]
            st.markdown(f'<div class="src-card {cls}"><strong>{rank}</strong> — {src["content"][:220]}{"…" if len(src["content"])>220 else ""}<div class="src-meta">{line}</div></div>',unsafe_allow_html=True)

def build_transcript_md():
    lines = [f"# ChatSolveAI Conversation\n_Session `{st.session_state.session_id}` · exported {datetime.utcnow():%Y-%m-%d %H:%M UTC}_\n"]
    for m in st.session_state.messages:
        who = "**🧑 You**" if m["role"]=="user" else "**🤖 ChatSolveAI**"
        lines.append(f"{who}\n\n{m['content']}\n")
        if m["role"]=="assistant" and m.get("meta"):
            meta = m["meta"]
            lines.append(f"> intent:{meta.get('intent','general')} · confidence:{int(float(meta.get('confidence',0))*100)}% · latency:{meta.get('latency_ms',0)} ms\n")
    return "\n".join(lines)

def submit_query(query, append_user=True):
    if not query or not str(query).strip(): return False
    if append_user:
        st.session_state.messages.append({"role":"user","content":query})
    with st.chat_message("user",avatar=USER_AVATAR):
        st.markdown(query)
    with st.chat_message("assistant",avatar=ASSISTANT_AVATAR):
        box = st.empty()
        result = call_chat_stream(query,box) if USE_STREAMING else call_chat(query)
        if not USE_STREAMING and result: box.markdown(result.get("answer",""))
    if not result:
        if append_user and st.session_state.messages and st.session_state.messages[-1]["role"]=="user" and st.session_state.messages[-1]["content"]==query:
            st.session_state.messages.pop()
        return False
    answer = result["answer"]
    sources = result.get("source_documents",[])
    meta = {
        "intent":result.get("intent","general"),"confidence":result.get("confidence",0),
        "latency_ms":result.get("latency_ms",0),"condensed_query":result.get("condensed_query",query)
    }
    # Follow-up suggestions removed from UI — skip the /suggest call to save
    # an LLM round-trip per turn.
    st.session_state.messages.append({
        "role":"assistant","content":answer,"sources":sources,"meta":meta,
    })
    st.session_state.last_sources = sources
    st.session_state.last_meta = meta
    return True

def _perform_full_reset():
    try: st.query_params.clear()
    except: pass
    api_health.clear()
    st.session_state["session_id"] = str(uuid.uuid4())
    st.session_state["conv_id"] = str(uuid.uuid4())[:8]
    st.session_state["messages"] = []
    st.session_state["last_sources"] = []
    st.session_state["last_meta"] = {}
    st.session_state["pending_query"] = None
    st.session_state["pending_append_user"] = True
    st.session_state["history_loaded_for"] = None
    st.session_state.pop("followups",None)
    st.session_state.pop("selected_topic",None)
    _sync_session_url()
    for k in list(st.session_state.keys()):
        if isinstance(k,str) and k.startswith(("fb_","up_","down_","fu_","chip_","topic_")):
            del st.session_state[k]

# ══════════════════════════════════════════════
# Sidebar — view selector + per-view extras
# ══════════════════════════════════════════════
NAV_CHAT, NAV_ADMIN = "💬 Chat", "📊 Admin Dashboard"

with st.sidebar:
    view = st.radio("View", [NAV_CHAT, NAV_ADMIN], key="nav_view", label_visibility="collapsed")
    st.divider()

# ══════════════════════════════════════════════
# Chat view
# ══════════════════════════════════════════════
def render_chat(sidebar_slot, main_slot):
    _init_state()
    _adopt_url_session()
    _sync_session_url()

    with sidebar_slot:
        st.image("https://img.icons8.com/fluency/96/chatbot.png", width=64)
        st.title("ChatSolveAI")
        st.caption("LangChain · FAISS · GPT‑3.5‑turbo\nMongoDB · FastAPI · Docker · HF Spaces")
        st.divider()
        healthy = api_health()
        if healthy:
            st.success("API connected", icon="✅")
        else:
            st.error(f"API unreachable at {API_URL}", icon="🔴")
        st.caption(f"Session: `{st.session_state.session_id[:8]}…`")
        st.divider()
        if st.button("🗑 New chat", key="btn_new_chat", use_container_width=True):
            _perform_full_reset()
            st.rerun()
        if st.session_state.messages:
            st.download_button("⬇️ Export chat (.md)", data=build_transcript_md(),
                               file_name=f"chatsolveai_{st.session_state.session_id[:8]}.md",
                               mime="text/markdown", use_container_width=True)

    with main_slot:
        st.markdown('<div class="hero-title">💬 ChatSolveAI — Customer Support</div>', unsafe_allow_html=True)
        st.markdown('<p class="hero-sub">LangChain RAG · GPT‑3.5‑turbo · MongoDB · FastAPI …</p>', unsafe_allow_html=True)

        if st.session_state.pending_query:
            if healthy:
                q = st.session_state.pending_query
                append = st.session_state.get("pending_append_user", True)
                st.session_state.pending_query = None
                st.session_state.pending_append_user = True
                if submit_query(q, append_user=append):
                    st.rerun()
            else:
                st.warning("Backend cold‑starting … try again in ~30s.")
                st.session_state.pending_query = None

        # Dedicated placeholder for the start-page area (cards or drill-down).
        # We always instantiate this slot at the same script position, then
        # explicitly call .empty() on it BEFORE writing the new content. This
        # is the only reliable way I've found to make Streamlit drop the
        # previous run's buttons when the conditional branch changes
        # (top-level ↔ drill-down ↔ chat-history).
        startpage_slot = st.empty()
        if not st.session_state.messages:
            conv = st.session_state.conv_id
            selected_topic = st.session_state.get("selected_topic")
            # Wipe the slot first; container() below repopulates it.
            startpage_slot.empty()

            if selected_topic is None:
                # Top-level: show 4 topical category cards.
                with startpage_slot.container():
                    st.markdown("**👋 What do you need help with?**")
                    cols = st.columns(2)
                    for i, (emoji, name, _qs) in enumerate(TOPIC_CATEGORIES):
                        with cols[i % 2]:
                            st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
                            if st.button(f"{emoji}   {name}", key=f"topic_{conv}_{i}",
                                         use_container_width=True):
                                st.session_state["selected_topic"] = i
                                st.rerun()
                            st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Drill-down: show 3 questions for selected category + back button.
                emoji, name, questions = TOPIC_CATEGORIES[selected_topic]
                with startpage_slot.container():
                    head_cols = st.columns([1, 6])
                    with head_cols[0]:
                        if st.button("← Back", key=f"topic_back_{conv}"):
                            st.session_state.pop("selected_topic", None)
                            st.rerun()
                    with head_cols[1]:
                        st.markdown(f"**{emoji} {name}** — pick a question:")
                    for i, q in enumerate(questions):
                        st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
                        st.button(q, key=f"chip_{conv}_{selected_topic}_{i}",
                                  use_container_width=True,
                                  on_click=_queue_query, args=(q,))
                        st.markdown('</div>', unsafe_allow_html=True)
        else:
            startpage_slot.empty()  # No start-page on chat-history view.
            msgs = st.session_state.messages
            conv_id = st.session_state.conv_id
            with st.container(height=520):
                for idx, msg in enumerate(msgs):
                    avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
                    with st.chat_message(msg["role"], avatar=avatar):
                        st.markdown(msg["content"])
                        if msg["role"] == "assistant":
                            render_meta(msg.get("meta", {}))
                            render_sources(msg.get("sources", []))
                            fb = f"fb_{conv_id}_{idx}"
                            if st.session_state.get(fb) is None:
                                c1, c2, _ = st.columns([1, 1, 8])
                                with c1:
                                    if st.button("👍", key=f"up_{conv_id}_{idx}"):
                                        _record_feedback(idx, "up"); st.rerun()
                                with c2:
                                    if st.button("👎", key=f"down_{conv_id}_{idx}"):
                                        _record_feedback(idx, "down"); st.rerun()
                            else:
                                rating = st.session_state[fb]
                                st.caption(f"You rated: {'👍' if rating == 'up' else '👎'}")
                                if rating == "down" and st.button("Regenerate", key=f"regen_{conv_id}_{idx}"):
                                    _queue_regenerate(idx); del st.session_state[fb]; st.rerun()

        if prompt := st.chat_input("Ask about orders, billing, account, or technical issues…"):
            if not healthy:
                st.error("API unreachable.")
                st.stop()
            _queue_query(prompt)
            st.rerun()

# ══════════════════════════════════════════════
# Admin view
# ══════════════════════════════════════════════
def render_admin(sidebar_slot, main_slot):
    # Sidebar additions (only Sign-out when password gate is enabled).
    if ADMIN_PASSWORD and st.session_state.get("admin_ok"):
        with sidebar_slot:
            if st.button("Sign out", key="admin_signout"):
                st.session_state.pop("admin_ok", None)
                st.rerun()

    with main_slot:
        # Optional password gate
        if ADMIN_PASSWORD and not st.session_state.get("admin_ok"):
            with st.form("admin_login"):
                pw = st.text_input("Admin password", type="password")
                if st.form_submit_button("Sign in") and pw == ADMIN_PASSWORD:
                    st.session_state["admin_ok"] = True
                    st.rerun()
                if pw and pw != ADMIN_PASSWORD:
                    st.error("Invalid password.")
            return

        try:
            summary    = _fetch_admin("/analytics")
            timeseries = _fetch_admin("/analytics/timeseries?days=14")
            intents    = _fetch_admin("/analytics/intents")
            latency    = _fetch_admin("/analytics/latency")
            feedback   = _fetch_admin("/analytics/feedback")
            sessions   = _fetch_admin("/sessions?limit=20")
        except Exception:
            st.warning("Backend unreachable – analytics unavailable.")
            return

        st.title("ChatSolveAI Admin Dashboard")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sessions", summary.get("total_sessions", 0))
        c2.metric("Queries", summary.get("total_queries", 0))
        c3.metric("Today", summary.get("queries_today", 0))
        c4.metric("Avg Session Length", summary.get("avg_session_length", 0))
        f1, f2, f3 = st.columns(3)
        f1.metric("P95 Latency", f"{latency.get('p95', 0)} ms")
        f2.metric("👍 Upvotes", feedback.get("up", 0))
        f3.metric("👎 Downvotes", feedback.get("down", 0))
        st.divider()
        left, right = st.columns(2)
        with left:
            st.subheader("Daily Queries")
            if timeseries: st.bar_chart(timeseries, x="date", y="count")
            else: st.info("No data yet.")
        with right:
            st.subheader("Intent Distribution")
            if intents: st.bar_chart(intents, x="intent", y="count")
            else: st.info("No data yet.")
        st.divider()
        tab1, tab2 = st.tabs(["Top Questions", "Recent Sessions"])
        with tab1:
            top = summary.get("top_questions", [])
            if top: st.dataframe(top, use_container_width=True, hide_index=True)
            else: st.info("No questions logged.")
        with tab2:
            if sessions: st.dataframe(sessions, use_container_width=True, hide_index=True)
            else: st.info("No sessions.")
        st.caption(f"Last refreshed: {datetime.utcnow():%Y-%m-%d %H:%M UTC}")

# ══════════════════════════════════════════════
# Dispatch
# ──────────────────────────────────────────────
# Each view writes into its own st.empty() placeholder. The inactive
# placeholder is explicitly emptied so no stale DOM remains.
# st.chat_input docks to the page body (outside any container), so we
# hide it via CSS when on the admin view.
# ══════════════════════════════════════════════
if view == NAV_ADMIN:
    st.markdown(
        "<style>[data-testid='stChatInput']{display:none !important;}</style>",
        unsafe_allow_html=True,
    )

# st.empty() placeholders (NOT keyed containers): each rerun, the inactive
# placeholder is explicitly emptied and the active one is filled. This
# replaces ALL prior content in the slot rather than relying on Streamlit's
# key-based reconciliation (which kept stale drill-down buttons alive after
# "New chat" reset).
chat_sidebar_slot  = st.sidebar.empty()
admin_sidebar_slot = st.sidebar.empty()
chat_main_slot     = st.empty()
admin_main_slot    = st.empty()

if view == NAV_CHAT:
    # Clear the inactive view's placeholders explicitly.
    admin_sidebar_slot.empty()
    admin_main_slot.empty()
    render_chat(chat_sidebar_slot.container(), chat_main_slot.container())
else:
    chat_sidebar_slot.empty()
    chat_main_slot.empty()
    render_admin(admin_sidebar_slot.container(), admin_main_slot.container())
