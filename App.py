"""
ChatSolveAI — Streamlit frontend (v2.1) — Chat only.
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
_SECRET_API_URL = _SECRET_API_KEY = None
try:
    _SECRET_API_URL = st.secrets.get("API_URL")
    _SECRET_API_KEY = st.secrets.get("API_KEY")
except: pass

API_URL = (_SECRET_API_URL or os.getenv("API_URL") or "https://Nikollass-chatsolveai-api.hf.space").rstrip("/")
API_KEY = _SECRET_API_KEY or os.getenv("API_KEY") or ""
HEALTH_TIMEOUT_S = int(os.getenv("API_HEALTH_TIMEOUT","20"))
HEALTH_RETRIES   = int(os.getenv("API_HEALTH_RETRIES","2"))
USE_STREAMING    = os.getenv("USE_STREAMING","true").lower() not in {"0","false","no"}

def _api_headers(): return {"X-API-Key": API_KEY} if API_KEY else {}

# ══════════════════════════════════════════════
# Page setup (single set_page_config for the whole app)
# ══════════════════════════════════════════════
st.set_page_config(page_title="ChatSolveAI", page_icon="🤖", layout="wide",
                   initial_sidebar_state="expanded")
if "_app_render_id" not in st.session_state:
    st.session_state._app_render_id = 0
# ══════════════════════════════════════════════
# Page isolation guard
# ══════════════════════════════════════════════
if st.session_state.get("_page") == "admin":
    st.session_state._app_render_id += 1
    st.session_state["_clear_admin"] = True
    st.session_state["_page"] = "app"
    st.rerun()
st.session_state["_page"] = "app"

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
# Replace the old admin container with an empty one so its charts disappear.
if st.session_state.get("_clear_admin"):
    st.session_state.pop("_clear_admin")
    if "_admin_render_id" in st.session_state:
        st.container(key=f"admin_main_{st.session_state._admin_render_id}")
# ══════════════════════════════════════════════
# Session state helpers
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

def _init_state():
    url_id = _session_id_from_url()
    for k,v in {"session_id":url_id or str(uuid.uuid4()),"conv_id":str(uuid.uuid4())[:8],
                "messages":[],"last_sources":[],"last_meta":{},"pending_query":None,
                "pending_append_user":True,"history_loaded_for":None}.items():
        if k not in st.session_state: st.session_state[k] = v
    st.session_state.pop("followups",None)
_init_state()
_adopt_url_session()
_sync_session_url()

# ══════════════════════════════════════════════
# API helpers (identical to before)
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

# ══════════════════════════════════════════════
# Callbacks & render helpers
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
    st.session_state.messages.append({
        "role":"assistant","content":answer,"sources":sources,"meta":meta,
        "followups":call_suggest(answer)
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
    _sync_session_url()
    for k in list(st.session_state.keys()):
        if isinstance(k,str) and k.startswith(("fb_","up_","down_","fu_","chip_")):
            del st.session_state[k]

# ══════════════════════════════════════════════
# Sidebar – only on App page
# ══════════════════════════════════════════════
if st.session_state.get("_page") == "app":
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/chatbot.png",width=64)
        st.title("ChatSolveAI")
        st.caption("LangChain · FAISS · GPT‑3.5‑turbo\nMongoDB · FastAPI · Docker · HF Spaces")
        st.divider()
        healthy = api_health()
        if healthy:
            st.success("API connected",icon="✅")
        else:
            st.error(f"API unreachable at {API_URL}",icon="🔴")
        st.caption(f"Session: `{st.session_state.session_id[:8]}…`")
        st.divider()
        if st.button("🗑 New chat",key="btn_new_chat",use_container_width=True):
            _perform_full_reset()
            st.rerun()
        if st.session_state.messages:
            st.download_button("⬇️ Export chat (.md)",data=build_transcript_md(),
                               file_name=f"chatsolveai_{st.session_state.session_id[:8]}.md",
                               mime="text/markdown",use_container_width=True)
else:
    st.sidebar.empty()

# ══════════════════════════════════════════════
# Main chat UI (wrapped in a container)
# ══════════════════════════════════════════════
main_container = st.container(key=f"app_main_{st.session_state._app_render_id}")
with main_container:
    st.markdown('<div class="hero-title">💬 ChatSolveAI — Customer Support</div>',unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">LangChain RAG · GPT‑3.5‑turbo · MongoDB · FastAPI …</p>',unsafe_allow_html=True)

    if st.session_state.pending_query:
        if healthy:
            q = st.session_state.pending_query
            append = st.session_state.get("pending_append_user",True)
            st.session_state.pending_query = None
            st.session_state.pending_append_user = True
            if submit_query(q,append_user=append):
                st.rerun()
        else:
            st.warning("Backend cold‑starting … try again in ~30s.")
            st.session_state.pending_query = None

    if not st.session_state.messages:
        st.markdown("**👋 Try one of these to get started:**")
        cols = st.columns(2)
        conv = st.session_state.conv_id
        for i,(emoji,q) in enumerate(EXAMPLE_QUESTIONS):
            with cols[i%2]:
                st.markdown('<div class="chip-btn">',unsafe_allow_html=True)
                st.button(f"{emoji}   {q}",key=f"chip_{conv}_{i}",use_container_width=True,on_click=_queue_query,args=(q,))
                st.markdown('</div>',unsafe_allow_html=True)
    else:
        msgs = st.session_state.messages
        last_idx = len(msgs)-1
        conv_id = st.session_state.conv_id
        with st.container(height=900):
            for idx,msg in enumerate(msgs):
                avatar = USER_AVATAR if msg["role"]=="user" else ASSISTANT_AVATAR
                with st.chat_message(msg["role"],avatar=avatar):
                    st.markdown(msg["content"])
                    if msg["role"]=="assistant":
                        render_meta(msg.get("meta",{}))
                        render_sources(msg.get("sources",[]))
                        fb = f"fb_{conv_id}_{idx}"
                        if st.session_state.get(fb) is None:
                            c1,c2,_ = st.columns([1,1,8])
                            with c1:
                                if st.button("👍",key=f"up_{conv_id}_{idx}"):
                                    _record_feedback(idx,"up"); st.rerun()
                            with c2:
                                if st.button("👎",key=f"down_{conv_id}_{idx}"):
                                    _record_feedback(idx,"down"); st.rerun()
                        else:
                            rating = st.session_state[fb]
                            st.caption(f"You rated: {'👍' if rating=='up' else '👎'}")
                            if rating=="down" and st.button("Regenerate",key=f"regen_{conv_id}_{idx}"):
                                _queue_regenerate(idx); del st.session_state[fb]; st.rerun()
        with st.container():
            if msgs[-1]["role"]=="assistant":
                fu = msgs[-1].get("followups") or []
                if fu:
                    st.markdown('<div style="margin-top:14px;font-weight:600;color:#c9b8ff;">💡 Suggested follow‑ups</div>',unsafe_allow_html=True)
                    cols = st.columns(len(fu))
                    clicked = None
                    for i,q in enumerate(fu):
                        with cols[i]:
                            st.markdown('<div class="chip-btn">',unsafe_allow_html=True)
                            if st.button(q,key=f"fu_{conv_id}_{last_idx}_{i}",use_container_width=True):
                                clicked = q
                            st.markdown('</div>',unsafe_allow_html=True)
                    if clicked:
                        st.session_state.pending_query = clicked
                        st.rerun()

    if prompt := st.chat_input("Ask about orders, billing, account, or technical issues…"):
        if not healthy:
            st.error("API unreachable.")
            st.stop()
        _queue_query(prompt)
        st.rerun()