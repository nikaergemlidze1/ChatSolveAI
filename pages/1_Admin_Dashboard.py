import streamlit as st
import requests, os
from datetime import datetime

st.set_page_config(page_title="ChatSolveAI Admin", page_icon="📊", layout="wide")

# Clear any chat state that may have leaked
for key in ("messages", "pending_query", "last_sources", "last_meta", "followups"):
    st.session_state.pop(key, None)

# ── Password gate (optional) ─────────────────────────────────
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD") or st.secrets.get("ADMIN_PASSWORD", "")
if ADMIN_PASSWORD:
    if not st.session_state.get("admin_ok"):
        with st.form("admin_login"):
            pw = st.text_input("Admin password", type="password")
            if st.form_submit_button("Sign in") and pw == ADMIN_PASSWORD:
                st.session_state["admin_ok"] = True
                st.rerun()
            if pw and pw != ADMIN_PASSWORD:
                st.error("Invalid password.")
        st.stop()
    col1, col2 = st.columns([6,1])
    with col2:
        if st.button("Sign out"):
            st.session_state.pop("admin_ok")
            st.rerun()

# ── Backend connection ───────────────────────────────────────
API_URL = os.getenv("API_URL", "https://Nikollass-chatsolveai-api.hf.space").rstrip("/")
API_KEY = os.getenv("API_KEY") or st.secrets.get("API_KEY", "")
headers = {"X-API-Key": API_KEY} if API_KEY else {}

@st.cache_data(ttl=30, show_spinner=False)
def _fetch(path):
    r = requests.get(f"{API_URL}{path}", headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()

try:
    summary    = _fetch("/analytics")
    timeseries = _fetch("/analytics/timeseries?days=14")
    intents    = _fetch("/analytics/intents")
    latency    = _fetch("/analytics/latency")
    feedback   = _fetch("/analytics/feedback")
    sessions   = _fetch("/sessions?limit=20")
except Exception:
    st.warning("Backend unreachable – analytics unavailable.")
    st.stop()

# ── Dashboard UI ──────────────────────────────────────────────
st.title("ChatSolveAI Admin Dashboard")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Sessions", summary.get("total_sessions",0))
c2.metric("Queries", summary.get("total_queries",0))
c3.metric("Today", summary.get("queries_today",0))
c4.metric("Avg Session Length", summary.get("avg_session_length",0))
f1,f2,f3 = st.columns(3)
f1.metric("P95 Latency", f"{latency.get('p95',0)} ms")
f2.metric("👍 Upvotes", feedback.get("up",0))
f3.metric("👎 Downvotes", feedback.get("down",0))
st.divider()
left,right = st.columns(2)
with left:
    st.subheader("Daily Queries")
    if timeseries: st.bar_chart(timeseries, x="date", y="count")
    else: st.info("No data yet.")
with right:
    st.subheader("Intent Distribution")
    if intents: st.bar_chart(intents, x="intent", y="count")
    else: st.info("No data yet.")
st.divider()
tab1,tab2 = st.tabs(["Top Questions","Recent Sessions"])
with tab1:
    top = summary.get("top_questions",[])
    if top: st.dataframe(top, use_container_width=True, hide_index=True)
    else: st.info("No questions logged.")
with tab2:
    if sessions: st.dataframe(sessions, use_container_width=True, hide_index=True)
    else: st.info("No sessions.")
st.caption(f"Last refreshed: {datetime.utcnow():%Y-%m-%d %H:%M UTC}")