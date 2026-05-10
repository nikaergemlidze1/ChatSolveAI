"""Admin Dashboard — separate Streamlit page.

Each Streamlit page runs as its own script execution, so navigating
between Chat and Admin Dashboard fully unmounts the prior page's
DOM (chart iframes, Lottie players, chat history) without any of
the JS-cleanup gymnastics the single-script dispatch required.
"""
import os
from datetime import datetime

import requests
import streamlit as st
import altair as alt
from dotenv import load_dotenv

load_dotenv()

# ── Config (mirrors App.py; kept local so this page can run independently
# of the chat module).
_SECRET_API_URL = _SECRET_API_KEY = _SECRET_ADMIN_PW = None
try:
    _SECRET_API_URL = st.secrets.get("API_URL")
    _SECRET_API_KEY = st.secrets.get("API_KEY")
    _SECRET_ADMIN_PW = st.secrets.get("ADMIN_PASSWORD")
except Exception:
    pass

API_URL = (_SECRET_API_URL or os.getenv("API_URL") or "https://Nikollass-chatsolveai-api.hf.space").rstrip("/")
API_KEY = _SECRET_API_KEY or os.getenv("API_KEY") or ""
ADMIN_PASSWORD = _SECRET_ADMIN_PW or os.getenv("ADMIN_PASSWORD") or ""


def _api_headers():
    return {"X-API-Key": API_KEY} if API_KEY else {}


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_admin(path):
    r = requests.get(f"{API_URL}{path}", headers=_api_headers(), timeout=20)
    r.raise_for_status()
    return r.json()


# ── Page setup
st.set_page_config(
    page_title="Customer Support AI · Admin",
    page_icon="logo/Logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Lexend:wght@600;700&display=swap" rel="stylesheet">
    <style>
    :root {--accent:#4F8BF9;--accent-2:#8E6BFF;--font-ui:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,system-ui,sans-serif;--font-display:'Lexend','Inter',-apple-system,BlinkMacSystemFont,system-ui,sans-serif;}
    html,body,[data-testid='stApp'],[data-testid='stSidebar']{font-family:var(--font-ui)!important}
    h1,h2,h3{font-family:var(--font-display)!important;letter-spacing:-.01em}
    [data-testid='StyledFullScreenButton'],[data-testid='stFullScreenButton'],[data-testid='stElementToolbar'],[data-testid='stHeaderActionElements']{display:none!important}
    .vega-actions,.vega-embed details,.vega-embed summary,.vega-bindings{display:none!important}
    [data-testid='stSidebar']{background:rgba(20,24,32,.65)!important;backdrop-filter:blur(14px) saturate(140%);-webkit-backdrop-filter:blur(14px) saturate(140%);border-right:1px solid rgba(255,255,255,.06)}
    .nav-links{display:flex;flex-direction:column;gap:4px;margin:8px 0 4px}
    .nav-links .nav-link{display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:10px;color:#cdd5e0!important;font-weight:500;font-size:.92rem;text-decoration:none!important;border:1px solid transparent;transition:all .2s ease}
    .nav-links .nav-link:hover{background:rgba(79,139,249,.12)!important;color:#fff!important;border-color:rgba(79,139,249,.25)}
    .nav-links .nav-link--active{background:rgba(79,139,249,.18)!important;color:#fff!important;border-color:rgba(79,139,249,.40);box-shadow:inset 3px 0 0 #4F8BF9}
    .st-key-admin_grid [data-testid='stMetric']{background:rgba(20,24,32,.55);border:1px solid rgba(255,255,255,.06);border-radius:14px;padding:16px 20px;backdrop-filter:blur(8px) saturate(140%);transition:transform .25s ease,box-shadow .25s ease}
    .st-key-admin_grid [data-testid='stMetric']:hover{transform:translateY(-2px);box-shadow:0 6px 24px rgba(79,139,249,.12);border-color:rgba(79,139,249,.25)}
    .st-key-admin_grid [data-testid='stMetricLabel']{color:#7a8190!important;font-size:.7rem!important;letter-spacing:.08em;text-transform:uppercase;font-weight:600!important}
    .st-key-admin_grid [data-testid='stMetricValue']{font-family:var(--font-display)!important;font-size:1.85rem!important;font-weight:600!important;color:#E5E7EB!important;text-shadow:0 0 14px rgba(79,139,249,.22)}
    .st-key-admin_grid h3{font-family:var(--font-display)!important;font-weight:600;letter-spacing:.01em;color:#E5E7EB;margin:.4rem 0 1rem!important;font-size:1.1rem}
    .st-key-admin_grid h1{font-size:1.85rem!important;margin-bottom:1.2rem!important}
    .st-key-admin_grid hr{margin:1.6rem 0!important;opacity:.4}
    .st-key-admin_grid [data-testid='stHorizontalBlock']{gap:1rem}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar (Sign out only when password gate enabled)
with st.sidebar:
    st.image("logo/Logo.png", width=256)
    st.title("Customer Support AI")
    st.divider()
    if ADMIN_PASSWORD and st.session_state.get("admin_ok"):
        if st.button("Sign out", key="admin_signout"):
            st.session_state.pop("admin_ok", None)
            st.rerun()


# ── Optional password gate
if ADMIN_PASSWORD and not st.session_state.get("admin_ok"):
    with st.form("admin_login"):
        pw = st.text_input("Admin password", type="password")
        if st.form_submit_button("Sign in") and pw == ADMIN_PASSWORD:
            st.session_state["admin_ok"] = True
            st.rerun()
        if pw and pw != ADMIN_PASSWORD:
            st.error("Invalid password.")
    st.stop()


# ── Fetch analytics
try:
    summary    = _fetch_admin("/analytics")
    timeseries = _fetch_admin("/analytics/timeseries?days=14")
    intents    = _fetch_admin("/analytics/intents")
    latency    = _fetch_admin("/analytics/latency")
    feedback   = _fetch_admin("/analytics/feedback")
    sessions   = _fetch_admin("/sessions?limit=20")
except Exception:
    st.warning("Backend unreachable – analytics unavailable.")
    st.stop()


# ── Render dashboard
with st.container(key="admin_grid"):
    st.title("Customer Support AI · Admin Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions", summary.get("total_sessions", 0))
    c2.metric("Queries", summary.get("total_queries", 0))
    c3.metric("Today", summary.get("queries_today", 0))
    c4.metric("Avg Session", summary.get("avg_session_length", 0))

    f1, f2, f3 = st.columns(3)
    f1.metric("P95 Latency", f"{latency.get('p95', 0)} ms")
    f2.metric("👍 Upvotes", feedback.get("up", 0))
    f3.metric("👎 Downvotes", feedback.get("down", 0))

    st.divider()

    left, right = st.columns([65, 35], gap="large")
    with left:
        st.subheader("Support Volume")
        if timeseries and any(r.get("count", 0) > 0 for r in timeseries):
            line = (
                alt.Chart(alt.Data(values=timeseries))
                .mark_line(
                    point=alt.OverlayMarkDef(filled=True, size=55, color="#4F8BF9"),
                    color="#4F8BF9",
                    strokeWidth=2.5,
                    interpolate="monotone",
                )
                .encode(
                    x=alt.X("date:T", title=None, axis=alt.Axis(labelColor="#7a8190", grid=False)),
                    y=alt.Y("count:Q", title=None, axis=alt.Axis(labelColor="#7a8190", grid=True, gridOpacity=0.15)),
                    tooltip=["date:T", "count:Q"],
                )
                .properties(height=280)
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(line, use_container_width=True)
        else:
            st.info("No data yet.")
    with right:
        st.subheader("Category Breakdown")
        if intents and any(r.get("count", 0) > 0 for r in intents):
            donut = (
                alt.Chart(alt.Data(values=intents))
                .mark_arc(innerRadius=50, outerRadius=100, stroke="#0f1218", strokeWidth=2)
                .encode(
                    theta=alt.Theta("count:Q"),
                    color=alt.Color(
                        "intent:N",
                        legend=alt.Legend(title=None, orient="bottom", labelFontSize=11, columns=2, labelColor="#cdd5e0"),
                        scale=alt.Scale(scheme="tableau10"),
                    ),
                    tooltip=["intent:N", "count:Q"],
                )
                .properties(height=280)
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(donut, use_container_width=True)
        else:
            st.info("No data yet.")

    st.divider()

    tab1, tab2 = st.tabs(["Top Questions", "Recent Sessions"])
    with tab1:
        top = summary.get("top_questions", [])
        if top:
            st.dataframe(top, use_container_width=True, hide_index=True, height=260)
        else:
            st.info("No questions logged.")
    with tab2:
        if sessions:
            st.dataframe(sessions, use_container_width=True, hide_index=True, height=260)
        else:
            st.info("No sessions.")
    st.caption(f"Last refreshed: {datetime.utcnow():%Y-%m-%d %H:%M UTC}")
