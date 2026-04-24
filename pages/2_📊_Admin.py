"""
ChatSolveAI — Admin dashboard.

Password-gated read-only view of usage, latency, intent mix, feedback.
All data is read from the FastAPI backend (no direct Mongo access).
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
_SECRET_API_URL = None
_ADMIN_PASSWORD = None
try:
    _SECRET_API_URL = st.secrets.get("API_URL")
    _ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD")
except Exception:
    pass

API_URL        = (_SECRET_API_URL or os.getenv("API_URL") or "https://Nikollass-chatsolveai-api.hf.space").rstrip("/")
ADMIN_PASSWORD = _ADMIN_PASSWORD or os.getenv("ADMIN_PASSWORD", "admin")

INTENT_COLORS = {
    "billing":   "#FF9800",
    "account":   "#2196F3",
    "shipping":  "#4CAF50",
    "technical": "#9C27B0",
    "general":   "#607D8B",
}

st.set_page_config(page_title="ChatSolveAI Admin", page_icon="📊", layout="wide")


# ── Password gate ─────────────────────────────────────────────────────────────

if "admin_authed" not in st.session_state:
    st.session_state.admin_authed = False

if not st.session_state.admin_authed:
    st.markdown("## 🔒 Admin access")
    st.caption("Enter the admin password to view dashboards.")
    pw = st.text_input("Password", type="password")
    if st.button("Unlock", type="primary"):
        if pw == ADMIN_PASSWORD:
            st.session_state.admin_authed = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()


# ── Data fetchers ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def _get_json(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_URL}{path}", params=params or {}, timeout=15)
        if r.ok:
            return r.json()
    except requests.RequestException:
        pass
    return None


# ── Page ──────────────────────────────────────────────────────────────────────

st.title("📊 ChatSolveAI — Admin dashboard")
st.caption(f"Live from `{API_URL}` · auto-refreshes every 30 s")

col_reset, _ = st.columns([1, 8])
with col_reset:
    if st.button("🔄 Refresh now"):
        _get_json.clear()
        st.rerun()

st.divider()


# ── KPIs row ──────────────────────────────────────────────────────────────────

summary = _get_json("/analytics") or {}
latency = _get_json("/analytics/latency") or {}
fb      = _get_json("/analytics/feedback") or {}

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total sessions", summary.get("total_sessions", 0))
k2.metric("Total queries",  summary.get("total_queries", 0))
k3.metric("Today",          summary.get("queries_today", 0))
k4.metric("Avg latency (ms)", int(latency.get("avg", 0) or 0))
k5.metric("👍 / 👎", f"{fb.get('up', 0)} / {fb.get('down', 0)}")

st.divider()


# ── Time series + intents ─────────────────────────────────────────────────────

left, right = st.columns((2, 1))

with left:
    st.subheader("📈 Queries per day (last 14 days)")
    ts = _get_json("/analytics/timeseries", {"days": 14})
    if ts:
        df = pd.DataFrame(ts)
        df["date"] = pd.to_datetime(df["date"])
        fig = px.area(
            df, x="date", y="count",
            color_discrete_sequence=["#4F8BF9"],
        )
        fig.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="", yaxis_title="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
        )
        fig.update_traces(line=dict(width=2), fillcolor="rgba(79,139,249,0.2)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No time-series data yet.")

with right:
    st.subheader("🎯 Intent mix")
    intents = _get_json("/analytics/intents")
    if intents:
        df = pd.DataFrame(intents)
        fig = px.pie(
            df, names="intent", values="count", hole=0.55,
            color="intent",
            color_discrete_map=INTENT_COLORS,
        )
        fig.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No intent data yet.")


# ── Latency + feedback ratio ──────────────────────────────────────────────────

left, right = st.columns(2)

with left:
    st.subheader("⚡ Latency percentiles (last 500 chat calls)")
    if latency.get("n", 0):
        df = pd.DataFrame({
            "percentile": ["p50", "p95", "avg"],
            "ms":          [latency["p50"], latency["p95"], latency["avg"]],
        })
        fig = px.bar(
            df, x="percentile", y="ms", text="ms",
            color="percentile",
            color_discrete_map={"p50": "#4CAF50", "p95": "#FF9800", "avg": "#4F8BF9"},
        )
        fig.update_traces(texttemplate="%{text:.0f} ms", textposition="outside")
        fig.update_layout(
            height=300, showlegend=False,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            yaxis_title="milliseconds",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Based on {latency['n']} samples")
    else:
        st.info("No latency data yet.")

with right:
    st.subheader("👍 Feedback ratio")
    up   = fb.get("up",   0)
    down = fb.get("down", 0)
    total = up + down
    if total:
        df = pd.DataFrame({"rating": ["👍 helpful", "👎 needs work"], "count": [up, down]})
        fig = px.bar(
            df, x="count", y="rating", orientation="h", text="count",
            color="rating",
            color_discrete_map={"👍 helpful": "#4CAF50", "👎 needs work": "#F44336"},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=300, showlegend=False,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.02)",
            xaxis_title="", yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)
        score = up / total if total else 0
        st.caption(f"Approval rate: **{score*100:.1f}%**")
    else:
        st.info("No feedback yet.")


st.divider()

# ── Top questions + recent sessions ───────────────────────────────────────────

left, right = st.columns(2)

with left:
    st.subheader("🔥 Top questions")
    tops = summary.get("top_questions", [])
    if tops:
        df = pd.DataFrame(tops)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No queries logged yet.")

with right:
    st.subheader("🕑 Recent sessions")
    sess = _get_json("/sessions", {"limit": 10})
    if sess:
        df = pd.DataFrame(sess)
        df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
        df = df[["created_at", "turn_count", "last_message", "session_id"]]
        df.columns = ["Created", "Turns", "Last message", "Session"]
        df["Session"] = df["Session"].str.slice(0, 8) + "…"
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No sessions yet.")

st.divider()
st.caption(f"Rendered {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
