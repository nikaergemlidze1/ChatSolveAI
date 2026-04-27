"""Password-protected Streamlit admin dashboard for ChatSolveAI analytics."""

from __future__ import annotations

import os
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return str(value or os.getenv(name) or default)


API_URL = _secret("API_URL", "http://localhost:8000").rstrip("/")
API_KEY = _secret("API_KEY")
ADMIN_PASSWORD = _secret("ADMIN_PASSWORD")


def _headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


@st.cache_data(ttl=30, show_spinner=False)
def _get_json(path: str):
    response = requests.get(
        f"{API_URL}{path}",
        headers=_headers(),
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def _require_admin() -> bool:
    st.set_page_config(page_title="ChatSolveAI Admin", page_icon="📊", layout="wide")
    st.title("ChatSolveAI Admin")

    if not ADMIN_PASSWORD:
        st.error("Admin dashboard is disabled until ADMIN_PASSWORD is configured.")
        st.stop()

    if st.session_state.get("admin_ok"):
        return True

    with st.form("admin_login"):
        password = st.text_input("Admin password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        if password == ADMIN_PASSWORD:
            st.session_state["admin_ok"] = True
            st.rerun()
        st.error("Invalid password.")

    return False


def _render_metric_row(summary: dict, latency: dict, feedback: dict):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions", summary.get("total_sessions", 0))
    c2.metric("Queries", summary.get("total_queries", 0))
    c3.metric("Queries Today", summary.get("queries_today", 0))
    c4.metric("P95 Latency", f"{latency.get('p95', 0)} ms")

    f1, f2, f3 = st.columns(3)
    f1.metric("Avg Session Length", summary.get("avg_session_length", 0))
    f2.metric("Thumbs Up", feedback.get("up", 0))
    f3.metric("Thumbs Down", feedback.get("down", 0))


def _render_charts(timeseries: list[dict], intents: list[dict]):
    left, right = st.columns(2)
    with left:
        st.subheader("Daily Queries")
        if timeseries:
            st.bar_chart(timeseries, x="date", y="count")
        else:
            st.info("No query timeseries data yet.")

    with right:
        st.subheader("Intent Distribution")
        if intents:
            st.bar_chart(intents, x="intent", y="count")
        else:
            st.info("No intent data yet.")


def _render_tables(summary: dict, sessions: list[dict]):
    left, right = st.columns(2)
    with left:
        st.subheader("Top Questions")
        top_questions = summary.get("top_questions", [])
        if top_questions:
            st.dataframe(top_questions, use_container_width=True, hide_index=True)
        else:
            st.info("No question logs yet.")

    with right:
        st.subheader("Recent Sessions")
        if sessions:
            st.dataframe(sessions, use_container_width=True, hide_index=True)
        else:
            st.info("No sessions yet.")


def main():
    if not _require_admin():
        return

    top = st.columns([1, 1, 4])
    with top[0]:
        if st.button("Refresh", use_container_width=True):
            _get_json.clear()
            st.rerun()
    with top[1]:
        if st.button("Sign out", use_container_width=True):
            st.session_state.pop("admin_ok", None)
            st.rerun()
    with top[2]:
        st.caption(f"API: `{API_URL}` · refreshed {datetime.utcnow():%Y-%m-%d %H:%M UTC}")

    try:
        summary = _get_json("/analytics")
        timeseries = _get_json("/analytics/timeseries?days=14")
        intents = _get_json("/analytics/intents")
        latency = _get_json("/analytics/latency")
        feedback = _get_json("/analytics/feedback")
        sessions = _get_json("/sessions?limit=20")
    except requests.RequestException as exc:
        st.error(f"Could not load analytics from backend: {exc}")
        st.stop()

    _render_metric_row(summary, latency, feedback)
    st.divider()
    _render_charts(timeseries, intents)
    st.divider()
    _render_tables(summary, sessions)


main()
