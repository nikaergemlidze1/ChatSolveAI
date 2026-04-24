"""Validate the Pydantic API models (shape + defaults)."""

from datetime import datetime, timezone

import pytest

from api.models import (
    ChatRequest, ChatResponse, SourceDocument,
    FeedbackRequest, SuggestRequest, SuggestResponse,
    TimeseriesPoint, IntentBucket, LatencyStats, SessionSummary,
)


def test_chat_request_accepts_valid():
    req = ChatRequest(session_id="abc", query="hello")
    assert req.query == "hello"


def test_chat_request_rejects_empty():
    with pytest.raises(Exception):
        ChatRequest(session_id="abc", query="")


def test_chat_response_defaults():
    resp = ChatResponse(
        session_id="abc", query="x", answer="y",
        timestamp=datetime.now(timezone.utc),
    )
    assert resp.confidence == 0.0
    assert resp.intent == "general"
    assert resp.latency_ms == 0
    assert resp.source_documents == []


def test_feedback_rating_enum():
    FeedbackRequest(session_id="s", query="q", answer="a", rating="up")
    FeedbackRequest(session_id="s", query="q", answer="a", rating="down")
    with pytest.raises(Exception):
        FeedbackRequest(session_id="s", query="q", answer="a", rating="maybe")


def test_suggest_bounds():
    SuggestRequest(last_answer="hi", n=3)
    with pytest.raises(Exception):
        SuggestRequest(last_answer="hi", n=10)


def test_source_document_optional_score():
    sd = SourceDocument(content="x")
    assert sd.score is None


def test_timeseries_point():
    TimeseriesPoint(date="2026-04-24", count=5)


def test_latency_stats_zeros():
    s = LatencyStats(p50=0, p95=0, avg=0, n=0)
    assert s.n == 0
