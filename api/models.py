"""Pydantic request / response models for the ChatSolveAI API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique session UUID")
    query:      str = Field(..., min_length=1, max_length=2000)


class SourceDocument(BaseModel):
    content:  str
    metadata: dict[str, Any] = {}
    score:    float | None = None


class ChatResponse(BaseModel):
    session_id:       str
    query:            str
    answer:           str
    source_documents: list[SourceDocument] = []
    confidence:       float = 0.0
    condensed_query:  str   = ""
    intent:           str   = "general"
    latency_ms:       int   = 0
    timestamp:        datetime


# ── Session / History ─────────────────────────────────────────────────────────

class MessageRecord(BaseModel):
    role:      str   # "user" | "assistant"
    content:   str
    timestamp: datetime


class SessionHistory(BaseModel):
    session_id: str
    messages:   list[MessageRecord]
    created_at: datetime


class SessionSummary(BaseModel):
    session_id:   str
    created_at:   datetime
    turn_count:   int
    last_message: str


# ── Analytics ─────────────────────────────────────────────────────────────────

class AnalyticsSummary(BaseModel):
    total_sessions:     int
    total_queries:      int
    queries_today:      int
    avg_session_length: float
    top_questions:      list[dict[str, Any]]


class TimeseriesPoint(BaseModel):
    date:  str
    count: int


class IntentBucket(BaseModel):
    intent: str
    count:  int


class LatencyStats(BaseModel):
    p50: float
    p95: float
    avg: float
    n:   int


# ── Feedback ──────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    session_id: str
    query:      str
    answer:     str
    rating:     Literal["up", "down"]
    note:       str | None = None


class FeedbackResponse(BaseModel):
    ok: bool = True


# ── Suggest follow-ups ────────────────────────────────────────────────────────

class SuggestRequest(BaseModel):
    last_answer: str = Field(..., min_length=1, max_length=4000)
    n:           int = Field(3, ge=1, le=5)


class SuggestResponse(BaseModel):
    suggestions: list[str]
