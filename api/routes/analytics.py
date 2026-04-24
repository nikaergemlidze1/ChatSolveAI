"""
Analytics routes — read-only aggregations from MongoDB.

GET /analytics              → aggregate stats across all sessions
GET /analytics/timeseries   → daily query count for the last N days
GET /analytics/intents      → intent distribution
GET /analytics/latency      → p50 / p95 / avg chat latency
GET /analytics/feedback     → 👍 / 👎 counts
GET /sessions               → recent sessions list
GET /history/{session_id}   → full message history for one session
GET /health                 → liveness check
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.models import (
    AnalyticsSummary, SessionHistory, MessageRecord,
    TimeseriesPoint, IntentBucket, LatencyStats, SessionSummary,
)
from api import database as db

router = APIRouter(tags=["analytics"])


@router.get("/health", summary="Liveness check")
async def health():
    return {"status": "ok"}


@router.get("/analytics", response_model=AnalyticsSummary)
async def analytics():
    return AnalyticsSummary(
        total_sessions=await db.total_sessions(),
        total_queries=await db.total_queries(),
        queries_today=await db.queries_today(),
        avg_session_length=await db.avg_session_length(),
        top_questions=await db.top_questions(limit=10),
    )


@router.get("/analytics/timeseries", response_model=list[TimeseriesPoint])
async def analytics_timeseries(days: int = Query(14, ge=1, le=90)):
    return [TimeseriesPoint(**p) for p in await db.queries_timeseries(days=days)]


@router.get("/analytics/intents", response_model=list[IntentBucket])
async def analytics_intents():
    return [IntentBucket(**b) for b in await db.intent_distribution()]


@router.get("/analytics/latency", response_model=LatencyStats)
async def analytics_latency():
    return LatencyStats(**await db.latency_stats(path_prefix="/chat"))


@router.get("/analytics/feedback")
async def analytics_feedback():
    return await db.feedback_counts()


@router.get("/sessions", response_model=list[SessionSummary])
async def list_sessions(limit: int = Query(10, ge=1, le=50)):
    return [SessionSummary(**s) for s in await db.recent_sessions(limit=limit)]


@router.get("/history/{session_id}", response_model=SessionHistory)
async def session_history(session_id: str):
    doc = await db.get_session(session_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    return SessionHistory(
        session_id=session_id,
        created_at=doc["created_at"],
        messages=[
            MessageRecord(
                role=m["role"],
                content=m["content"],
                timestamp=m["timestamp"],
            )
            for m in doc.get("messages", [])
        ],
    )
