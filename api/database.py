"""
MongoDB connection and collection helpers (async via motor).

Collections
-----------
sessions    : one document per chat session
  {
    _id          : session_id (str),
    created_at   : datetime,
    messages     : [
      { role, content, timestamp }
    ]
  }

query_logs  : one document per chat turn (for analytics)
  {
    session_id : str,
    query      : str,
    answer     : str,
    sources    : list[str],
    timestamp  : datetime,
  }
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import motor.motor_asyncio
from pymongo import DESCENDING

# ── Connection ────────────────────────────────────────────────────────────────
MONGO_URL = os.getenv("MONGO_URL") or os.getenv("MONGODB_URI") or "mongodb://localhost:27017"
DB_NAME   = "chatsolveai"

_client: motor.motor_asyncio.AsyncIOMotorClient | None = None


def get_client() -> motor.motor_asyncio.AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = motor.motor_asyncio.AsyncIOMotorClient(
            MONGO_URL,
            serverSelectionTimeoutMS=5_000,
        )
    return _client


def get_db():
    return get_client()[DB_NAME]


# ── Shorthand collection accessors ────────────────────────────────────────────

def sessions_col():
    return get_db()["sessions"]


def logs_col():
    return get_db()["query_logs"]


def feedback_col():
    return get_db()["feedback"]


def latency_col():
    return get_db()["latency"]


# ── Session helpers ───────────────────────────────────────────────────────────

async def ensure_session(session_id: str) -> None:
    """Create a session document if it does not already exist."""
    col = sessions_col()
    await col.update_one(
        {"_id": session_id},
        {"$setOnInsert": {"_id": session_id, "created_at": _now(), "messages": []}},
        upsert=True,
    )


async def append_message(session_id: str, role: str, content: str) -> None:
    """Push a message onto the session's messages array."""
    col = sessions_col()
    await col.update_one(
        {"_id": session_id},
        {"$push": {"messages": {"role": role, "content": content, "timestamp": _now()}}},
    )


async def get_session(session_id: str) -> dict | None:
    """Return the raw session document or None."""
    return await sessions_col().find_one({"_id": session_id})


async def delete_session(session_id: str) -> int:
    """Delete a session and its logs. Returns deleted count."""
    await sessions_col().delete_one({"_id": session_id})
    result = await logs_col().delete_many({"session_id": session_id})
    return result.deleted_count


# ── Log helpers ───────────────────────────────────────────────────────────────

async def log_query(
    session_id: str,
    query:      str,
    answer:     str,
    sources:    list[str],
    intent:     str   = "general",
    confidence: float = 0.0,
) -> None:
    """Insert a single query-answer log entry."""
    await logs_col().insert_one(
        {
            "session_id": session_id,
            "query":      query,
            "answer":     answer,
            "sources":    sources,
            "intent":     intent,
            "confidence": confidence,
            "timestamp":  _now(),
        }
    )


# ── Feedback ──────────────────────────────────────────────────────────────────

async def log_feedback(
    session_id: str,
    query:      str,
    answer:     str,
    rating:     str,
    note:       str | None = None,
) -> None:
    await feedback_col().insert_one(
        {
            "session_id": session_id,
            "query":      query,
            "answer":     answer,
            "rating":     rating,
            "note":       note,
            "timestamp":  _now(),
        }
    )


async def feedback_counts() -> dict[str, int]:
    pipeline = [{"$group": {"_id": "$rating", "count": {"$sum": 1}}}]
    out = {"up": 0, "down": 0}
    async for doc in feedback_col().aggregate(pipeline):
        if doc["_id"] in out:
            out[doc["_id"]] = doc["count"]
    return out


# ── Latency ───────────────────────────────────────────────────────────────────

async def log_latency(path: str, method: str, ms: float, status: int) -> None:
    await latency_col().insert_one(
        {
            "path":      path,
            "method":    method,
            "ms":        ms,
            "status":    status,
            "timestamp": _now(),
        }
    )


async def latency_stats(path_prefix: str = "/chat") -> dict:
    """Aggregate p50, p95, avg over the last 500 samples for *path_prefix*."""
    import statistics
    samples: list[float] = []
    cursor = (
        latency_col()
        .find({"path": {"$regex": f"^{path_prefix}"}, "status": {"$lt": 500}})
        .sort("timestamp", DESCENDING)
        .limit(500)
    )
    async for doc in cursor:
        samples.append(float(doc["ms"]))
    if not samples:
        return {"p50": 0.0, "p95": 0.0, "avg": 0.0, "n": 0}
    samples.sort()
    n = len(samples)
    p50 = samples[n // 2]
    p95 = samples[min(n - 1, int(n * 0.95))]
    avg = sum(samples) / n
    return {"p50": round(p50, 1), "p95": round(p95, 1), "avg": round(avg, 1), "n": n}


# ── Analytics helpers ─────────────────────────────────────────────────────────

async def total_sessions() -> int:
    return await sessions_col().count_documents({})


async def total_queries() -> int:
    return await logs_col().count_documents({})


async def queries_today() -> int:
    from datetime import date
    start = datetime.combine(date.today(), datetime.min.time(), tzinfo=timezone.utc)
    return await logs_col().count_documents({"timestamp": {"$gte": start}})


async def avg_session_length() -> float:
    """Average number of messages per session."""
    pipeline = [
        {"$project": {"msg_count": {"$size": "$messages"}}},
        {"$group":   {"_id": None, "avg": {"$avg": "$msg_count"}}},
    ]
    async for doc in sessions_col().aggregate(pipeline):
        return round(doc["avg"], 2)
    return 0.0


async def top_questions(limit: int = 10) -> list[dict]:
    """Most frequent query strings across all sessions."""
    pipeline = [
        {"$group":   {"_id": "$query", "count": {"$sum": 1}}},
        {"$sort":    {"count": DESCENDING}},
        {"$limit":   limit},
        {"$project": {"question": "$_id", "count": 1, "_id": 0}},
    ]
    results = []
    async for doc in logs_col().aggregate(pipeline):
        results.append(doc)
    return results


async def queries_timeseries(days: int = 14) -> list[dict]:
    """Daily query counts for the last *days* days (UTC)."""
    from datetime import date, timedelta
    start = datetime.combine(date.today() - timedelta(days=days - 1),
                             datetime.min.time(), tzinfo=timezone.utc)
    pipeline = [
        {"$match":   {"timestamp": {"$gte": start}}},
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
            "count": {"$sum": 1},
        }},
        {"$sort":    {"_id": 1}},
        {"$project": {"date": "$_id", "count": 1, "_id": 0}},
    ]
    out = []
    async for doc in logs_col().aggregate(pipeline):
        out.append(doc)
    # Fill gaps so charts are continuous
    by_date = {d["date"]: d["count"] for d in out}
    result  = []
    for i in range(days):
        d = (date.today() - timedelta(days=days - 1 - i)).isoformat()
        result.append({"date": d, "count": by_date.get(d, 0)})
    return result


async def intent_distribution() -> list[dict]:
    pipeline = [
        {"$group":   {"_id": "$intent", "count": {"$sum": 1}}},
        {"$sort":    {"count": DESCENDING}},
        {"$project": {"intent": "$_id", "count": 1, "_id": 0}},
    ]
    out = []
    async for doc in logs_col().aggregate(pipeline):
        out.append({"intent": doc.get("intent") or "general", "count": doc["count"]})
    return out


async def recent_sessions(limit: int = 10) -> list[dict]:
    """Return latest sessions with last message + turn count."""
    cursor = (
        sessions_col()
        .find({})
        .sort("created_at", DESCENDING)
        .limit(limit)
    )
    out = []
    async for doc in cursor:
        msgs = doc.get("messages", [])
        last = msgs[-1]["content"] if msgs else ""
        out.append({
            "session_id":   doc["_id"],
            "created_at":   doc["created_at"],
            "turn_count":   len(msgs),
            "last_message": (last[:80] + "…") if len(last) > 80 else last,
        })
    return out


# ── Internal ──────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)
