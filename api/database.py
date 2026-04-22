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
) -> None:
    """Insert a single query-answer log entry."""
    await logs_col().insert_one(
        {
            "session_id": session_id,
            "query":      query,
            "answer":     answer,
            "sources":    sources,
            "timestamp":  _now(),
        }
    )


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


# ── Internal ──────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)
