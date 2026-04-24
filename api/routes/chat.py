"""
Chat routes — /chat and /chat/stream.

POST /chat              → blocking JSON response (with confidence, intent, latency)
POST /chat/stream       → token-by-token SSE stream
DELETE /session/{id}    → wipe session from MongoDB
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from api.models import ChatRequest, ChatResponse, SourceDocument
from api import database as db
from pipeline.intent_lite import tag_intent

router = APIRouter(prefix="/chat", tags=["chat"])


def _get_rag(request: Request):
    return request.app.state.rag


# ── Blocking chat ─────────────────────────────────────────────────────────────

@router.post("", response_model=ChatResponse, summary="Send a message (blocking)")
async def chat(payload: ChatRequest, request: Request):
    rag = _get_rag(request)
    t0 = time.perf_counter()

    await db.ensure_session(payload.session_id)
    await db.append_message(payload.session_id, "user", payload.query)

    result = rag.chat(payload.query)

    answer     = result["answer"]
    sources    = [s["content"] for s in result.get("source_documents", [])]
    confidence = float(result.get("confidence", 0.0))
    condensed  = result.get("condensed_query", payload.query)
    intent     = tag_intent(payload.query)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    await db.append_message(payload.session_id, "assistant", answer)
    await db.log_query(
        payload.session_id, payload.query, answer, sources,
        intent=intent, confidence=confidence,
    )

    return ChatResponse(
        session_id=payload.session_id,
        query=payload.query,
        answer=answer,
        source_documents=[
            SourceDocument(
                content=s["content"],
                metadata=s.get("metadata", {}),
                score=s.get("score"),
            )
            for s in result.get("source_documents", [])
        ],
        confidence=confidence,
        condensed_query=condensed,
        intent=intent,
        latency_ms=latency_ms,
        timestamp=datetime.now(timezone.utc),
    )


# ── Streaming chat ────────────────────────────────────────────────────────────

@router.post("/stream", summary="Send a message (SSE streaming)")
async def chat_stream(payload: ChatRequest, request: Request):
    rag = _get_rag(request)
    intent = tag_intent(payload.query)

    await db.ensure_session(payload.session_id)
    await db.append_message(payload.session_id, "user", payload.query)

    full_answer: list[str] = []

    async def generate():
        # Opening metadata event — frontend can show intent/condensed query early
        yield f"data: {json.dumps({'event': 'meta', 'intent': intent})}\n\n"

        async for token in rag.astream(payload.query):
            full_answer.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"

        answer = "".join(full_answer)
        await db.append_message(payload.session_id, "assistant", answer)
        await db.log_query(
            payload.session_id, payload.query, answer, [],
            intent=intent, confidence=0.0,
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── Session reset ─────────────────────────────────────────────────────────────

@router.delete(
    "/session/{session_id}",
    summary="Delete a session and clear its conversation memory",
)
async def delete_session(session_id: str, request: Request):
    rag = _get_rag(request)
    rag.reset()
    deleted = await db.delete_session(session_id)
    return {"session_id": session_id, "logs_deleted": deleted}
