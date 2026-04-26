"""
Feedback route — capture 👍 / 👎 on individual answers for quality tracking.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from api.limits import limiter
from api.models import FeedbackRequest, FeedbackResponse
from api import database as db

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse)
@limiter.limit("60/minute")
async def submit_feedback(payload: FeedbackRequest, request: Request):
    await db.log_feedback(
        session_id=payload.session_id,
        query=payload.query,
        answer=payload.answer,
        rating=payload.rating,
        note=payload.note,
    )
    return FeedbackResponse(ok=True)
