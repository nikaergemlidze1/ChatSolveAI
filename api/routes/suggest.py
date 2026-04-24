"""
Suggest route — LLM-generated follow-up questions after an assistant answer.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from api.models import SuggestRequest, SuggestResponse

router = APIRouter(prefix="/suggest", tags=["suggest"])


@router.post("", response_model=SuggestResponse)
async def suggest(payload: SuggestRequest, request: Request):
    rag = request.app.state.rag
    suggestions = rag.suggest_followups(payload.last_answer, n=payload.n)
    return SuggestResponse(suggestions=suggestions)
