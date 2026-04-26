"""
Suggest route — LLM-generated follow-up questions after an assistant answer.

A small in-process LRU cache fronts the LLM call. The same `last_answer`
asked across users will hit the cache instead of paying for another
chat-completion. Cache is keyed on (last_answer, n) only — no session
context, since suggestions are stateless w.r.t. the conversation.
"""

from __future__ import annotations

from collections import OrderedDict

from fastapi import APIRouter, Request

from api.limits import limiter
from api.models import SuggestRequest, SuggestResponse

router = APIRouter(prefix="/suggest", tags=["suggest"])


# Bounded LRU. 256 entries × ~1 KB each = trivial memory footprint.
_SUGGEST_CACHE_MAX = 256
_suggest_cache: "OrderedDict[tuple[str, int], list[str]]" = OrderedDict()


def _cache_get(key: tuple[str, int]) -> list[str] | None:
    val = _suggest_cache.get(key)
    if val is None:
        return None
    _suggest_cache.move_to_end(key)
    return val


def _cache_put(key: tuple[str, int], value: list[str]) -> None:
    _suggest_cache[key] = value
    _suggest_cache.move_to_end(key)
    while len(_suggest_cache) > _SUGGEST_CACHE_MAX:
        _suggest_cache.popitem(last=False)


@router.post("", response_model=SuggestResponse)
@limiter.limit("60/minute")
async def suggest(payload: SuggestRequest, request: Request):
    key = (payload.last_answer, payload.n)
    cached = _cache_get(key)
    if cached is not None:
        return SuggestResponse(suggestions=cached)

    rag = request.app.state.rag
    suggestions = rag.suggest_followups(payload.last_answer, n=payload.n)
    _cache_put(key, suggestions)
    return SuggestResponse(suggestions=suggestions)
