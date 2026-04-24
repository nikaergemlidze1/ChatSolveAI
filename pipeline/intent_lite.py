"""
Lightweight keyword-based intent tagging.

The full ``IntentClassifier`` (embeddings-based) is accurate but costs an
OpenAI embedding call per query. For simple analytics labelling we do a
token-overlap match against the category keyword lists defined in
``config.INTENT_CATEGORIES`` — zero API cost, good enough for dashboards.
"""

from __future__ import annotations

import re
from .config import INTENT_CATEGORIES

_WORD = re.compile(r"[a-zA-Z']+")


def tag_intent(query: str) -> str:
    """Return the best-matching intent name, or ``"general"`` if none match."""
    tokens = {t.lower() for t in _WORD.findall(query)}
    if not tokens:
        return "general"

    best_intent = "general"
    best_score  = 0
    for intent, keywords in INTENT_CATEGORIES.items():
        kw_set = set(keywords.lower().split())
        score  = len(tokens & kw_set)
        if score > best_score:
            best_score  = score
            best_intent = intent
    return best_intent
