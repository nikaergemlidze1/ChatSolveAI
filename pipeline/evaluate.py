"""
Evaluation module — Precision@K, MRR, NDCG for retrieval quality.

Usage
-----
from pipeline.evaluate import EVAL_SET, evaluate
results = evaluate(retriever, reranker, EVAL_SET)
print(results)
"""

from __future__ import annotations

import math
from typing import Protocol


class RetrieverLike(Protocol):
    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        ...


class RerankerLike(Protocol):
    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        ...

# ── Ground-truth evaluation set ───────────────────────────────────────────────
# (query, expected_response) pairs — hand-curated from predefined_responses.json

EVAL_SET: list[tuple[str, str]] = [
    (
        "How do I reset my password?",
        "You can reset your password by clicking on 'Forgot Password' at the login page.",
    ),
    (
        "How can I contact customer support?",
        "You can contact our customer support via email or live chat on our website.",
    ),
    (
        "What is your refund policy?",
        "Our refund policy allows refunds within 30 days of purchase with a valid receipt.",
    ),
    (
        "How do I track my order?",
        "Track your order by logging into your account and checking the 'Orders' section.",
    ),
    (
        "What payment methods do you accept?",
        "We accept Visa, MasterCard, PayPal, and Apple Pay.",
    ),
    (
        "Can I change my shipping address after placing an order?",
        "Unfortunately, you cannot change the shipping address after placing the order.",
    ),
    (
        "Do you offer international shipping?",
        "Yes, we offer international shipping to select countries.",
    ),
    (
        "Are there any discounts for first-time customers?",
        "Yes! First-time customers can use code WELCOME10 for 10% off.",
    ),
    (
        "I received a damaged product, how can I get a replacement?",
        "If you received a damaged product, please contact support with images for a replacement.",
    ),
    (
        "How do I update my account details?",
        "You can update your account details in the 'Settings' section of your profile.",
    ),
    (
        "Can I cancel my order after it's been shipped?",
        "Orders cannot be canceled after they have been shipped.",
    ),
    (
        "Are my payment details secure?",
        "Your payment details are encrypted and secure with our system.",
    ),
    (
        "Do you offer gift wrapping?",
        "Yes, we offer gift wrapping for an additional fee.",
    ),
    (
        "How do I redeem a promo code?",
        "Enter your promo code at checkout to apply the discount.",
    ),
    (
        "What should I do if my payment fails?",
        "If your payment fails, try a different card or contact your bank.",
    ),
]


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _dcg(relevances: list[float]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def _ndcg(relevances: list[float]) -> float:
    ideal = _dcg(sorted(relevances, reverse=True))
    return _dcg(relevances) / ideal if ideal > 0 else 0.0


# ── Main evaluation function ───────────────────────────────────────────────────

def evaluate(
    retriever: RetrieverLike,
    reranker:  RerankerLike,
    eval_set:  list[tuple[str, str]] = EVAL_SET,
    top_k:     int = 5,
) -> dict:
    """
    Run end-to-end retrieval + reranking on *eval_set*.

    Returns
    -------
    dict with keys: precision@1, precision@3, mrr, ndcg@5, n_queries
    """
    p1_hits, p3_hits, reciprocal_ranks, ndcg_scores = [], [], [], []

    for query, expected in eval_set:
        candidates = retriever.search(query, top_k=top_k * 4)   # extra candidates for reranker
        reranked   = reranker.rerank(query, candidates, top_k=top_k)
        texts      = [c["text"] for c in reranked]

        # Relevance: binary (1 if exact match, else 0)
        relevances = [1.0 if t == expected else 0.0 for t in texts]

        p1_hits.append(1 if (relevances[:1] and relevances[0] == 1.0) else 0)
        p3_hits.append(1 if any(r == 1.0 for r in relevances[:3]) else 0)

        rr = next((1.0 / (i + 1) for i, r in enumerate(relevances) if r == 1.0), 0.0)
        reciprocal_ranks.append(rr)
        ndcg_scores.append(_ndcg(relevances[:top_k]))

    n = len(eval_set)
    return {
        "n_queries":    n,
        "precision@1":  round(sum(p1_hits) / n, 4),
        "precision@3":  round(sum(p3_hits) / n, 4),
        "mrr":          round(sum(reciprocal_ranks) / n, 4),
        f"ndcg@{top_k}": round(sum(ndcg_scores) / n, 4),
    }
