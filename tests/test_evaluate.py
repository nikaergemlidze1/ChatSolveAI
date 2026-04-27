"""Deterministic retrieval metric regression tests."""

from __future__ import annotations

from pipeline.evaluate import evaluate


class FakeRetriever:
    def __init__(self, rankings: dict[str, list[str]]) -> None:
        self.rankings = rankings

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        texts = self.rankings[query]
        return [{"text": text} for text in texts[:top_k]]


class IdentityReranker:
    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        return candidates[:top_k]


def test_evaluate_metrics_are_stable_for_known_rankings():
    eval_set = [
        ("q1", "a1"),
        ("q2", "a2"),
        ("q3", "a3"),
    ]
    retriever = FakeRetriever(
        {
            "q1": ["a1", "x", "y"],
            "q2": ["x", "a2", "y"],
            "q3": ["x", "y", "z"],
        }
    )

    metrics = evaluate(retriever, IdentityReranker(), eval_set=eval_set, top_k=3)

    assert metrics == {
        "n_queries": 3,
        "precision@1": 0.3333,
        "precision@3": 0.6667,
        "mrr": 0.5,
        "ndcg@3": 0.5436,
    }
