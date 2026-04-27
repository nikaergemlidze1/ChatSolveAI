"""API integration tests with database and RAG dependencies faked."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

pytest.importorskip("slowapi")
pytest.importorskip("motor")
pytest.importorskip("langchain_openai")

from fastapi.testclient import TestClient

from api.main import app
from api.routes import chat as chat_routes
import api.auth as auth


class FakeRAG:
    def chat(self, query: str, session_id: str | None = None) -> dict:
        return {
            "answer": f"Answer for {query}",
            "source_documents": [
                {
                    "content": "Reset your password from the login page.",
                    "metadata": {"topic": "account"},
                    "score": 0.1,
                }
            ],
            "confidence": 0.95,
            "condensed_query": query,
        }

    async def astream_response(self, query: str, session_id: str | None = None):
        yield {"event": "token", "token": "Streaming "}
        yield {"event": "token", "token": "answer"}
        yield {
            "event": "final",
            "answer": "Streaming answer",
            "source_documents": [
                {
                    "content": "Use the Orders page to track shipping.",
                    "metadata": {"topic": "shipping"},
                    "score": 0.2,
                }
            ],
            "confidence": 0.88,
            "condensed_query": query,
        }

    def reset(self, session_id: str | None = None) -> None:
        return None


def _patch_dependencies(monkeypatch):
    app.state.rag = FakeRAG()
    auth._EXPECTED_KEY = None
    monkeypatch.setattr(chat_routes.db, "ensure_session", AsyncMock())
    monkeypatch.setattr(chat_routes.db, "append_message", AsyncMock())
    monkeypatch.setattr(chat_routes.db, "log_query", AsyncMock())
    monkeypatch.setattr(chat_routes.db, "delete_session", AsyncMock(return_value=2))


def test_blocking_chat_returns_enriched_response(monkeypatch):
    _patch_dependencies(monkeypatch)
    client = TestClient(app)

    response = client.post("/chat", json={"session_id": "s1", "query": "reset password"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Answer for reset password"
    assert body["confidence"] == 0.95
    assert body["condensed_query"] == "reset password"
    assert body["source_documents"][0]["metadata"]["topic"] == "account"


def test_streaming_chat_emits_tokens_and_final_metadata(monkeypatch):
    _patch_dependencies(monkeypatch)
    client = TestClient(app)

    response = client.post("/chat/stream", json={"session_id": "s1", "query": "track order"})

    assert response.status_code == 200
    assert 'data: {"token": "Streaming "}' in response.text
    assert 'data: {"token": "answer"}' in response.text
    assert '"event": "final"' in response.text
    assert '"confidence": 0.88' in response.text
    assert "Use the Orders page" in response.text
