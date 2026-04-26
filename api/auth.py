"""
Lightweight API-key auth for ChatSolveAI.

Set the ``API_KEY`` environment variable (or Streamlit secret) to enable.
Clients send ``X-API-Key: <value>`` on every request.

If ``API_KEY`` is unset, auth is disabled — useful for local dev where
the Streamlit container talks to the API on the same Docker network.
"""

from __future__ import annotations

import os

from fastapi import Header, HTTPException, status


_EXPECTED_KEY: str | None = os.getenv("API_KEY") or None


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency. Raises 401 if header missing or wrong.

    No-op when ``API_KEY`` env var is unset (dev mode).
    """
    if not _EXPECTED_KEY:
        return
    if not x_api_key or x_api_key != _EXPECTED_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )
