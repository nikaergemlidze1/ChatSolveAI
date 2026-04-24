"""Latency-tracking middleware — persists request times to MongoDB."""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from api import database as db


# Paths worth tracking; skip docs/static to keep the collection small.
_TRACKED_PREFIXES = ("/chat", "/feedback", "/suggest")


class LatencyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start  = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            path = request.url.path
            if any(path.startswith(p) for p in _TRACKED_PREFIXES):
                # Fire-and-forget — never block the response for logging
                try:
                    await db.log_latency(path, request.method, elapsed_ms, status)
                except Exception:
                    pass
