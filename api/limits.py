"""
Per-IP rate limiter (slowapi). Imported by both ``api.main`` (for the
exception handler / state attachment) and the per-route modules (for
the ``@limiter.limit(...)`` decorator).

Limits chosen so /chat (LLM cost) is the tightest, /chat/stream slightly
looser (multiple tokens per second is normal SSE behaviour), and the
cheap endpoints (/feedback, /suggest, /analytics) get a sane ceiling.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address


# Global default for any route that doesn't set its own limit.
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])
