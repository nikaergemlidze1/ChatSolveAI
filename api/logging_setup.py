"""
Structured logging for the ChatSolveAI API.

Two formats, switchable via ``LOG_FORMAT``:
  - ``json``  → one JSON object per line; suitable for log shippers
                like Logtail, Datadog, Axiom.
  - ``text``  → human-readable; convenient for local dev.

Level via ``LOG_LEVEL`` (default INFO).

Usage
-----
Call :func:`setup_logging` once at app startup (before any logger
emits records).  Modules then use ``logging.getLogger(__name__)`` as
usual.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Render LogRecords as a single JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts":      datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Allow callers to attach structured fields via the `extra=` kwarg.
        for key, val in record.__dict__.items():
            if key in {
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "message", "module",
                "msecs", "msg", "name", "pathname", "process", "processName",
                "relativeCreated", "stack_info", "thread", "threadName",
                "taskName",
            }:
                continue
            payload.setdefault(key, val)
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Configure the root logger. Idempotent — safe to call more than once."""
    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    fmt_name   = (fmt   or os.getenv("LOG_FORMAT", "json")).lower()

    handler = logging.StreamHandler(sys.stdout)
    if fmt_name == "text":
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        ))
    else:
        handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level_name)

    # Quiet noisy third parties (uvicorn access logs already covered by
    # our middleware-recorded latency entries; httpx logs every request).
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
