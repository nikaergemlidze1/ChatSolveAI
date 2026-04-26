"""
ChatSolveAI FastAPI application.

Startup sequence
----------------
1. Load LangChain RAG chain (builds FAISS vectorstore from chatbot_responses.json)
2. Mount chat / feedback / suggest / analytics routers
3. Register latency middleware + rate limiting
4. Start serving

Run locally
-----------
uvicorn api.main:app --reload --port 8000

In Docker / HF Spaces
---------------------
Handled by Dockerfile (respects $PORT).

Interactive docs
----------------
http://localhost:8000/docs  (Swagger UI)
http://localhost:8000/redoc (ReDoc)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from slowapi.errors import RateLimitExceeded

from pipeline.config import data_path
from pipeline.rag import build_rag_chain

from api import database as db
from api.auth import verify_api_key
from api.limits import limiter
from api.logging_setup import setup_logging
from api.middleware import LatencyMiddleware
from api.routes.chat       import router as chat_router
from api.routes.analytics  import router as analytics_router
from api.routes.feedback   import router as feedback_router
from api.routes.suggest    import router as suggest_router
from api.sentry_setup import init_sentry

# Initialise logging + Sentry as early as possible so import-time errors
# in any module below are captured.
setup_logging()
init_sentry()

logger = logging.getLogger(__name__)


# ── Lifespan: build RAG chain once on startup ─────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Building LangChain RAG chain")
    app.state.rag = build_rag_chain(
        corpus_path=data_path("chatbot_responses.json"),
        predefined_path=data_path("predefined_responses.json"),
    )
    logger.info("RAG chain ready")

    # TTL indexes — MongoDB auto-prunes old logs / sessions per
    # MONGO_TTL_DAYS (default 90). Idempotent; safe to call every boot.
    try:
        await db.ensure_indexes()
        logger.info("MongoDB TTL indexes ensured")
    except Exception as exc:
        # Index creation must never block app startup; log and continue.
        logger.warning("ensure_indexes failed: %r", exc)

    yield
    logger.info("Shutting down")


# ── App factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="ChatSolveAI API",
    description=(
        "Production customer-support backend powered by LangChain RAG.\n\n"
        "**Stack**: FastAPI · LangChain · FAISS · GPT-3.5-turbo · MongoDB\n\n"
        "Use `/chat` for blocking responses or `/chat/stream` for SSE streaming."
    ),
    version="2.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Rate limiting (per-IP) ────────────────────────────────────────────────────
# `limiter` is defined in `api.limits` so per-route modules can import it
# without circular dependency on this file. Per-route ceilings live next
# to the routes themselves (see `@limiter.limit(...)` decorators).
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def ratelimit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )

# ── CORS ──────────────────────────────────────────────────────────────────────
# Pin allowed origins via ALLOWED_ORIGINS env var (comma-separated list).
# Defaults cover local Docker / Streamlit dev. Production: set the env var
# to the actual Streamlit Cloud / HF Spaces frontend URL.
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:8501,http://localhost:3000,http://127.0.0.1:8501",
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)
app.add_middleware(LatencyMiddleware)

# ── Routers ───────────────────────────────────────────────────────────────────
# Chat / feedback / suggest require X-API-Key (when API_KEY env var is set).
# Analytics + health are public reads — protect those at the network layer
# (HF Spaces secret URL, IP allow-list, etc.) if they need locking down.
_AUTHED = [Depends(verify_api_key)]

app.include_router(chat_router,     dependencies=_AUTHED)
app.include_router(feedback_router, dependencies=_AUTHED)
app.include_router(suggest_router,  dependencies=_AUTHED)
app.include_router(analytics_router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "ChatSolveAI API",
        "version": "2.1.0",
        "docs":    "/docs",
        "endpoints": [
            "/chat", "/chat/stream", "/feedback", "/suggest",
            "/analytics", "/analytics/timeseries", "/analytics/intents",
            "/analytics/latency", "/analytics/feedback",
            "/sessions", "/history/{session_id}", "/health",
        ],
    }
