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

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from pipeline.config import data_path
from pipeline.rag import build_rag_chain

from api.middleware import LatencyMiddleware
from api.routes.chat       import router as chat_router
from api.routes.analytics  import router as analytics_router
from api.routes.feedback   import router as feedback_router
from api.routes.suggest    import router as suggest_router


# ── Lifespan: build RAG chain once on startup ─────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("▶ Building LangChain RAG chain…")
    app.state.rag = build_rag_chain(
        corpus_path=data_path("chatbot_responses.json"),
        predefined_path=data_path("predefined_responses.json"),
    )
    print("✓ RAG chain ready")
    yield
    print("◼ Shutting down")


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
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def ratelimit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LatencyMiddleware)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(chat_router)
app.include_router(feedback_router)
app.include_router(suggest_router)
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
