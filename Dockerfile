# ── FastAPI + LangChain RAG service ──────────────────────────────────────────
# Multi-stage build: keeps the final image lean by separating dependency
# installation from the runtime layer.

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System libs needed by faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.api.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.api.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY pipeline/ ./pipeline/
COPY api/       ./api/
COPY chatbot_responses.json   .
COPY knowledge_base.csv       .
COPY predefined_responses.json .
COPY processed_queries.csv    .

# Non-root user for security
RUN useradd --no-create-home appuser && chown -R appuser /app
USER appuser

# Port is provided by the host (Render, HF Spaces) via $PORT; 7860 is the HF Spaces default.
# Shell form so ${PORT:-7860} is expanded at runtime.
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1

EXPOSE 7860
