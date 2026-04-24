"""Centralised configuration — models, paths, tuning knobs."""

import os
from pathlib import Path

# python-dotenv is handy for local dev but not required in production
# (Docker / HF Spaces / CI inject env vars directly). Keep the import optional
# so the package can be consumed by lightweight test environments.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:  # pragma: no cover
    pass

# ── API ────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# ── Models ─────────────────────────────────────────────────────────────────────
EMBED_MODEL   = "text-embedding-3-small"
CHAT_MODEL    = "gpt-3.5-turbo"
RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

def data_path(filename: str) -> Path:
    """Return first existing path: project root or /mnt/data (DataCamp)."""
    for candidate in [BASE_DIR / filename, Path("/mnt/data") / filename]:
        if candidate.exists():
            return candidate
    return BASE_DIR / filename  # let callers raise a clear FileNotFoundError

# ── Retrieval ──────────────────────────────────────────────────────────────────
CHUNK_SIZE       = 100    # max texts per embedding API call
SIM_THRESHOLD    = 0.78   # confidence below this → GPT fallback
HYBRID_ALPHA     = 0.5    # weight: semantic vs lexical (0 = pure BM25, 1 = pure FAISS)
TOP_K_CANDIDATES = 20     # candidates fed into the cross-encoder
TOP_K_RERANK     = 3      # results returned after reranking

# ── Intent categories (used by IntentClassifier) ───────────────────────────────
INTENT_CATEGORIES: dict[str, str] = {
    "billing":   "billing payment invoice refund subscription price discount charge",
    "account":   "account profile password email login settings authentication security",
    "shipping":  "shipping delivery order tracking address package arrived",
    "technical": "technical support error bug problem issue not working app troubleshoot",
    "general":   "general information help question policy terms conditions",
}
