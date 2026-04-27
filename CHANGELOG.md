# Changelog

## Unreleased

- Added configurable CORS allowlist, optional API-key auth, and per-route rate
  limits for protected backend endpoints.
- Added PII redaction before MongoDB storage, TTL indexes, and per-session
  message caps.
- Added structured JSON/text logging and optional Sentry integration.
- Added pinned dependency upper bounds for API, Streamlit, and local
  requirements.
- Removed stale analytics claims from the Streamlit sidebar documentation.

## 2026-04-27

- Stabilized Streamlit chat reset behavior by making empty-state and
  history-state rendering mutually exclusive.
- Moved sidebar tech-stack rendering to a single location to avoid duplicate
  Streamlit Cloud DOM artifacts.
- Added follow-up suggestion chip containment so new-chat resets clear stale
  widgets reliably.

## Initial

- Built Streamlit frontend, FastAPI backend, LangChain RAG pipeline, FAISS
  vectorstore, MongoDB persistence, Dockerfiles, and notebook workflow.
