---
title: ChatSolveAI API
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: FastAPI + LangChain RAG backend for ChatSolveAI
---

# ChatSolveAI — FastAPI Backend

This Space hosts the FastAPI + LangChain RAG backend for **ChatSolveAI**, a
customer-support chatbot. The Streamlit frontend lives on Streamlit Community
Cloud and calls this Space over HTTPS.

## Endpoints

- `GET  /health`     — liveness probe
- `GET  /docs`       — Swagger UI
- `POST /chat`       — blocking chat
- `POST /chat/stream` — SSE token stream
- `GET  /analytics`  — usage stats
- `GET  /history/{session_id}` — full conversation history
- `DELETE /chat/session/{id}`  — reset session

## Required Space secrets

Add these under **Settings → Variables and secrets**:

| Key              | Value                                                                 |
|------------------|-----------------------------------------------------------------------|
| `OPENAI_API_KEY` | Your OpenAI API key                                                   |
| `MONGO_URL`      | `mongodb+srv://USER:PASS@HOST/chatsolveai?retryWrites=true&w=majority` |

## Stack

FastAPI · LangChain 1.x · FAISS · GPT-3.5-turbo · MongoDB Atlas
