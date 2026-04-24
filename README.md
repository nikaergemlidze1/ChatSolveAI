<div align="center">

# 🤖 ChatSolveAI
### AI-Powered Customer Support Automation

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.x-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0-47A248?style=for-the-badge&logo=mongodb&logoColor=white)](https://mongodb.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

*A production-grade, retrieval-augmented generation (RAG) chatbot for automated customer support — built with LangChain, FastAPI, MongoDB, and Docker.*

</div>

---

## 📌 Overview

**ChatSolveAI** is a full-stack AI customer support system that intelligently routes customer queries using a multi-stage RAG pipeline. It retrieves answers from a curated knowledge base when confidence is high, and falls back to GPT-3.5-turbo generation for novel or out-of-scope questions.

A production-ready portfolio application demonstrating key skills demanded by the modern AI engineering job market.

---

## 🏗️ System Architecture

```
┌────────────────┐      ┌─────────────────────────┐      ┌──────────────────┐
│  🖥️ Streamlit  |      │       ⚡FastAPI          │      │   🍃 MongoDB     │
│   (Frontend)   │ HTTP │        (Backend)        │ API  │     Atlas        │
│ Streamlit Cloud│◄────►│      Hosted on Render   │◄────►│ (Cloud Database) │
└────────────────┘      └────────────┬────────────┘      └──────────────────┘
                                     │
                        ┌────────────▼────────────┐
                        │   🦜 LangChain RAG      │
                        │   FAISS + GPT-3.5       │
                        └─────────────────────────┘
```
### 🌐 Cloud Deployment
The system is fully deployed across a distributed cloud architecture:

- Frontend: Streamlit Community Cloud (Auto-scaling UI).
- Backend: Render (FastAPI service with custom Docker environment).
- Database: MongoDB Atlas (Managed NoSQL cluster with network security & IP whitelisting).
- Security: All API keys and database credentials managed via encrypted environment variables.

### 🔄 Query Routing Pipeline

```
Customer Query
      │
      ▼
  🦜 LangChain LCEL Chain
      │
      ├─► Condense question with chat history
      │
      ├─► FAISS semantic retrieval (top-k docs)
      │
      ├─► Format QA prompt with context
      │
      ▼
  Confidence ≥ threshold?
      │
      ├── YES ──► 🔍 Return retrieved answer  (fast, ~100ms)
      │
      └── NO ───► 🤖 GPT-3.5-turbo generation (with conversation history)
                        │
                        ▼
                  💾 Log to MongoDB
                  (session + analytics)
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🦜 **LangChain LCEL RAG** | Modern composable chain with FAISS vectorstore and GPT-3.5-turbo |
| 🔍 **Hybrid Retrieval** | FAISS semantic search + BM25 lexical search fused with Reciprocal Rank Fusion |
| 🏆 **Cross-Encoder Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks top-K candidates for precision |
| 🏷️ **Intent Classification** | Zero-shot routing to billing / account / shipping / technical / general |
| ⚡ **SSE Streaming** | Token-by-token streaming from FastAPI → Streamlit via Server-Sent Events |
| 💾 **MongoDB Persistence** | Full session history and query analytics stored asynchronously via motor |
| 📊 **Live Analytics** | Real-time usage stats: sessions, queries, top questions, avg session length |
| 👍 **Feedback loop** | Per-answer thumbs up/down, stored for quality tracking |
| 💡 **Smart follow-ups** | LLM-generated follow-up question chips after each reply |
| 🎯 **Confidence + latency** | Confidence meter, intent pill, and server latency shown inline |
| 🛡️ **Rate limiting** | Per-IP 120 req/min via `slowapi` to protect the API |
| ⬇️ **Chat export** | Download any conversation as Markdown |
| 🐳 **Docker Compose** | One-command deployment of all 3 services with health checks |
| ☁️ **Hugging Face Spaces** | Backend hosted on HF Spaces (Docker SDK); frontend on Streamlit Cloud |
| 🔁 **GitHub Actions CI** | Automated lint + pytest on every push and PR |
| 📓 **Research Notebook** | 5-task educational notebook with evaluations (P@1, MRR, NDCG) |

---

## 🛠️ Tech Stack

### AI / ML
| Tool | Role |
|------|------|
| **OpenAI** `text-embedding-3-small` | Document and query embeddings (1536-dim) |
| **OpenAI** `gpt-3.5-turbo` | Response generation and question condensing |
| **LangChain LCEL** | RAG chain composition, prompt templates, memory management |
| **FAISS** | Approximate nearest-neighbour vector search |
| **BM25** | Lexical keyword search (rank-bm25) |
| **sentence-transformers** | Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) |

### Backend
| Tool | Role |
|------|------|
| **FastAPI** | Async REST API with SSE streaming and Swagger docs |
| **Uvicorn** | ASGI server |
| **Pydantic v2** | Request/response validation and serialisation |
| **motor** | Async MongoDB driver |
| **MongoDB 7** | Session storage and analytics aggregation |

### Frontend & DevOps
| Tool | Role |
|------|------|
| **Streamlit** | Chat UI with live sidebar analytics |
| **Docker + Compose** | Multi-service containerisation with health checks |
| **Jenkins** | CI/CD pipeline (lint → test → build → push) |

---

## 📁 Project Structure

```
ChatSolveAI/
│
├── 📓 notebook.ipynb              # 5-task research & evaluation notebook
│
├── 🧠 pipeline/                   # Reusable ML pipeline modules
│   ├── config.py                  # Constants, paths, model names
│   ├── embeddings.py              # OpenAI embedding utilities + retry logic
│   ├── rag.py                     # 🦜 LangChain LCEL RAG chain (main)
│   ├── retrieval.py               # FAISS + BM25 hybrid retriever
│   ├── reranker.py                # Cross-encoder reranker
│   ├── classifier.py              # Zero-shot intent classifier
│   ├── chatbot.py                 # Orchestration class (notebook use)
│   └── evaluate.py                # Precision@K, MRR, NDCG metrics
│
├── ⚡ api/                        # FastAPI backend
│   ├── main.py                    # App factory, lifespan, routers
│   ├── database.py                # MongoDB helpers (motor async)
│   ├── models.py                  # Pydantic schemas
│   └── routes/
│       ├── chat.py                # POST /chat, POST /chat/stream
│       └── analytics.py           # GET /analytics, GET /history/{id}
│
├── 🖥️  app.py                     # Streamlit frontend (calls FastAPI)
│
├── 🐳 Dockerfile                  # API service (multi-stage build)
├── 🐳 Dockerfile.streamlit        # Streamlit service
├── 🐳 docker-compose.yml          # Full stack orchestration
├── 🔁 Jenkinsfile                 # CI/CD pipeline definition
│
├── 📊 data/
│   ├── knowledge_base.csv         # 501 support documents
│   ├── chatbot_responses.json     # 19 curated QA pairs (RAG corpus)
│   ├── predefined_responses.json  # 19 predefined response templates
│   └── processed_queries.csv      # 501 test queries
│
├── requirements.txt               # All dependencies (local/notebook)
├── requirements.api.txt           # API service dependencies (Docker)
├── requirements.streamlit.txt     # Streamlit dependencies (Docker)
└── .env.example                   # Environment variable template
```

---

## 🚀 Quick Start

### Option 1 — Docker (Recommended)

```bash
# 1. Clone the repo
git clone https://github.com/your-username/chatsolveai.git
cd chatsolveai

# 2. Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-...

# 3. Build and start all services
docker compose up --build

# 4. Open the app
# 🖥️  Chat UI  → http://localhost:8501
# 📖 API docs  → http://localhost:8000/docs
# ❤️  Health   → http://localhost:8000/health
```

### Option 2 — Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-...
#            MONGO_URL=mongodb://localhost:27017

# 3. Start MongoDB (requires Docker or local install)
docker run -d -p 27017:27017 mongo:7

# 4. Start FastAPI backend
uvicorn api.main:app --reload --port 8000

# 5. Start Streamlit frontend (new terminal)
streamlit run app.py
```

### Option 3 — Jupyter Notebook (Research / Offline)

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

---

## 💬 Demo Questions

Try these in the chat to see both **retrieval** and **generation** in action:

| # | Question | Expected behaviour |
|---|----------|--------------------|
| 1 | `How do I reset my password?` | 🔍 **Retrieved** — exact match in knowledge base |
| 2 | `What payment methods do you accept?` | 🔍 **Retrieved** — high confidence answer |
| 3 | `I received a damaged product, how do I get a replacement?` | 🔍 **Retrieved** — paraphrase match via semantic search |
| 4 | `Do you offer phone support on weekends?` | 🤖 **Generated** — out-of-scope, GPT answers from context |
| 5 | `Can I use cryptocurrency to pay for my order?` | 🤖 **Generated** — novel query, no match in knowledge base |

> 💡 Watch the **API docs** at `http://localhost:8000/docs` while chatting — you can see each request hit the `/chat/stream` endpoint in real time.

---

## 🔑 Environment Variables

Copy `.env.example` to `.env` and fill in:

```env
# Required
OPENAI_API_KEY=sk-your-api-key-here

# Optional (defaults shown)
MONGO_URL=mongodb://localhost:27017   # use mongodb://mongo:27017 in Docker
API_URL=http://localhost:8000         # use http://api:8000 in Docker
```

---

## 📡 API Reference

The FastAPI backend is fully documented at **http://localhost:8000/docs** (Swagger UI).

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Blocking chat — full JSON response |
| `POST` | `/chat/stream` | SSE streaming — token-by-token |
| `DELETE` | `/chat/session/{id}` | Clear session from MongoDB |
| `GET` | `/analytics` | Aggregate usage statistics |
| `GET` | `/history/{session_id}` | Full conversation history |
| `GET` | `/health` | Liveness check |

### Example — Blocking Chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc-123",
    "query": "How do I reset my password?"
  }'
```

```json
{
  "session_id": "abc-123",
  "query": "How do I reset my password?",
  "answer": "You can reset your password by clicking 'Forgot Password' on the login page.",
  "source_documents": [
    {
      "content": "You can reset your password by clicking on 'Forgot Password' at the login page.",
      "metadata": { "source_query": "What's the process for resetting my password?", "confidence_score": 0.95 }
    }
  ],
  "timestamp": "2026-04-08T10:00:00Z"
}
```

---

## 📓 Notebook Walkthrough

The research notebook (`notebook.ipynb`) covers 5 tasks:

| Task | Description | Key Output |
|------|-------------|------------|
| **Task 1** 🗂️ | Knowledge base embeddings | 501 documents → `knowledge_embeddings.json` |
| **Task 2** 🔍 | Query similarity search | Top-3 responses per query with confidence scores |
| **Task 3** 🤖 | Conversational chatbot | Retrieved vs generated routing demo |
| **Task 4** 📊 | Advanced retrieval evaluation | P@1=1.0, MRR=1.0, NDCG@5=1.0 |
| **Task 5** 🔬 | End-to-end diagnostics | Intent + confidence + routing per query |

### Evaluation Results (Task 4)

```
┌──────────────┬────────┐
│ Metric       │ Score  │
├──────────────┼────────┤
│ Precision@1  │ 1.0000 │
│ Precision@3  │ 1.0000 │
│ MRR          │ 1.0000 │
│ NDCG@5       │ 1.0000 │
└──────────────┴────────┘
15 labelled queries, all retrieved at rank 1.
```

> 💡 The evaluation set is built from the same corpus as the retriever — these scores
> reflect in-distribution retrieval precision. For a production system, an independent
> held-out test set with paraphrases and out-of-scope queries would give a more
> realistic picture.

---

## 🔁 CI/CD with Jenkins

The `Jenkinsfile` defines a full pipeline:

```
Checkout → Install → Lint (flake8) → Test (pytest + coverage) → Docker Build → Docker Push
```

> Push to registry only triggers on the `main` branch.
> Configure `OPENAI_API_KEY_CRED` and `DOCKER_REGISTRY_CRED` in Jenkins credentials.

---

## 🗺️ Roadmap / Future Improvements

- [ ] 🧪 Add pytest test suite (`tests/`) for pipeline and API routes
- [ ] 📈 Evaluation dashboard — visualise P@1 / MRR trends over time in Streamlit
- [ ] 🔀 Hybrid search threshold tuning — expose `alpha` and `SIM_THRESHOLD` as API params
- [ ] 🌐 Deploy to cloud — Railway, Render, or AWS ECS with the existing Docker images
- [ ] 🔐 MongoDB authentication — add username/password for production deployments
- [ ] 📱 Responsive UI — replace Streamlit with a React/Next.js frontend for mobile support
- [ ] 🔄 Webhook support — integrate with Slack or WhatsApp via FastAPI webhooks

---

## 🎓 Skills Demonstrated

This project directly maps to requirements seen in AI/ML engineering job postings:

| Job Requirement | Implementation |
|-----------------|----------------|
| LangChain / RAG | `pipeline/rag.py` — LCEL chain, FAISS vectorstore, streaming |
| OpenAI API | Embeddings + GPT-3.5-turbo throughout |
| Vector databases | FAISS (local) — swap-ready for Pinecone / Weaviate |
| FastAPI / REST API | Async routes, SSE streaming, Pydantic, Swagger |
| MongoDB / NoSQL | motor async driver, aggregation pipelines for analytics |
| Docker | Multi-stage builds, non-root user, health checks |
| docker-compose | 3-service orchestration with dependency management |
| Jenkins CI/CD | Full pipeline: lint → test → build → push |
| Python best practices | Type hints, docstrings, modular package structure |

---

## 👤 Author

**Nika Ergemlidze**
- Data Scientist / AI Engineer (Certified)

---

<div align="center">

*Built with ❤️ using LangChain, FastAPI, MongoDB, and OpenAI*

</div>
