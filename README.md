# 🤖 Agentic RAG — Career Point University Assistant

> Production-grade Retrieval-Augmented Generation system built with LangGraph, FastAPI, and Qdrant. Prioritizes retrieval quality over LLM orchestration complexity — lower cost, lower latency, better accuracy.

---

## Architecture Overview

```
User Request
     │
     ▼
┌─────────────┐
│   FastAPI   │  POST /chat
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│           LangGraph Workflow         │
│                                     │
│  ┌──────────┐    ┌──────────────┐   │
│  │Guardrails│───▶│  Retriever   │   │
│  └──────────┘    └──────┬───────┘   │
│       │                 │           │
│   [Unsafe]          [Context]       │
│       │                 ▼           │
│       │          ┌──────────────┐   │
│       │          │    Agent     │   │
│       │          │ Llama 3.3 70B│   │
│       │          └──────┬───────┘   │
│       │                 │           │
└───────┼─────────────────┼───────────┘
        │                 │
        ▼                 ▼
   Safe Refusal      Final Answer
```

---

## Key Design Decisions

| Traditional RAG | This System |
|---|---|
| Fixed-size chunking | **Semantic chunking** — preserves meaning |
| Multiple LLM routing layers | **Retrieval-first** — fewer LLM calls |
| No safety checks | **Guardrail classifier** on every request |
| No conversation memory | **Redis checkpointing** per session |
| No privacy protection | **PII middleware** — email, card masking |
| Hallucination-prone | **Context-grounded** responses |

---

## Tech Stack

- **Backend** — FastAPI
- **Orchestration** — LangGraph
- **LLM** — Groq Llama 3.3 70B
- **Embeddings** — BAAI/bge-base-en-v1.5 (HuggingFace)
- **Vector DB** — Qdrant
- **Memory** — Redis
- **Deployment** — Docker Compose

---

## How It Works

### 1. Guardrails
Every query passes through a structured LLM classifier first. Blocks harmful queries (hacking, fraud, phishing, illegal activities) before any retrieval happens. Returns a safe refusal message instantly.

### 2. Semantic Retrieval
Instead of fixed-size chunking, the system uses meaning-based text splitting. This preserves context boundaries naturally — eliminating the need for a query classifier, retrieval validator, or routing LLM.

### 3. Agent Reasoning
The LangChain agent combines:
- Retrieved context from Qdrant (top 3 semantic matches)
- Full conversation history from Redis
- DuckDuckGo search tool (fallback only when context is insufficient)
- PII middleware (strips sensitive data before LLM sees it)

### 4. Conversation Memory
Redis-based LangGraph checkpointing maintains session state per `thread_id`. Each conversation is isolated and persistent across requests.

---

## Project Structure

```
.
├── main.py          # FastAPI app, lifespan, CORS, endpoints
├── graph.py         # LangGraph workflow definition
├── agent.py         # LangChain agent with tools + PII middleware
├── retriever.py     # Qdrant similarity search
├── extractor.py     # Web scraper + semantic chunking + ingestion
├── guardrails.py    # Safety classifier (structured output)
├── state.py         # AgentState schema
├── llm.py           # LLM + embedding model config
├── docker-compose.yml
└── .env
```

---

## Getting Started

### Prerequisites
- Docker + Docker Compose
- Groq API key
- Qdrant (local via Docker or Qdrant Cloud)

### Run with Docker

```bash
git clone https://github.com/yourusername/agentic-rag-cpu
cd agentic-rag-cpu
cp .env.example .env   # add your API keys
docker-compose up --build
```

Services started:
- `fast_api` → http://localhost:8000
- `qdrant` → http://localhost:6333
- `redis` → localhost:6379

### Run Manually

```bash
pip install uv
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

> Use Python 3.11 for best compatibility. Avoid 3.13.

---

## API

### Health Check
```
GET /
```

### Chat
```
POST /chat
Content-Type: application/json

{
  "msg": "What are the admission requirements?",
  "thread_id": "user-session-123"
}
```

**Response:**
```json
{
  "answer": "Career Point University admission requires..."
}
```

---

## Environment Variables

```env
GROQ_API_KEY=your_groq_key
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=optional_for_cloud
REDIS_URL=redis://localhost:6379
```

---

## Why Retrieval-First?

Most RAG systems add LLM layers to compensate for poor retrieval — query classifiers, retrieval validators, re-rankers. This system inverts that approach:

> **Better retrieval eliminates the need for extra LLM layers.**

Semantic chunking ensures retrieved chunks are contextually complete. This means:
- Fewer LLM calls per request
- Lower inference cost
- Reduced latency
- Simpler architecture
- Better scalability

---

## Built By

**Mohit** — AI Engineer  
[GitHub](https://github.com/yourusername) · [LinkedIn](https://linkedin.com/in/yourprofile)