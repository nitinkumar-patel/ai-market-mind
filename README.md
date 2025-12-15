## MarketMind – Agentic Market Research Assistant

**MarketMind** is a FastAPI + LangGraph backend that turns an LLM into an **autonomous market research agent** for brands and topics relevant to marketing and communications (e.g., a Stagwell-style agency workflow).

Given a topic like **"Trends in GenAI for Marketing 2025"**, MarketMind:

- **Plans** focused queries
- **Searches** the web via a tool (Tavily)
- **Summarizes & stores** key insights into **Postgres + pgvector** (RAG memory)
- **Writes** an executive-ready summary for marketers
- **Reviews** the output with guardrails before returning a structured result

The `/api/v1/research` endpoint streams progress as **Server-Sent Events (SSE)**: `"Searching..."`, `"Summarizing..."`, `"Reviewing..."`, etc., then returns a final JSON payload that you can render in any frontend.

---

## Architecture Overview

- **Framework**: `FastAPI` (async)
- **Orchestration**: `LangGraph` (explicit graph with loops)
- **LLM & Embeddings**: OpenAI (`gpt-4o-mini` + embeddings)
- **Vector Store**: Postgres with `pgvector` (Dockerized)
- **Tools**: Tavily Search API via `httpx` (pluggable)
- **Guardrails**:
  - Input: Pydantic validation (`ResearchRequest`)
  - Output: Reviewer node checks for hallucinations / unsafe content and can loop back to the writer

**Key JD-friendly concepts shown:**

- Agentic AI (tool-using, multi-step graph, reviewer loop)
- RAG & vector stores (pgvector memory with distance threshold)
- Async FastAPI + streaming (SSE)
- Guardrails & eval-style review pass

---

## Project Structure

```text
app/
  api/
    __init__.py
    routes.py           # Mounts versioned API routers
    v1/
      __init__.py
      research.py       # /api/v1/research SSE endpoint
  agent/
    graph.py            # LangGraph workflow: planner -> tool -> ingest -> writer -> reviewer
  core/
    config.py           # Settings (env vars, DB, API keys)
  models/
    schemas.py          # Pydantic models and guardrail types
  services/
    research.py         # Orchestrates memory check + graph run + streaming
  db.py                 # Postgres + pgvector helper functions
  main.py               # FastAPI app factory + startup schema init
Dockerfile
docker-compose.yml
requirements.txt
README.md
```

---

## Running the Stack (Docker Compose)

### 1. Set environment variables

Create a `.env` file in the project root (same level as `docker-compose.yml`):

```bash
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...    # optional but recommended for real web search
```

> Without `TAVILY_API_KEY`, the agent still runs, but the tool node will return a stub message instead of live search results (useful for quick local demos).

### 2. Start Postgres + App

From the project root:

```bash
docker-compose up --build
```

This will:

- Start `marketmind-db` (Postgres + pgvector)
- Start `marketmind-app` (FastAPI + LangGraph)
- Expose the API on `http://localhost:8000`

The app will automatically:

- Initialize the `pgvector` extension
- Create a simple `research_chunks` table with an IVFFLAT index

---

## API: `/api/v1/research` (SSE)

### Request

- **Method**: `POST`
- **URL**: `http://localhost:8000/api/v1/research`
- **Content-Type**: `application/json`

Body:

```json
{
  "topic": "Trends in GenAI for Marketing 2025",
  "depth": "detailed" // or "quick"
}
```

### Response (SSE Stream)

The endpoint uses **Server-Sent Events**. Each event has:

- `event: update`
- `data: <JSON string>`

Two kinds of JSON payloads are streamed:

1. **StepUpdate** – progress messages

   ```json
   {
     "step": "search",
     "message": "Running web search via tools..."
   }
   ```

2. **ResearchResult** – final structured answer

   ```json
   {
     "executive_summary": "...",
     "key_findings": [
       "Finding 1",
       "Finding 2"
     ],
     "citations": [],
     "reused_from_memory": true
   }
   ```

You can consume this in a frontend by listening to SSE and updating the UI as `"step"` messages arrive, then rendering the final `ResearchResult`.

---

## How the Agent Works (LangGraph)

The agent state (`AgentState`) flows through the following nodes:

- **Planner Node**
  - Input: `topic`, `depth`
  - Output: 3 focused search queries about the marketing/brand topic

- **Tool Node**
  - Uses **Tavily API** (via `httpx`) to run web search
  - Normalizes results to `{ query, title, url, content }`

- **Ingest Node**
  - Summarizes the raw results into concise marketing bullets
  - Embeds those bullets via OpenAI embeddings
  - Upserts them into Postgres/pgvector (`research_chunks` table)

- **Writer Node**
  - Uses the current memory context (from vector memory and/or fresh search)
  - Generates:
    - Executive summary for marketers
    - Key findings as bullets

- **Reviewer Node (Guardrail)**
  - Checks the draft answer for:
    - Fabricated metrics / confidential data
    - Non-marketing / unsafe content
  - Returns:
    - `OK` → accept
    - `RETRY` → loop back to Writer node to refine
    - `REJECT` → stop and mark as rejected (simple behavior here)

There is also a **memory check** before the graph:

- Embed the input `topic`
- Query pgvector for chunks with distance `< 0.2`
- If hits are found:
  - Build a memory context from stored bullets
  - Skip planner/tool/ingest and go straight to writer + reviewer
  - Set `reused_from_memory = true` in the final result

This demonstrates **RAG + cost/latency optimization** explicitly.

---

## Local Development (without Docker)

You can also run the app directly on your machine:

1. Ensure you have Postgres running with `pgvector` and matching credentials, or run:

   ```bash
   docker-compose up db
   ```

2. Create and activate a virtualenv, then install deps:

   ```bash
   pip install -r requirements.txt
   ```

3. Set env vars:

   ```bash
   export OPENAI_API_KEY=sk-...
   export TAVILY_API_KEY=tvly-...
   ```

4. Start FastAPI:

   ```bash
   uvicorn app.main:app --reload
   ```

API will be at `http://localhost:8000`.

---

## How to Talk About This Project in an Interview

- **One-liner**:  
  “I built **MarketMind**, an *agentic market research backend* that plans, searches, summarizes, and stores marketing insights in Postgres/pgvector, exposing the whole workflow as an async FastAPI API with SSE.”

- **JD keywords to hit**:
  - **Agentic AI**: LangGraph graph with planner → tools → ingest → writer → reviewer loop.
  - **RAG + Vector Stores**: pgvector memory, distance threshold, reuse of past research to reduce cost/latency.
  - **FastAPI & Async**: Async endpoints, async LLM/tool calls, SSE streaming.
  - **Guardrails/Evals**: Reviewer node that validates outputs and can trigger rewrites.

- **“Why this is relevant to Stagwell / marketing agencies”**:
  - It mimics how an insights team does work: plan queries, scan sources, distill insights, store them as reusable knowledge, and deliver polished executive summaries for brand/strategy conversations.

---

## Sharing on GitHub

> **MarketMind – Agentic Market Research Assistant** (FastAPI + LangGraph + Postgres/pgvector). GitHub: `[https://github.com/nitinkumar-patel/ai-market-mind]`

