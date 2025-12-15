## MarketMind – Agentic Market Research Assistant

**MarketMind** is a FastAPI + LangGraph backend that turns an LLM into an **autonomous market research agent** for brands and topics relevant to marketing and communications agencies.

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
- **LLM & Embeddings**: 
  - **OpenAI** (`gpt-4o-mini` + `text-embedding-3-small`) - default
  - **Ollama** (local, free) - supports any model like `llama3.1`, `mistral`, `phi3`, etc.
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

### Option A: Using OpenAI (Default)

#### 1. Set environment variables

Create a `.env` file in the project root (same level as `docker-compose.yml`):

```bash
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...    # optional but recommended for real web search
LLM_PROVIDER=openai
```

> Without `TAVILY_API_KEY`, the agent still runs, but the tool node will return a stub message instead of live search results (useful for quick local demos).

#### 2. Start services

From the project root:

```bash
docker compose up --build
```

This will:

- Start `marketmind-db` (Postgres + pgvector)
- Start `marketmind-app` (FastAPI + LangGraph)
- Expose the API on `http://localhost:8000`

The app will automatically:

- Initialize the `pgvector` extension
- Create a simple `research_chunks` table with an IVFFLAT index

---

### Option B: Using Ollama (Free, Local)

Ollama runs models locally, so you don't need an OpenAI API key or quota. This is perfect for demos and development.

#### 1. Set environment variables

Create a `.env` file:

```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1    # or mistral, phi3, etc.
TAVILY_API_KEY=tvly-...  # optional
```

#### 2. Start all services (including Ollama)

```bash
docker compose up --build
```

This starts:
- `marketmind-ollama` (Ollama server)
- `marketmind-db` (Postgres + pgvector)
- `marketmind-app` (FastAPI + LangGraph)

#### 3. Install a model in Ollama (first time only)

Once containers are running, pull a model into the Ollama container:

```bash
docker compose exec ollama ollama pull llama3.1
```

**Recommended models:**
- `llama3.1` - Good balance of quality and speed (~4.7GB)
- `mistral` - Smaller, faster (~4.1GB)
- `phi3` - Very compact, fast (~2.3GB)
- `llama3.2` - Latest, larger (~2GB for 1B variant)

**Note:** Models are stored in a Docker volume (`ollama_models`), so they persist across container restarts.

#### 4. Verify Ollama is working

Test the Ollama service directly:

```bash
curl http://localhost:11434/api/tags
```

You should see your installed models listed.

#### 5. Use the API

The FastAPI app will automatically use Ollama when `LLM_PROVIDER=ollama`. Test with:

```bash
curl -N \
  -H "Accept: text/event-stream" \
  -H "Content-Type: application/json" \
  -X POST \
  -d '{"topic": "Trends in GenAI for Marketing 2025", "depth": "detailed"}' \
  http://localhost:8000/api/v1/research
```

**Switching between providers:** Just change `LLM_PROVIDER` in `.env` and restart:

```bash
docker compose restart app
```

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
   docker compose up db
   ```

2. Create and activate a virtualenv, then install deps:

   ```bash
   pip install -r requirements.txt
   ```

3. Set env vars:

   **For OpenAI:**
   ```bash
   export OPENAI_API_KEY=sk-...
   export TAVILY_API_KEY=tvly-...
   export LLM_PROVIDER=openai
   ```

   **For Ollama (local):**
   ```bash
   export LLM_PROVIDER=ollama
   export OLLAMA_BASE_URL=http://localhost:11434
   export OLLAMA_MODEL=llama3.1
   export TAVILY_API_KEY=tvly-...
   ```
   
   Then start Ollama locally:
   ```bash
   brew install ollama
   ollama serve
   ollama pull llama3.1
   ```

4. Start FastAPI:

   ```bash
   uvicorn app.main:app --reload
   ```

API will be at `http://localhost:8000`.