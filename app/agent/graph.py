from typing import Any, Dict, List, Optional, TypedDict

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langgraph.graph import END, StateGraph

from app.core.config import settings
from app.db import query_similar_chunks, upsert_chunks
from app.models.schemas import GuardrailReport, GuardrailStatus, ResearchDepth, ResearchResult


class AgentState(TypedDict, total=False):
    topic: str
    depth: ResearchDepth
    search_queries: List[str]
    search_results: List[Dict[str, Any]]
    memory_context: str
    draft_answer: str
    guardrail_report: GuardrailReport
    reused_from_memory: bool


if settings.llm_provider == "ollama":
    chat_model = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.2,
    )
    embeddings_model = OllamaEmbeddings(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
    )
else:
    embeddings_model = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=settings.openai_api_key)


async def planner_node(state: AgentState) -> AgentState:
    topic = state["topic"]
    depth = state.get("depth", ResearchDepth.detailed)
    depth_hint = "high-level overview" if depth == ResearchDepth.quick else "detailed multi-angle analysis"

    system = SystemMessage(
        content="You are a market research planner. Break a topic into 3 focused web search queries for marketing insights."
    )
    human = HumanMessage(
        content=f"Topic: {topic}\nDepth: {depth_hint}\nReturn EXACTLY 3 search queries, one per line."
    )
    resp = await chat_model.ainvoke([system, human])
    lines = [l.strip("- ").strip() for l in resp.content.splitlines() if l.strip()]
    queries = lines[:3] if len(lines) >= 3 else lines

    return {**state, "search_queries": queries}


async def _tavily_search(query: str) -> List[Dict[str, Any]]:
    if not settings.tavily_api_key:
        return [
            {
                "query": query,
                "title": "Tavily API key not configured",
                "url": "https://tavily.com",
                "content": "Tavily API key is missing. Configure TAVILY_API_KEY to enable real web search.",
            }
        ]
    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "max_results": 5,
        "search_depth": "advanced",
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post("https://api.tavily.com/search", json=payload)
        r.raise_for_status()
        data = r.json()
        return data.get("results", [])


async def tool_node(state: AgentState) -> AgentState:
    queries = state.get("search_queries", [])
    all_results: List[Dict[str, Any]] = []
    for q in queries:
        results = await _tavily_search(q)
        for r in results:
            all_results.append(
                {
                    "query": q,
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "content": r.get("content"),
                }
            )
    return {**state, "search_results": all_results}


async def ingest_node(state: AgentState) -> AgentState:
    """
    Summarize raw search results into compact chunks, embed and upsert into pgvector.
    """
    topic = state["topic"]
    results = state.get("search_results", [])
    if not results:
        return state

    joined = "\n\n".join(
        [f"Title: {r.get('title')}\nURL: {r.get('url')}\nContent: {r.get('content')}" for r in results]
    )

    system = SystemMessage(
        content=(
            "You condense raw web results into 5-10 concise factual bullet points about marketing insights. "
            "Focus on brand positioning, consumer behavior, competitive landscape, and channel trends."
        )
    )
    human = HumanMessage(
        content=f"Topic: {topic}\n\nRaw results:\n{joined}\n\nReturn bullet points, each on its own line."
    )
    resp = await chat_model.ainvoke([system, human])
    bullets = [b.strip("- ").strip() for b in resp.content.splitlines() if b.strip()]

    # Embed and store
    vectors = await embeddings_model.aembed_documents(bullets)
    chunks = []
    for bullet, vec in zip(bullets, vectors):
        chunks.append((bullet, None, vec))

    upsert_chunks(topic, chunks)

    memory_context = "\n".join(f"- {b}" for b in bullets)
    return {**state, "memory_context": memory_context, "reused_from_memory": False}


async def writer_node(state: AgentState) -> AgentState:
    topic = state["topic"]
    depth = state.get("depth", ResearchDepth.detailed)
    memory_context = state.get("memory_context", "")
    reused = state.get("reused_from_memory", False)

    depth_instructions = (
        "Be concise but insightful (3–4 key points)."
        if depth == ResearchDepth.quick
        else "Provide a detailed executive summary and at least 5–7 key findings."
    )

    system = SystemMessage(
        content=(
            "You are a senior marketing strategist at a global agency (like Stagwell). "
            "You write structured, executive-ready research summaries."
        )
    )
    human = HumanMessage(
        content=(
            f"Topic: {topic}\n"
            f"Context (from vector memory and/or fresh search):\n{memory_context}\n\n"
            f"Write an executive summary focused on:\n"
            f"- Market and consumer trends\n"
            f"- Implications for brand strategy and media\n"
            f"- Opportunities and risks for marketers\n\n"
            f"{depth_instructions}\n"
            f"Return sections: Executive Summary paragraph + bullet-point key findings."
        )
    )
    resp = await chat_model.ainvoke([system, human])

    content = resp.content
    # Simple split: first paragraph vs bullets
    parts = content.split("\n\n", 1)
    executive_summary = parts[0].strip()
    key_lines: List[str] = []
    if len(parts) > 1:
        for line in parts[1].splitlines():
            if line.strip():
                key_lines.append(line.strip("-• ").strip())

    draft = ResearchResult(
        executive_summary=executive_summary,
        key_findings=key_lines,
        citations=[],  # could be mapped from search_results in a richer version
        reused_from_memory=reused,
    )

    return {**state, "draft_answer": draft}


async def reviewer_node(state: AgentState) -> AgentState:
    """
    Lightweight guardrail: check for obvious hallucinations or unsafe content.
    """
    draft: ResearchResult = state["draft_answer"]
    text = draft.executive_summary + "\n" + "\n".join(draft.key_findings)

    system = SystemMessage(
        content=(
            "You are a safety and quality checker for market research outputs.\n"
            "If the answer looks factual, marketing-focused, and does not fabricate specific internal data, "
            "respond with 'OK'.\n"
            "If it appears to make up metrics, confidential data, or non-existent brands, respond with 'RETRY' "
            "and briefly list the issues.\n"
            "If it is clearly unsafe or unrelated to marketing, respond with 'REJECT' and briefly list the issues."
        )
    )
    human = HumanMessage(content=f"Answer to review:\n{text}")
    resp = await chat_model.ainvoke([system, human])
    raw = resp.content.strip()

    status: GuardrailStatus
    issues: List[str] = []
    if raw.upper().startswith("OK"):
        status = GuardrailStatus.ok
    elif raw.upper().startswith("REJECT"):
        status = GuardrailStatus.reject
        issues = [raw]
    else:
        status = GuardrailStatus.retry
        issues = [raw]

    report = GuardrailReport(status=status, issues=issues)
    return {**state, "guardrail_report": report}


def _router_fn(state: AgentState) -> str:
    report: Optional[GuardrailReport] = state.get("guardrail_report")
    if not report:
        return "writer"
    if report.status == GuardrailStatus.ok:
        return END
    if report.status == GuardrailStatus.reject:
        return END
    # retry
    return "writer"


def build_graph() -> Any:
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("tool", tool_node)
    graph.add_node("ingest", ingest_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "tool")
    graph.add_edge("tool", "ingest")
    graph.add_edge("ingest", "writer")
    graph.add_edge("writer", "reviewer")

    graph.add_conditional_edges("reviewer", _router_fn, {"writer": "writer", END: END})

    return graph.compile()


