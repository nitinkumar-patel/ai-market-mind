from typing import Any, AsyncGenerator, Dict

from app.agent.graph import AgentState, build_graph, embeddings_model
from app.db import query_similar_chunks
from app.models.schemas import ResearchDepth, ResearchRequest, ResearchResult, StepUpdate


graph = build_graph()


async def run_research_with_stream(
    request: ResearchRequest,
) -> AsyncGenerator[str, None]:
    """
    Orchestrate:
    1) Memory check via pgvector
    2) If needed: LangGraph planner -> tool -> ingest -> writer -> reviewer loop
    3) Stream step updates as SSE
    """
    topic = request.topic
    depth = request.depth

    # Step 1: embed query and hit vector memory
    yield StepUpdate(step="memory_check", message="Checking vector memory for existing research...").model_dump_json()
    query_vector = await embeddings_model.aembed_query(topic)
    memory_rows = query_similar_chunks(topic=topic, embedding=query_vector, max_distance=0.2, limit=8)

    initial_state: AgentState = {"topic": topic, "depth": depth}
    reused_from_memory = False

    if memory_rows:
        reused_from_memory = True
        context = "\n".join(f"- {c}" for c, _ in memory_rows)
        initial_state["memory_context"] = context
        initial_state["reused_from_memory"] = True
        yield StepUpdate(step="memory_hit", message="Found relevant prior research in memory.").model_dump_json()
    else:
        yield StepUpdate(step="memory_miss", message="No relevant memory; triggering fresh web research.").model_dump_json()

    # If we already have strong memory, skip planner/tool/ingest and just write + review
    if reused_from_memory:
        async for event in graph.astream(initial_state, config={"configurable": {"thread_id": "writer_only"}}):
            # LangGraph streaming gives node-by-node updates; we only care about final state
            pass
        final_state: Dict[str, Any] = await graph.ainvoke(initial_state)
    else:
        yield StepUpdate(step="planner", message="Planning focused search queries...").model_dump_json()
        yield StepUpdate(step="search", message="Running web search via tools...").model_dump_json()
        yield StepUpdate(step="ingest", message="Summarizing and storing findings into vector memory...").model_dump_json()
        yield StepUpdate(step="writer", message="Drafting executive summary for marketers...").model_dump_json()

        final_state: Dict[str, Any] = await graph.ainvoke(initial_state)

    draft: ResearchResult = final_state["draft_answer"]
    reused_flag = bool(final_state.get("reused_from_memory", reused_from_memory))
    draft.reused_from_memory = reused_flag

    yield StepUpdate(step="review", message="Running guardrail review on the answer...").model_dump_json()

    yield draft.model_dump_json()


