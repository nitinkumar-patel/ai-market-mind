from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from app.models.schemas import ResearchRequest
from app.services.research import run_research_with_stream

router = APIRouter()


@router.post("/research", response_class=StreamingResponse)
async def research(request: ResearchRequest) -> EventSourceResponse:
    """
    SSE endpoint:
    - Emits intermediate StepUpdate events (\"Searching...\", \"Summarizing...\", etc.)
    - Ends with a final ResearchResult payload as JSON.
    """
    if not request.topic.strip():
        raise HTTPException(status_code=400, detail="Topic must not be empty.")

    async def event_generator() -> AsyncGenerator[dict, None]:
        async for payload in run_research_with_stream(request):
            # Payloads are JSON-serialized Pydantic models.
            yield {"event": "update", "data": payload}

    return EventSourceResponse(event_generator())


