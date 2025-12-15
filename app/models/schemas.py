from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ResearchDepth(str, Enum):
    quick = "quick"
    detailed = "detailed"


class ResearchRequest(BaseModel):
    topic: str = Field(..., description="Brand or topic to research, e.g. 'GenAI in Marketing 2025'.")
    depth: ResearchDepth = Field(
        default=ResearchDepth.detailed,
        description="How deep the research should go (affects number of searches and summarization detail).",
    )


class StepUpdate(BaseModel):
    step: str
    message: str


class Citation(BaseModel):
    source_url: str
    title: Optional[str] = None


class ResearchResult(BaseModel):
    executive_summary: str
    key_findings: List[str]
    citations: List[Citation]
    reused_from_memory: bool = False


class GuardrailStatus(str, Enum):
    ok = "ok"
    retry = "retry"
    reject = "reject"


class GuardrailReport(BaseModel):
    status: GuardrailStatus
    issues: List[str] = []


