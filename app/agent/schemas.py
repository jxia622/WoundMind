from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

VerifierStatus = Literal["PASS", "FAIL", "UNCERTAIN"]


class DocumentChunk(BaseModel):
    text: str
    source: str
    doc_id: str | None = None
    section: str | None = None
    condition_tag: str = "general"
    score: float | None = None


class VerifierResult(BaseModel):
    result: VerifierStatus
    citation: str | None = None
    flag: str | None = None
    needs_more_evidence: bool = False
    followup_query: str | None = None
    reasoning: str | None = None


class QAExchange(BaseModel):
    question: str
    answer: str


class AgentTraceStep(BaseModel):
    step: int
    tool: str
    inputs_summary: dict[str, Any] = Field(default_factory=dict)
    outputs_summary: dict[str, Any] = Field(default_factory=dict)


class ClinicalOutput(BaseModel):
    case_id: str
    condition: str | None = None
    condition_confidence: float | None = None
    severity_stage: str | None = None
    severity_confidence: float | None = None
    recommendation: str
    verifier_result: VerifierStatus
    citation: str | None = None
    flags: list[str] = Field(default_factory=list)
    visual_evidence: dict[str, Any] = Field(default_factory=dict)
    evaluation_summary: dict[str, Any] = Field(default_factory=dict)
    qa_exchanges: list[QAExchange] = Field(default_factory=list)
    agent_trace: list[AgentTraceStep] = Field(default_factory=list)
    model_variant_used: str | None = None
    artifact_path: str | None = None
    disclaimer: str


class AgentAnalyzeResponse(BaseModel):
    case_id: str
    output: ClinicalOutput
    retrieved_chunks: list[DocumentChunk] = Field(default_factory=list)
