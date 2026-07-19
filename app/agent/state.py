from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from app.agent.schemas import AgentTraceStep, DocumentChunk, QAExchange, VerifierResult


@dataclass
class AgentState:
    case_id: str
    image_path: str | None = None
    patient_context: dict[str, Any] = field(default_factory=dict)
    doc_chunks: list[DocumentChunk] = field(default_factory=list)
    validation_result: dict[str, Any] | None = None
    depth_result: dict[str, Any] | None = None
    mask_result: dict[str, Any] | None = None
    selected_mask: dict[str, Any] | None = None
    classification_result: dict[str, Any] | None = None
    condition_policy: dict[str, Any] | None = None
    tool_plan: dict[str, Any] | None = None
    severity_result: dict[str, Any] | None = None
    visual_evidence: dict[str, Any] | None = None
    qa_history: list[QAExchange] = field(default_factory=list)
    draft_assessment: dict[str, Any] | None = None
    evaluation_summary: dict[str, Any] | None = None
    verifier_result: VerifierResult | None = None
    trace: list[AgentTraceStep] = field(default_factory=list)
    iteration_count: int = 0
    fail_count: int = 0
    input_image: Image.Image | None = None
    mask_image: Image.Image | None = None
    selected_mask_image: Image.Image | None = None
    depth_preview_image: Image.Image | None = None

    def add_trace(
        self,
        tool: str,
        inputs_summary: dict[str, Any] | None = None,
        outputs_summary: dict[str, Any] | None = None,
    ) -> None:
        self.trace.append(
            AgentTraceStep(
                step=len(self.trace) + 1,
                tool=tool,
                inputs_summary=inputs_summary or {},
                outputs_summary=outputs_summary or {},
            )
        )
