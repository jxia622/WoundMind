from __future__ import annotations

from typing import Any, TypedDict


class EvaluationState(TypedDict, total=False):
    case_id: str
    pipeline_output: dict[str, Any]
    supplied_case_evidence: dict[str, Any]
    case_evidence: dict[str, Any]
    hidden_model_output: dict[str, Any]
    initial_verbalization: dict[str, Any]
    iteration_count: int
    max_iterations: int
    supporting_evidence: list[str]
    contradicting_evidence: list[str]
    unresolved_questions: list[str]
    alternative_hypotheses: list[str]
    verbalizer_history: list[dict[str, Any]]
    pubagent_history: list[dict[str, Any]]
    current_condition_hypothesis: str | None
    current_stage_hypothesis: str | None
    next_action: str | None
    next_question: str | None
    pending_action: dict[str, Any]
    independent_assessment: dict[str, Any]
    independent_assessment_digest: str
    assessment_committed: bool
    model_revealed: bool
    model_condition: str | None
    model_stage: str | None
    model_confidence: float | None
    comparison: dict[str, Any]
    final_evaluation: dict[str, Any]
    final_result: dict[str, Any]
    audit_log: list[dict[str, Any]]
