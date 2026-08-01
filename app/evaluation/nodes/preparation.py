from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.evaluation.constants import MAX_EVALUATION_ITERATIONS
from app.evaluation.nodes.common import append_audit
from app.evaluation.schemas import HiddenModelOutput
from app.evaluation.state import EvaluationState


_EVIDENCE_TOP_LEVEL_KEYS = {
    "validation",
    "mask",
    "depth",
    "visual_evidence",
    "measurements",
    "image_features",
    "wound_features",
    "derived_features",
}
_HIDDEN_KEY_FRAGMENTS = {
    "condition",
    "classification",
    "stage",
    "severity",
    "confidence",
    "probability",
    "probabilities",
    "prediction",
    "predicted",
    "candidate",
}


async def prepare_evaluation_state(state: EvaluationState) -> dict[str, Any]:
    pipeline_output = deepcopy(state.get("pipeline_output", {}))
    hidden = extract_hidden_model_output(pipeline_output)
    supplied = deepcopy(state.get("supplied_case_evidence", {}))
    case_evidence = (
        scrub_blinded_evidence(supplied)
        if supplied
        else extract_case_evidence(pipeline_output)
    )
    prepared: dict[str, Any] = {
        "case_evidence": case_evidence,
        "hidden_model_output": hidden.model_dump(),
        "iteration_count": 0,
        "max_iterations": min(
            MAX_EVALUATION_ITERATIONS,
            max(1, int(state.get("max_iterations", MAX_EVALUATION_ITERATIONS))),
        ),
        "supporting_evidence": [],
        "contradicting_evidence": [],
        "unresolved_questions": [],
        "alternative_hypotheses": [],
        "verbalizer_history": [],
        "pubagent_history": [],
        "assessment_committed": False,
        "model_revealed": False,
        "audit_log": [],
    }
    audit_state = {**state, **prepared}
    prepared["audit_log"] = append_audit(
        audit_state,
        node="prepare_evaluation_state",
        details={
            "case_evidence_fields": sorted(case_evidence),
            "hidden_fields": [
                "model_condition",
                "model_stage",
                "model_confidence",
            ],
            "blinding_active": True,
        },
    )
    return prepared


def extract_hidden_model_output(pipeline_output: dict[str, Any]) -> HiddenModelOutput:
    condition_payload = _as_dict(
        pipeline_output.get("condition")
        or pipeline_output.get("classification")
        or pipeline_output.get("classification_result")
    )
    severity_payload = _as_dict(
        pipeline_output.get("severity")
        or pipeline_output.get("severity_result")
    )

    condition = _first_string(
        condition_payload.get("model_top1_label"),
        condition_payload.get("top1_label"),
        condition_payload.get("selected_condition"),
        condition_payload.get("condition"),
        pipeline_output.get("model_condition"),
        (
            pipeline_output.get("condition")
            if isinstance(pipeline_output.get("condition"), str)
            else None
        ),
    )
    stage = _first_string(
        severity_payload.get("severity_prediction"),
        severity_payload.get("stage"),
        pipeline_output.get("model_stage"),
        pipeline_output.get("severity_stage"),
    )
    condition_confidence = _first_float(
        condition_payload.get("confidence"),
        condition_payload.get("condition_confidence"),
        pipeline_output.get("condition_confidence"),
    )
    stage_confidence = _first_float(
        severity_payload.get("severity_confidence"),
        severity_payload.get("confidence"),
        pipeline_output.get("severity_confidence"),
        pipeline_output.get("model_confidence"),
    )
    return HiddenModelOutput(
        condition=condition,
        stage=stage,
        confidence=stage_confidence if stage_confidence is not None else condition_confidence,
        condition_confidence=condition_confidence,
        stage_confidence=stage_confidence,
    )


def extract_case_evidence(pipeline_output: dict[str, Any]) -> dict[str, Any]:
    evidence = {
        key: deepcopy(value)
        for key, value in pipeline_output.items()
        if key in _EVIDENCE_TOP_LEVEL_KEYS
    }
    return scrub_blinded_evidence(evidence)


def scrub_blinded_evidence(value: Any) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if any(fragment in normalized for fragment in _HIDDEN_KEY_FRAGMENTS):
                continue
            clean[str(key)] = scrub_blinded_evidence(item)
        return clean
    if isinstance(value, list):
        return [scrub_blinded_evidence(item) for item in value]
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_string(*values: Any) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value)
    return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None
