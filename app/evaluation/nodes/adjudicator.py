from __future__ import annotations

import hashlib
import json
from typing import Any

from app.evaluation.adjudicator import AdjudicatorProtocol
from app.evaluation.nodes.common import append_audit
from app.evaluation.schemas import (
    AdjudicationContext,
    ComparisonResult,
    EvidenceLedger,
    EvaluationEvidence,
    EvaluationResult,
    FinalEvaluation,
    HiddenModelOutput,
    IndependentAssessment,
    PubAgentResponse,
)
from app.evaluation.state import EvaluationState


async def commit_independent_assessment(state: EvaluationState) -> dict[str, Any]:
    if state.get("assessment_committed"):
        raise RuntimeError("Independent assessment has already been committed.")
    action = state.get("pending_action", {})
    if action.get("action") != "COMMIT_ASSESSMENT":
        raise RuntimeError("Commit node requires a COMMIT_ASSESSMENT action.")
    assessment = IndependentAssessment(
        condition=action.get("current_condition_hypothesis"),
        stage=action.get("current_stage_hypothesis"),
        confidence=float(action.get("independent_confidence") or 0.0),
        rationale=str(action.get("rationale") or "Best available blinded assessment."),
        key_supporting_evidence=state.get("supporting_evidence", []),
        key_contradicting_evidence=state.get("contradicting_evidence", []),
        remaining_uncertainties=state.get("unresolved_questions", []),
        clinical_evidence_used=[
            PubAgentResponse.model_validate(item)
            for item in state.get("pubagent_history", [])
        ],
    )
    payload = assessment.model_dump()
    digest = _digest(payload)
    update = {
        "independent_assessment": payload,
        "independent_assessment_digest": digest,
        "assessment_committed": True,
    }
    update["audit_log"] = append_audit(
        {**state, **update},
        node="commit_independent_assessment",
        route="COMMIT_ASSESSMENT",
        details={"assessment": payload, "assessment_digest": digest},
    )
    return update


async def reveal_model_output(state: EvaluationState) -> dict[str, Any]:
    if not state.get("assessment_committed") or not state.get("independent_assessment"):
        raise RuntimeError("Model output cannot be revealed before independent assessment commit.")
    _assert_assessment_locked(state)
    hidden = HiddenModelOutput.model_validate(state["hidden_model_output"])
    update = {
        "model_revealed": True,
        "model_condition": hidden.condition,
        "model_stage": hidden.stage,
        "model_confidence": hidden.confidence,
    }
    update["audit_log"] = append_audit(
        {**state, **update},
        node="reveal_model_output",
        details={"model_output": hidden.model_dump(), "revealed_after_commit": True},
    )
    return update


class AdjudicatorNode:
    def __init__(self, adjudicator: AdjudicatorProtocol) -> None:
        self.adjudicator = adjudicator

    async def __call__(self, state: EvaluationState) -> dict[str, Any]:
        if not state.get("model_revealed"):
            raise RuntimeError("Adjudication requires revealed model output.")
        _assert_assessment_locked(state)
        context = AdjudicationContext(
            independent_assessment=IndependentAssessment.model_validate(
                state["independent_assessment"]
            ),
            model_output=HiddenModelOutput.model_validate(state["hidden_model_output"]),
            evidence_ledger=_ledger(state),
        )
        decision = await self.adjudicator.adjudicate(context)
        _assert_assessment_locked(state)
        comparison = ComparisonResult(
            condition_agreement=decision.condition_agreement,
            stage_agreement=decision.stage_agreement,
            evidence_consistent=decision.evidence_consistent,
            clinically_meaningful_disagreement=decision.clinically_meaningful_disagreement,
            discrepancies=decision.discrepancies,
        )
        final = FinalEvaluation(
            status=decision.status,
            confidence=decision.confidence,
            rationale=decision.rationale,
        )
        update = {
            "comparison": comparison.model_dump(),
            "final_evaluation": final.model_dump(),
        }
        update["audit_log"] = append_audit(
            {**state, **update},
            node="comparison_adjudication",
            details={
                "comparison": comparison.model_dump(),
                "final_evaluation": final.model_dump(),
            },
        )
        return update


async def finalize_evaluation(state: EvaluationState) -> dict[str, Any]:
    _assert_assessment_locked(state)
    final_audit = append_audit(
        state,
        node="finalize_evaluation",
        details={"status": state["final_evaluation"]["status"]},
    )
    result = EvaluationResult(
        case_id=state["case_id"],
        model_output=HiddenModelOutput.model_validate(state["hidden_model_output"]),
        independent_evaluation=IndependentAssessment.model_validate(
            state["independent_assessment"]
        ),
        comparison=ComparisonResult.model_validate(state["comparison"]),
        evidence=EvaluationEvidence(
            supporting=state.get("supporting_evidence", []),
            contradicting=state.get("contradicting_evidence", []),
            clinical_evidence=[
                PubAgentResponse.model_validate(item)
                for item in state.get("pubagent_history", [])
            ],
            unresolved=state.get("unresolved_questions", []),
        ),
        final_evaluation=FinalEvaluation.model_validate(state["final_evaluation"]),
        iterations=int(state["iteration_count"]),
        max_iterations=int(state["max_iterations"]),
        verbalizer_history=state.get("verbalizer_history", []),
        pubagent_history=state.get("pubagent_history", []),
        audit_log=final_audit,
    )
    return {"audit_log": final_audit, "final_result": result.model_dump()}


def _ledger(state: EvaluationState) -> EvidenceLedger:
    return EvidenceLedger(
        current_condition_hypothesis=state.get("current_condition_hypothesis"),
        current_stage_hypothesis=state.get("current_stage_hypothesis"),
        supporting_evidence=state.get("supporting_evidence", []),
        contradicting_evidence=state.get("contradicting_evidence", []),
        clinical_evidence=[
            PubAgentResponse.model_validate(item)
            for item in state.get("pubagent_history", [])
        ],
        unresolved_questions=state.get("unresolved_questions", []),
        alternative_hypotheses=state.get("alternative_hypotheses", []),
        next_action=state.get("next_action"),
        next_question=state.get("next_question"),
    )


def _assert_assessment_locked(state: EvaluationState) -> None:
    expected = state.get("independent_assessment_digest")
    assessment = state.get("independent_assessment")
    if not expected or not assessment or _digest(assessment) != expected:
        raise RuntimeError("Committed independent assessment was modified after locking.")


def _digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
