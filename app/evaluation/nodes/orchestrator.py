from __future__ import annotations

from typing import Any

from app.evaluation.nodes.common import append_audit
from app.evaluation.orchestrator import BlindedOrchestrator, OrchestratorProtocol
from app.evaluation.schemas import (
    BlindedOrchestratorContext,
    EvidenceLedger,
    InitialVerbalization,
    PubAgentResponse,
    VerbalizerAnswer,
)
from app.evaluation.state import EvaluationState


class OrchestratorNode:
    def __init__(self, orchestrator: OrchestratorProtocol) -> None:
        self.orchestrator = orchestrator

    async def __call__(self, state: EvaluationState) -> dict[str, Any]:
        iteration = int(state.get("iteration_count", 0)) + 1
        context = _blinded_context(state, iteration)
        action = await self.orchestrator.decide(context)
        if iteration >= context.max_iterations and action.action != "COMMIT_ASSESSMENT":
            action = BlindedOrchestrator.force_commit(action, context)

        update = {
            "iteration_count": iteration,
            "supporting_evidence": _dedupe(
                [*state.get("supporting_evidence", []), *action.supporting_evidence]
            ),
            "contradicting_evidence": _dedupe(
                [
                    *state.get("contradicting_evidence", []),
                    *action.contradicting_evidence,
                ]
            ),
            "unresolved_questions": _dedupe(action.unresolved_questions),
            "alternative_hypotheses": _dedupe(
                [
                    *state.get("alternative_hypotheses", []),
                    *action.alternative_hypotheses,
                ]
            ),
            "current_condition_hypothesis": action.current_condition_hypothesis,
            "current_stage_hypothesis": action.current_stage_hypothesis,
            "next_action": action.action,
            "next_question": action.question,
            "pending_action": action.model_dump(),
        }
        update["audit_log"] = append_audit(
            {**state, **update},
            node="blinded_orchestrator",
            route=action.action,
            question=action.question,
            details={
                "current_condition_hypothesis": action.current_condition_hypothesis,
                "current_stage_hypothesis": action.current_stage_hypothesis,
                "supporting_evidence": action.supporting_evidence,
                "contradicting_evidence": action.contradicting_evidence,
                "unresolved_questions": action.unresolved_questions,
                "alternative_hypotheses": action.alternative_hypotheses,
                "rationale": action.rationale,
            },
        )
        return update


def route_orchestrator_action(state: EvaluationState) -> str:
    action = state.get("next_action")
    if action == "ASK_VERBALIZER":
        return "verbalizer"
    if action == "ASK_PUBAGENT":
        return "pubagent"
    if action == "COMMIT_ASSESSMENT":
        return "commit"
    raise RuntimeError(f"Unknown orchestrator action: {action!r}")


def _blinded_context(
    state: EvaluationState,
    iteration: int,
) -> BlindedOrchestratorContext:
    return BlindedOrchestratorContext(
        case_evidence=state["case_evidence"],
        initial_verbalization=InitialVerbalization.model_validate(
            state["initial_verbalization"]
        ),
        verbalizer_history=[
            VerbalizerAnswer.model_validate(item)
            for item in state.get("verbalizer_history", [])
        ],
        pubagent_history=[
            PubAgentResponse.model_validate(item)
            for item in state.get("pubagent_history", [])
        ],
        evidence_ledger=EvidenceLedger(
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
        ),
        iteration=iteration,
        max_iterations=int(state["max_iterations"]),
    )


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))
