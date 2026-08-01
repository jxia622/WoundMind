from __future__ import annotations

from typing import Protocol

from app.agent.config import AgentConfig
from app.evaluation.llm import StructuredOpenAIClient
from app.evaluation.prompts.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT
from app.evaluation.schemas import BlindedOrchestratorContext, EvaluationAction


class OrchestratorProtocol(Protocol):
    async def decide(self, context: BlindedOrchestratorContext) -> EvaluationAction: ...


class BlindedOrchestrator:
    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        llm: StructuredOpenAIClient | None = None,
    ) -> None:
        self.config = config or AgentConfig.from_env()
        self.llm = llm or StructuredOpenAIClient(self.config)

    async def decide(self, context: BlindedOrchestratorContext) -> EvaluationAction:
        if self.llm.enabled:
            return await self.llm.generate(
                schema=EvaluationAction,
                system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
                payload={
                    "task": "Choose the next bounded blinded-evaluation action.",
                    "blinded_context": context.model_dump(),
                },
            )
        return self._deterministic_commit(context)

    @staticmethod
    def force_commit(
        action: EvaluationAction,
        context: BlindedOrchestratorContext,
    ) -> EvaluationAction:
        if action.action == "COMMIT_ASSESSMENT":
            return action
        return EvaluationAction(
            action="COMMIT_ASSESSMENT",
            current_condition_hypothesis=action.current_condition_hypothesis,
            current_stage_hypothesis=action.current_stage_hypothesis,
            independent_confidence=action.independent_confidence or 0.0,
            supporting_evidence=action.supporting_evidence,
            contradicting_evidence=action.contradicting_evidence,
            unresolved_questions=_dedupe(
                [
                    *action.unresolved_questions,
                    action.question or "The final requested clarification was not obtained.",
                ]
            ),
            alternative_hypotheses=action.alternative_hypotheses,
            rationale=(
                f"{action.rationale} The iteration cap of {context.max_iterations} was reached; "
                "the best available blinded assessment is being committed."
            ),
        )

    @staticmethod
    def _deterministic_commit(context: BlindedOrchestratorContext) -> EvaluationAction:
        ledger = context.evidence_ledger
        unresolved = list(ledger.unresolved_questions)
        if not unresolved:
            unresolved.append(
                "An independent diagnosis or stage cannot be established without a "
                "configured evaluator model."
            )
        return EvaluationAction(
            action="COMMIT_ASSESSMENT",
            current_condition_hypothesis=ledger.current_condition_hypothesis,
            current_stage_hypothesis=ledger.current_stage_hypothesis,
            independent_confidence=0.0,
            supporting_evidence=ledger.supporting_evidence,
            contradicting_evidence=ledger.contradicting_evidence,
            unresolved_questions=unresolved,
            alternative_hypotheses=ledger.alternative_hypotheses,
            rationale=(
                "No structured evaluator model is configured, so the blinded layer cannot "
                "derive an independent condition or stage from the available evidence."
            ),
        )


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
