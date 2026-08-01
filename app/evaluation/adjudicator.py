from __future__ import annotations

import re
from typing import Protocol

from app.agent.config import AgentConfig
from app.evaluation.llm import StructuredOpenAIClient
from app.evaluation.prompts.adjudicator import ADJUDICATOR_SYSTEM_PROMPT
from app.evaluation.schemas import AdjudicationContext, AdjudicationDecision


class AdjudicatorProtocol(Protocol):
    async def adjudicate(self, context: AdjudicationContext) -> AdjudicationDecision: ...


class EvaluationAdjudicator:
    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        llm: StructuredOpenAIClient | None = None,
    ) -> None:
        self.config = config or AgentConfig.from_env()
        self.llm = llm or StructuredOpenAIClient(self.config)

    async def adjudicate(self, context: AdjudicationContext) -> AdjudicationDecision:
        if self.llm.enabled:
            return await self.llm.generate(
                schema=AdjudicationDecision,
                system_prompt=ADJUDICATOR_SYSTEM_PROMPT,
                payload={
                    "task": (
                        "Compare the locked independent assessment with the revealed "
                        "model output."
                    ),
                    "comparison_context": context.model_dump(),
                },
            )
        return self._deterministic_adjudication(context)

    @staticmethod
    def _deterministic_adjudication(context: AdjudicationContext) -> AdjudicationDecision:
        independent = context.independent_assessment
        model = context.model_output
        if not independent.condition or not model.condition:
            return AdjudicationDecision(
                status="INSUFFICIENT_EVIDENCE",
                confidence=0.0,
                rationale="A condition was not available in both assessments for comparison.",
                discrepancies=[],
            )

        condition_agreement = _normalize(independent.condition) == _normalize(model.condition)
        stage_agreement = _optional_agreement(independent.stage, model.stage)
        discrepancies: list[str] = []
        if not condition_agreement:
            discrepancies.append(
                f"Independent condition '{independent.condition}' differs from model "
                f"condition '{model.condition}'."
            )
        if stage_agreement is False:
            discrepancies.append(
                f"Independent stage '{independent.stage}' differs from model stage '{model.stage}'."
            )

        agreement_is_complete = condition_agreement and stage_agreement is not False
        status = "SUPPORTED" if agreement_is_complete else "FLAGGED"
        confidence = min(1.0, max(0.0, independent.confidence))
        return AdjudicationDecision(
            condition_agreement=condition_agreement,
            stage_agreement=stage_agreement,
            discrepancies=discrepancies,
            evidence_consistent=agreement_is_complete,
            clinically_meaningful_disagreement=not agreement_is_complete,
            status=status,
            confidence=confidence,
            rationale=(
                "The committed independent condition and stage are materially "
                "consistent with the model output."
                if status == "SUPPORTED"
                else (
                    "The committed independent assessment materially differs from "
                    "the model output."
                )
            ),
        )


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _optional_agreement(left: str | None, right: str | None) -> bool | None:
    if left is None and right is None:
        return None
    if left is None or right is None:
        return False
    return _normalize(left) == _normalize(right)
