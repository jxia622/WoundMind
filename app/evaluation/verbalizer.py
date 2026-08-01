from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from app.agent.config import AgentConfig
from app.evaluation.llm import StructuredOpenAIClient
from app.evaluation.prompts.verbalizer import VERBALIZER_SYSTEM_PROMPT
from app.evaluation.schemas import (
    BlindedVerbalizerContext,
    InitialVerbalization,
    VerbalizerAnswer,
)


TargetedEvidenceProvider = Callable[[str], Awaitable[dict[str, Any]]]


class VerbalizerProtocol(Protocol):
    async def initial(self, context: BlindedVerbalizerContext) -> InitialVerbalization: ...

    async def answer(
        self,
        context: BlindedVerbalizerContext,
        question: str,
    ) -> VerbalizerAnswer: ...


class EvaluationVerbalizer:
    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        llm: StructuredOpenAIClient | None = None,
        targeted_evidence_provider: TargetedEvidenceProvider | None = None,
    ) -> None:
        self.config = config or AgentConfig.from_env()
        self.llm = llm or StructuredOpenAIClient(self.config)
        self.targeted_evidence_provider = targeted_evidence_provider

    async def initial(self, context: BlindedVerbalizerContext) -> InitialVerbalization:
        if self.llm.enabled:
            return await self.llm.generate(
                schema=InitialVerbalization,
                system_prompt=VERBALIZER_SYSTEM_PROMPT,
                payload={
                    "task": "Create the initial concise wound-evidence description.",
                    "case_evidence": context.case_evidence,
                },
                model=self.config.vision_model,
            )
        return self._deterministic_initial(context.case_evidence)

    async def answer(
        self,
        context: BlindedVerbalizerContext,
        question: str,
    ) -> VerbalizerAnswer:
        targeted_evidence: dict[str, Any] = {}
        if self.targeted_evidence_provider is not None:
            targeted_evidence = await self.targeted_evidence_provider(question)

        if self.llm.enabled:
            return await self.llm.generate(
                schema=VerbalizerAnswer,
                system_prompt=VERBALIZER_SYSTEM_PROMPT,
                payload={
                    "task": "Answer the focused wound-specific question.",
                    "question": question,
                    "case_evidence": context.case_evidence,
                    "initial_verbalization": (
                        context.initial_verbalization.model_dump()
                        if context.initial_verbalization
                        else None
                    ),
                    "prior_answers": [item.model_dump() for item in context.prior_answers],
                    "new_targeted_wound_evidence": targeted_evidence,
                },
                model=self.config.vision_model,
            )
        return self._deterministic_answer(question, context.case_evidence, targeted_evidence)

    @staticmethod
    def _deterministic_initial(case_evidence: dict[str, Any]) -> InitialVerbalization:
        visual = case_evidence.get("visual_evidence") or {}
        findings = _string_list(visual.get("findings"))
        observations = _string_list(visual.get("staging_relevant_observations"))
        limitations = _string_list(visual.get("limitations"))

        derived: list[str] = []
        mask = case_evidence.get("mask") or {}
        if mask.get("mask_available"):
            fraction = mask.get("mask_fraction")
            derived.append(
                "A deterministic wound mask is available"
                + (
                    f" and covers {float(fraction):.1%} of the image"
                    if fraction is not None
                    else ""
                )
                + "."
            )
        depth = case_evidence.get("depth") or {}
        if depth.get("depth_available"):
            derived.append("A deterministic relative-depth output is available.")

        direct = _dedupe([*findings, *observations])
        if direct or derived:
            description = " ".join([*direct[:3], *derived[:2]])
        else:
            description = "No wound characteristics are available for independent verbalization."
        unavailable = limitations or (
            ["No direct visual findings were supplied."] if not direct else []
        )
        return InitialVerbalization(
            description=description,
            directly_observed=direct,
            deterministically_derived=derived,
            model_predicted=[],
            unavailable=unavailable,
        )

    @staticmethod
    def _deterministic_answer(
        question: str,
        case_evidence: dict[str, Any],
        targeted_evidence: dict[str, Any],
    ) -> VerbalizerAnswer:
        findings = _dedupe(
            [
                *_string_list(targeted_evidence.get("findings")),
                *_string_list((case_evidence.get("visual_evidence") or {}).get("findings")),
            ]
        )
        answer = targeted_evidence.get("answer")
        if not answer and findings:
            answer = " ".join(findings[:3])
        if answer:
            return VerbalizerAnswer(
                question=question,
                answer=str(answer),
                directly_observed=findings,
                deterministically_derived=[],
                model_predicted=[],
                unavailable=False,
            )
        return VerbalizerAnswer(
            question=question,
            answer="The requested information is unavailable in the supplied wound evidence.",
            directly_observed=[],
            deterministically_derived=[],
            model_predicted=[],
            unavailable=True,
            unavailable_reason=(
                "No supplied wound field or targeted observation answers the question."
            ),
        )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = json.dumps(value, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result
