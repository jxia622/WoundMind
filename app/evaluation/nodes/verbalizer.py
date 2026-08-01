from __future__ import annotations

from typing import Any

from app.evaluation.nodes.common import append_audit
from app.evaluation.schemas import (
    BlindedVerbalizerContext,
    InitialVerbalization,
    VerbalizerAnswer,
)
from app.evaluation.state import EvaluationState
from app.evaluation.verbalizer import VerbalizerProtocol


class VerbalizerNodes:
    def __init__(self, verbalizer: VerbalizerProtocol) -> None:
        self.verbalizer = verbalizer

    async def initial(self, state: EvaluationState) -> dict[str, Any]:
        context = BlindedVerbalizerContext(case_evidence=state["case_evidence"])
        result = await self.verbalizer.initial(context)
        result.model_predicted = []
        return {
            "initial_verbalization": result.model_dump(),
            "audit_log": append_audit(
                state,
                node="initial_verbalizer",
                details=result.model_dump(),
            ),
        }
    async def targeted(self, state: EvaluationState) -> dict[str, Any]:
        question = str(state.get("next_question") or "").strip()
        if not question:
            raise RuntimeError("Targeted Verbalizer node requires next_question.")
        context = BlindedVerbalizerContext(
            case_evidence=state["case_evidence"],
            initial_verbalization=InitialVerbalization.model_validate(
                state["initial_verbalization"]
            ),
            prior_answers=[
                VerbalizerAnswer.model_validate(item)
                for item in state.get("verbalizer_history", [])
            ],
        )
        answer = await self.verbalizer.answer(context, question)
        answer.model_predicted = []
        history = [*state.get("verbalizer_history", []), answer.model_dump()]
        return {
            "verbalizer_history": history,
            "audit_log": append_audit(
                state,
                node="targeted_verbalizer",
                route="ASK_VERBALIZER",
                question=question,
                details=answer.model_dump(),
            ),
        }
