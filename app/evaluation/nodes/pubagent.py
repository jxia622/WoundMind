from __future__ import annotations

from typing import Any

from app.evaluation.nodes.common import append_audit
from app.evaluation.pubagent import PubAgentProtocol
from app.evaluation.state import EvaluationState


class PubAgentNode:
    def __init__(self, pubagent: PubAgentProtocol) -> None:
        self.pubagent = pubagent

    async def __call__(self, state: EvaluationState) -> dict[str, Any]:
        question = str(state.get("next_question") or "").strip()
        if not question:
            raise RuntimeError("PubAgent node requires next_question.")
        response = await self.pubagent.ask(question)
        history = [*state.get("pubagent_history", []), response.model_dump()]
        return {
            "pubagent_history": history,
            "audit_log": append_audit(
                state,
                node="pubagent",
                route="ASK_PUBAGENT",
                question=question,
                details=response.model_dump(),
            ),
        }
