from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.config import AgentConfig
from app.evaluation.adjudicator import AdjudicatorProtocol, EvaluationAdjudicator
from app.evaluation.constants import MAX_EVALUATION_ITERATIONS
from app.evaluation.nodes.adjudicator import (
    AdjudicatorNode,
    commit_independent_assessment,
    finalize_evaluation,
    reveal_model_output,
)
from app.evaluation.nodes.orchestrator import OrchestratorNode, route_orchestrator_action
from app.evaluation.nodes.preparation import prepare_evaluation_state
from app.evaluation.nodes.pubagent import PubAgentNode
from app.evaluation.nodes.verbalizer import VerbalizerNodes
from app.evaluation.orchestrator import BlindedOrchestrator, OrchestratorProtocol
from app.evaluation.pubagent import PubAgentAdapter, PubAgentProtocol
from app.evaluation.schemas import EvaluationResult
from app.evaluation.state import EvaluationState
from app.evaluation.verbalizer import EvaluationVerbalizer, VerbalizerProtocol


class EvaluationWorkflow:
    def __init__(
        self,
        config: AgentConfig | None = None,
        *,
        verbalizer: VerbalizerProtocol | None = None,
        orchestrator: OrchestratorProtocol | None = None,
        pubagent: PubAgentProtocol | None = None,
        adjudicator: AdjudicatorProtocol | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self.config = config or AgentConfig.from_env()
        self.verbalizer = verbalizer or EvaluationVerbalizer(self.config)
        self.orchestrator = orchestrator or BlindedOrchestrator(self.config)
        self.pubagent = pubagent or PubAgentAdapter(self.config)
        self.adjudicator = adjudicator or EvaluationAdjudicator(self.config)
        self.checkpointer = checkpointer or InMemorySaver()
        self.graph = self._build_graph().compile(checkpointer=self.checkpointer)

    def _build_graph(self) -> StateGraph:
        verbalizer_nodes = VerbalizerNodes(self.verbalizer)
        builder = StateGraph(EvaluationState)
        builder.add_node("prepare_evaluation_state", prepare_evaluation_state)
        builder.add_node("initial_verbalizer", verbalizer_nodes.initial)
        builder.add_node("blinded_orchestrator", OrchestratorNode(self.orchestrator))
        builder.add_node("targeted_verbalizer", verbalizer_nodes.targeted)
        builder.add_node("pubagent", PubAgentNode(self.pubagent))
        builder.add_node("commit_independent_assessment", commit_independent_assessment)
        builder.add_node("reveal_model_output", reveal_model_output)
        builder.add_node("comparison_adjudication", AdjudicatorNode(self.adjudicator))
        builder.add_node("finalize_evaluation", finalize_evaluation)

        builder.add_edge(START, "prepare_evaluation_state")
        builder.add_edge("prepare_evaluation_state", "initial_verbalizer")
        builder.add_edge("initial_verbalizer", "blinded_orchestrator")
        builder.add_conditional_edges(
            "blinded_orchestrator",
            route_orchestrator_action,
            {
                "verbalizer": "targeted_verbalizer",
                "pubagent": "pubagent",
                "commit": "commit_independent_assessment",
            },
        )
        builder.add_edge("targeted_verbalizer", "blinded_orchestrator")
        builder.add_edge("pubagent", "blinded_orchestrator")
        builder.add_edge("commit_independent_assessment", "reveal_model_output")
        builder.add_edge("reveal_model_output", "comparison_adjudication")
        builder.add_edge("comparison_adjudication", "finalize_evaluation")
        builder.add_edge("finalize_evaluation", END)
        return builder

    async def run(
        self,
        *,
        case_id: str,
        pipeline_output: dict[str, Any],
        case_evidence: dict[str, Any] | None = None,
        max_iterations: int | None = None,
    ) -> EvaluationResult:
        initial_state: EvaluationState = {
            "case_id": case_id,
            "pipeline_output": pipeline_output,
            "supplied_case_evidence": case_evidence or {},
            "max_iterations": min(
                MAX_EVALUATION_ITERATIONS,
                max(1, max_iterations or self.config.max_evaluation_iterations),
            ),
        }
        final_state = await self.graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": case_id}},
        )
        return EvaluationResult.model_validate(final_state["final_result"])

    def get_state(self, case_id: str):
        return self.graph.get_state({"configurable": {"thread_id": case_id}})


def build_evaluation_graph(**kwargs) -> EvaluationWorkflow:
    return EvaluationWorkflow(**kwargs)
