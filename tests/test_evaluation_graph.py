from __future__ import annotations

import asyncio
from pathlib import Path

from app.agent.config import AgentConfig
from app.evaluation.adjudicator import EvaluationAdjudicator
from app.evaluation.graph import EvaluationWorkflow
from app.evaluation.pubagent import PubAgentAdapter
from app.evaluation.schemas import (
    BlindedOrchestratorContext,
    BlindedVerbalizerContext,
    EvaluationAction,
    InitialVerbalization,
    PubAgentResponse,
    VerbalizerAnswer,
)
from app.evaluation.verbalizer import EvaluationVerbalizer


class FakeVerbalizer:
    def __init__(self, *, unavailable: bool = False) -> None:
        self.initial_contexts: list[BlindedVerbalizerContext] = []
        self.answer_contexts: list[BlindedVerbalizerContext] = []
        self.questions: list[str] = []
        self.unavailable = unavailable

    async def initial(self, context: BlindedVerbalizerContext) -> InitialVerbalization:
        self.initial_contexts.append(context)
        return InitialVerbalization(
            description=(
                "A localized open wound with a deterministic mask and relative-depth output."
            ),
            directly_observed=["localized open wound"],
            deterministically_derived=["wound mask available", "relative-depth output available"],
            unavailable=[],
        )

    async def answer(
        self,
        context: BlindedVerbalizerContext,
        question: str,
    ) -> VerbalizerAnswer:
        self.answer_contexts.append(context)
        self.questions.append(question)
        if self.unavailable:
            return VerbalizerAnswer(
                question=question,
                answer="The requested information is unavailable in the supplied wound evidence.",
                unavailable=True,
                unavailable_reason="No exposed-structure field is present.",
            )
        return VerbalizerAnswer(
            question=question,
            answer="The deepest represented region is localized within the wound mask.",
            deterministically_derived=["localized relative-depth region"],
        )


class SequenceOrchestrator:
    def __init__(self, actions: list[EvaluationAction]) -> None:
        self.actions = actions
        self.contexts: list[BlindedOrchestratorContext] = []

    async def decide(self, context: BlindedOrchestratorContext) -> EvaluationAction:
        self.contexts.append(context)
        index = min(len(self.contexts) - 1, len(self.actions) - 1)
        return self.actions[index]


class AlwaysAskVerbalizer:
    def __init__(self) -> None:
        self.contexts: list[BlindedOrchestratorContext] = []

    async def decide(self, context: BlindedOrchestratorContext) -> EvaluationAction:
        self.contexts.append(context)
        return EvaluationAction(
            action="ASK_VERBALIZER",
            question=f"Focused question {context.iteration}?",
            current_condition_hypothesis="Diabetic_Foot_Ulcers_(DFU)",
            current_stage_hypothesis="Grade 3",
            independent_confidence=0.7,
            supporting_evidence=["deep tissue involvement"],
            unresolved_questions=["Is bone exposed?"],
            rationale="One more wound-specific answer may help.",
        )


class FakePubAgent:
    def __init__(self) -> None:
        self.questions: list[str] = []

    async def ask(self, question: str) -> PubAgentResponse:
        self.questions.append(question)
        return PubAgentResponse(
            question=question,
            summary="Wagner Grade 3 includes deep ulceration with abscess or osteomyelitis.",
            sufficient=True,
            evidence_quality_note="One directly relevant source was found.",
        )


def _config(tmp_path: Path) -> AgentConfig:
    return AgentConfig(
        model_provider="none",
        agent_model="gpt-4o",
        verifier_model="gpt-4o",
        embedding_model="text-embedding-3-small",
        literature_backend="pubmed_mcp",
        pubmed_mcp_project_path=tmp_path,
        pubmed_mcp_python="/usr/bin/python3",
        pubmed_mcp_timeout_seconds=5,
        pubmed_mcp_max_results=8,
        chroma_db_path=tmp_path / "chroma_db",
        clinical_docs_path=tmp_path / "clinical_docs",
        case_artifacts_path=tmp_path / "case_artifacts",
        max_iterations=8,
        max_qa=3,
        max_verifier_retries=2,
        confidence_threshold=0.75,
        severity_threshold=0.70,
        timeout_seconds=120,
        use_simulated_qa=True,
        max_evaluation_iterations=5,
        pubagent_project_path=tmp_path / "research_agent",
        pubagent_max_iterations=2,
        pubagent_per_source_limit=8,
    )


def _pipeline_output(
    *,
    condition: str = "Diabetic_Foot_Ulcers_(DFU)",
    stage: str = "Grade 3",
) -> dict:
    return {
        "validation": {"passed": True, "warnings": []},
        "condition": {
            "top1_label": condition,
            "confidence": 0.91,
            "top3": [{"label": condition, "probability": 0.91}],
        },
        "mask": {"mask_available": True, "mask_fraction": 0.24},
        "depth": {"depth_available": True, "depth_scale": "relative"},
        "severity": {
            "severity_prediction": stage,
            "severity_confidence": 0.88,
            "class_probabilities": {stage: 0.88},
        },
        "visual_evidence": {
            "findings": ["localized ulcer bed"],
            "limitations": ["single image"],
        },
    }


def _commit_action(
    *,
    condition: str | None = "Diabetic_Foot_Ulcers_(DFU)",
    stage: str | None = "Grade 3",
    confidence: float = 0.82,
) -> EvaluationAction:
    return EvaluationAction(
        action="COMMIT_ASSESSMENT",
        current_condition_hypothesis=condition,
        current_stage_hypothesis=stage,
        independent_confidence=confidence,
        supporting_evidence=["localized deep tissue involvement"],
        contradicting_evidence=[],
        unresolved_questions=[],
        rationale="The available wound and clinical evidence supports this assessment.",
    )


def _run(workflow: EvaluationWorkflow, *, case_id: str, output: dict | None = None):
    return asyncio.run(
        workflow.run(
            case_id=case_id,
            pipeline_output=output or _pipeline_output(),
        )
    )


def test_blinded_nodes_never_receive_hidden_model_output(tmp_path):
    verbalizer = FakeVerbalizer()
    orchestrator = SequenceOrchestrator([_commit_action()])
    workflow = EvaluationWorkflow(
        _config(tmp_path),
        verbalizer=verbalizer,
        orchestrator=orchestrator,
        pubagent=FakePubAgent(),
        adjudicator=EvaluationAdjudicator(_config(tmp_path)),
    )
    pipeline_output = _pipeline_output()
    pipeline_output["mask"]["condition_id"] = "hidden-policy-condition"
    pipeline_output["visual_evidence"]["candidate_stage"] = "Grade 3"

    _run(workflow, case_id="blinding", output=pipeline_output)

    verbalizer_payload = verbalizer.initial_contexts[0].model_dump()
    orchestrator_payload = orchestrator.contexts[0].model_dump()
    for payload in (verbalizer_payload, orchestrator_payload):
        serialized = str(payload)
        assert "Diabetic_Foot_Ulcers_(DFU)" not in serialized
        assert "Grade 3" not in serialized
        assert "0.91" not in serialized
        assert "hidden-policy-condition" not in serialized
        assert "model_condition" not in payload
        assert "model_stage" not in payload
        assert "model_confidence" not in payload


def test_structured_routes_reach_verbalizer_pubagent_and_commit(tmp_path):
    verbalizer = FakeVerbalizer()
    pubagent = FakePubAgent()
    orchestrator = SequenceOrchestrator(
        [
            EvaluationAction(
                action="ASK_VERBALIZER",
                question="Is the deepest region localized?",
                supporting_evidence=[],
                contradicting_evidence=[],
                unresolved_questions=["Depth distribution is unresolved."],
                rationale="A localized depth answer could change staging.",
            ),
            EvaluationAction(
                action="ASK_PUBAGENT",
                question="Does localized deep tissue involvement support Wagner Grade 3?",
                current_condition_hypothesis="Diabetic_Foot_Ulcers_(DFU)",
                current_stage_hypothesis="Grade 3",
                supporting_evidence=["localized relative-depth region"],
                unresolved_questions=["Clinical meaning of the depth finding."],
                rationale="Clinical criteria are needed.",
            ),
            _commit_action(),
        ]
    )
    workflow = EvaluationWorkflow(
        _config(tmp_path),
        verbalizer=verbalizer,
        orchestrator=orchestrator,
        pubagent=pubagent,
        adjudicator=EvaluationAdjudicator(_config(tmp_path)),
    )

    result = _run(workflow, case_id="routing")

    assert verbalizer.questions == ["Is the deepest region localized?"]
    assert pubagent.questions == [
        "Does localized deep tissue involvement support Wagner Grade 3?"
    ]
    assert result.evidence.supporting == [
        "localized relative-depth region",
        "localized deep tissue involvement",
    ]
    assert [event.node for event in result.audit_log] == [
        "prepare_evaluation_state",
        "initial_verbalizer",
        "blinded_orchestrator",
        "targeted_verbalizer",
        "blinded_orchestrator",
        "pubagent",
        "blinded_orchestrator",
        "commit_independent_assessment",
        "reveal_model_output",
        "comparison_adjudication",
        "finalize_evaluation",
    ]


def test_blinded_loop_is_hard_capped_at_five_iterations(tmp_path):
    verbalizer = FakeVerbalizer()
    orchestrator = AlwaysAskVerbalizer()
    workflow = EvaluationWorkflow(
        _config(tmp_path),
        verbalizer=verbalizer,
        orchestrator=orchestrator,
        pubagent=FakePubAgent(),
        adjudicator=EvaluationAdjudicator(_config(tmp_path)),
    )

    result = _run(workflow, case_id="loop-cap")

    assert result.iterations == 5
    assert len(orchestrator.contexts) == 5
    assert len(verbalizer.questions) == 4
    assert result.independent_evaluation.condition == "Diabetic_Foot_Ulcers_(DFU)"
    assert "iteration cap of 5" in result.independent_evaluation.rationale


def test_verbalizer_reports_missing_data_as_unavailable(tmp_path):
    verbalizer = EvaluationVerbalizer(_config(tmp_path))
    context = BlindedVerbalizerContext(
        case_evidence={"mask": {"mask_available": True}}
    )

    answer = asyncio.run(verbalizer.answer(context, "Is exposed bone visible?"))

    assert answer.unavailable is True
    assert "unavailable" in answer.answer.lower()
    assert answer.model_predicted == []


def test_independent_assessment_is_locked_before_reveal_and_persisted(tmp_path):
    workflow = EvaluationWorkflow(
        _config(tmp_path),
        verbalizer=FakeVerbalizer(),
        orchestrator=SequenceOrchestrator([_commit_action()]),
        pubagent=FakePubAgent(),
        adjudicator=EvaluationAdjudicator(_config(tmp_path)),
    )

    result = _run(workflow, case_id="assessment-lock")
    snapshot = workflow.get_state("assessment-lock")
    values = snapshot.values
    nodes = [event.node for event in result.audit_log]

    assert nodes.index("commit_independent_assessment") < nodes.index("reveal_model_output")
    assert values["assessment_committed"] is True
    assert values["model_revealed"] is True
    assert values["independent_assessment"] == result.independent_evaluation.model_dump()
    assert values["independent_assessment_digest"]


def test_agreement_returns_supported(tmp_path):
    workflow = EvaluationWorkflow(
        _config(tmp_path),
        verbalizer=FakeVerbalizer(),
        orchestrator=SequenceOrchestrator([_commit_action()]),
        pubagent=FakePubAgent(),
        adjudicator=EvaluationAdjudicator(_config(tmp_path)),
    )

    result = _run(workflow, case_id="agreement")

    assert result.final_evaluation.status == "SUPPORTED"
    assert result.comparison.condition_agreement is True
    assert result.comparison.stage_agreement is True
    assert result.model_output.confidence == 0.88
    assert result.independent_evaluation.confidence == 0.82


def test_material_disagreement_returns_flagged(tmp_path):
    workflow = EvaluationWorkflow(
        _config(tmp_path),
        verbalizer=FakeVerbalizer(),
        orchestrator=SequenceOrchestrator(
            [_commit_action(condition="Pressure_Injury_(PI)", stage="Stage 4")]
        ),
        pubagent=FakePubAgent(),
        adjudicator=EvaluationAdjudicator(_config(tmp_path)),
    )

    result = _run(workflow, case_id="disagreement")

    assert result.final_evaluation.status == "FLAGGED"
    assert result.comparison.condition_agreement is False
    assert result.comparison.stage_agreement is False
    assert result.comparison.discrepancies


def test_inadequate_independent_evidence_returns_insufficient(tmp_path):
    workflow = EvaluationWorkflow(
        _config(tmp_path),
        verbalizer=FakeVerbalizer(unavailable=True),
        orchestrator=SequenceOrchestrator(
            [_commit_action(condition=None, stage=None, confidence=0.0)]
        ),
        pubagent=FakePubAgent(),
        adjudicator=EvaluationAdjudicator(_config(tmp_path)),
    )

    result = _run(workflow, case_id="insufficient")

    assert result.final_evaluation.status == "INSUFFICIENT_EVIDENCE"
    assert result.final_evaluation.confidence == 0.0


def test_pubagent_adapter_uses_existing_run_interface_and_keeps_citations(tmp_path):
    class ExistingPubAgent:
        def __init__(self) -> None:
            self.calls = []

        def run(self, question, *, per_source_limit):
            self.calls.append((question, per_source_limit))

            class Answer:
                @staticmethod
                def to_dict():
                    return {
                        "summary": "Focused clinical answer.",
                        "sufficient": True,
                        "evidence_quality_note": "Relevant source.",
                        "ranked_evidence": [
                            {
                                "quote": "Grade 3 includes deep ulceration.",
                                "source_section": "abstract",
                                "article": {
                                    "title": "Wagner classification review",
                                    "source": "pubmed",
                                    "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
                                    "pmid": "123",
                                },
                            }
                        ],
                        "citations": [
                            {
                                "title": "Wagner classification review",
                                "source": "pubmed",
                                "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
                                "pmid": "123",
                            }
                        ],
                    }

            return Answer()

    existing = ExistingPubAgent()
    adapter = PubAgentAdapter(_config(tmp_path), agent=existing)

    response = asyncio.run(adapter.ask("What differentiates Grade 2 and Grade 3?"))

    assert existing.calls == [("What differentiates Grade 2 and Grade 3?", 8)]
    assert response.citations[0].pmid == "123"
    assert response.evidence[0].quote == "Grade 3 includes deep ulceration."
