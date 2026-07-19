from __future__ import annotations

import asyncio

from app.agent.config import AgentConfig
from app.agent.evaluation import EvaluationAgentLoop
from app.agent.schemas import DocumentChunk, VerifierResult


class FakeDiagnosticAgent:
    def __init__(self) -> None:
        self.calls = []

    async def propose(
        self,
        *,
        base_draft,
        visual_evidence,
        doc_chunks,
        evaluator_feedback,
        iteration,
    ):
        self.calls.append((iteration, len(doc_chunks), evaluator_feedback))
        return {
            **base_draft,
            "rationale": f"draft iteration {iteration}",
            "visual_findings_used": visual_evidence.get("findings", []),
        }


class FakeEvaluator:
    def __init__(self) -> None:
        self.calls = []

    async def evaluate(self, *, draft, visual_evidence, doc_chunks, iteration):
        self.calls.append((iteration, len(doc_chunks), draft["rationale"]))
        if iteration == 1:
            return VerifierResult(
                result="UNCERTAIN",
                flag="Need exact staging criteria.",
                needs_more_evidence=True,
                followup_query="Wagner grade 3 diabetic foot ulcer criteria",
                reasoning="Initial abstract was too broad.",
            )
        return VerifierResult(
            result="PASS",
            citation="PMID:222",
            needs_more_evidence=False,
            reasoning="Second retrieval supports the criteria.",
        )


def test_evaluation_loop_retrieves_more_evidence_and_re_evaluates(tmp_path):
    config = AgentConfig(
        model_provider="openai",
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
        max_iterations=3,
        max_qa=3,
        max_verifier_retries=2,
        confidence_threshold=0.75,
        severity_threshold=0.70,
        timeout_seconds=120,
        use_simulated_qa=True,
    )
    diagnostic = FakeDiagnosticAgent()
    evaluator = FakeEvaluator()
    loop = EvaluationAgentLoop(config, diagnostic_agent=diagnostic, evaluator=evaluator)
    retrieval_calls = []

    async def retrieve_docs(query, condition, top_k):
        retrieval_calls.append((query, condition, top_k))
        return [
            DocumentChunk(
                text="Wagner grade 3 criteria mention abscess or osteomyelitis.",
                source="https://pubmed.ncbi.nlm.nih.gov/222/",
                doc_id="PMID:222",
                section="PubMed abstract",
                condition_tag="dfu",
                score=0.91,
            )
        ]

    result = asyncio.run(
        loop.run(
            base_draft={
                "condition": "Diabetic_Foot_Ulcers_(DFU)",
                "severity_stage": "Grade 3",
                "severity_confidence": 0.88,
                "rationale": "model draft",
            },
            visual_evidence={"findings": ["deep-appearing tissue loss"]},
            doc_chunks=[
                DocumentChunk(
                    text="General diabetic foot ulcer article.",
                    source="https://pubmed.ncbi.nlm.nih.gov/111/",
                    doc_id="PMID:111",
                    section="PubMed abstract",
                    condition_tag="dfu",
                )
            ],
            condition="Diabetic_Foot_Ulcers_(DFU)",
            retrieve_docs=retrieve_docs,
        )
    )

    assert result.verifier_result.result == "PASS"
    assert result.summary["iterations"] == 2
    assert retrieval_calls == [
        (
            "Wagner grade 3 diabetic foot ulcer criteria",
            "Diabetic_Foot_Ulcers_(DFU)",
            4,
        )
    ]
    assert [step["tool"] for step in result.trace_steps] == [
        "diagnostic_agent",
        "evaluator",
        "retrieve_docs",
        "diagnostic_agent",
        "evaluator",
    ]
