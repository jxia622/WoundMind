from __future__ import annotations

import asyncio

from app.agent.config import AgentConfig
from app.agent.schemas import DocumentChunk
from app.agent.tools import AgentTools


class FakePubMedRetriever:
    def __init__(self) -> None:
        self.calls = []

    async def retrieve(self, query: str, condition: str | None, top_k: int = 4):
        self.calls.append((query, condition, top_k))
        return [
            DocumentChunk(
                text="PubMed evidence text",
                source="https://pubmed.ncbi.nlm.nih.gov/123/",
                doc_id="PMID:123",
                section="PubMed abstract",
                condition_tag="dfu",
                score=0.9,
            )
        ]


class FailingPubMedRetriever:
    async def retrieve(self, *args, **kwargs):
        raise AssertionError("PubMed MCP should not be used when backend is explicitly local")


class FailingMemory:
    def retrieve(self, *args, **kwargs):
        raise AssertionError("local memory should not be used by default")


class FakeMemory:
    def __init__(self) -> None:
        self.calls = []

    def retrieve(self, query, condition, top_k, allowed_doc_ids):
        self.calls.append((query, condition, top_k, allowed_doc_ids))
        return [
            DocumentChunk(
                text="Local clinical doc text",
                source="wagner_classification.pdf",
                doc_id="wagner_classification",
                section="chunk-1",
                condition_tag="dfu",
                score=1.0,
            )
        ]


def test_default_retrieval_uses_pubmed_mcp_not_local_memory(monkeypatch):
    monkeypatch.delenv("LITERATURE_RETRIEVAL_BACKEND", raising=False)
    retriever = FakePubMedRetriever()
    tools = AgentTools(
        pipeline=object(),
        memory=FailingMemory(),
        literature_retriever=retriever,
    )

    chunks = asyncio.run(
        tools.retrieve_docs(
            "diabetic foot ulcer Wagner grade 3 staging criteria",
            "Diabetic_Foot_Ulcers_(DFU)",
            top_k=1,
            allowed_doc_ids=["local-doc-id-that-should-not-filter-pubmed"],
        )
    )

    assert chunks[0].doc_id == "PMID:123"
    assert retriever.calls == [
        ("diabetic foot ulcer Wagner grade 3 staging criteria", "Diabetic_Foot_Ulcers_(DFU)", 1)
    ]


def test_explicit_local_backend_uses_local_memory_not_pubmed(monkeypatch):
    monkeypatch.setenv("LITERATURE_RETRIEVAL_BACKEND", "local")
    config = AgentConfig.from_env()
    assert config.literature_backend == "local"

    memory = FakeMemory()
    tools = AgentTools(
        pipeline=object(),
        config=config,
        memory=memory,
        literature_retriever=FailingPubMedRetriever(),
    )

    chunks = asyncio.run(
        tools.retrieve_docs(
            "diabetic foot ulcer Wagner grade 3 staging criteria",
            "Diabetic_Foot_Ulcers_(DFU)",
            top_k=1,
            allowed_doc_ids=["wagner_classification"],
        )
    )

    assert chunks[0].doc_id == "wagner_classification"
    assert memory.calls == [
        (
            "diabetic foot ulcer Wagner grade 3 staging criteria",
            "Diabetic_Foot_Ulcers_(DFU)",
            1,
            ["wagner_classification"],
        )
    ]
