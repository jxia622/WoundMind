from __future__ import annotations

import asyncio

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


class FailingMemory:
    def retrieve(self, *args, **kwargs):
        raise AssertionError("local memory should not be used by default")


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
