from __future__ import annotations

import asyncio

import mcp
import mcp.client.stdio as mcp_stdio
import pytest

from app.agent.config import AgentConfig
from app.agent.pubmed_mcp import PubMedMCPRetriever


class FakeAsyncContext:
    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeToolResult:
    def __init__(self, structured_content: dict):
        self.structuredContent = structured_content
        self.content = []


class FakeClientSession:
    calls: list[tuple[str, dict]] = []

    def __init__(self, read, write):
        self.read = read
        self.write = write

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def initialize(self):
        return None

    async def call_tool(self, name, arguments):
        FakeClientSession.calls.append((name, arguments))
        return FakeToolResult(
            {
                "query_used": "diabetic foot ulcer wagner grade 3 staging criteria",
                "ranked_articles": [
                    {
                        "title": "Wagner classification of diabetic foot ulcers",
                        "abstract": "Grade 3 lesions show deep ulceration with abscess or osteomyelitis.",
                        "journal": "J Wound Care",
                        "year": 2021,
                        "pmid": "12345",
                        "doi": None,
                        "url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
                        "relevance_score": 0.92,
                        "relevance_reasons": ["matches staging criteria"],
                        "publication_types": ["Review"],
                        "mesh_terms": ["Diabetic Foot"],
                    }
                ],
            }
        )


@pytest.fixture(autouse=True)
def _reset_fake_session_calls():
    FakeClientSession.calls = []
    yield
    FakeClientSession.calls = []


def _stub_mcp_transport(monkeypatch, captured_server_params: dict):
    def fake_stdio_client(server_params):
        captured_server_params["value"] = server_params
        return FakeAsyncContext(("fake-read", "fake-write"))

    monkeypatch.setattr(mcp_stdio, "stdio_client", fake_stdio_client)
    monkeypatch.setattr(mcp, "ClientSession", FakeClientSession)


def test_retrieve_calls_search_and_fetch_pubmed_tool_and_parses_pmid_chunk(monkeypatch, tmp_path):
    captured_server_params: dict = {}
    _stub_mcp_transport(monkeypatch, captured_server_params)

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
        max_iterations=8,
        max_qa=3,
        max_verifier_retries=2,
        confidence_threshold=0.75,
        severity_threshold=0.70,
        timeout_seconds=120,
        use_simulated_qa=True,
    )
    retriever = PubMedMCPRetriever(config)

    chunks = asyncio.run(
        retriever.retrieve(
            "diabetic foot ulcer Wagner grade 3 staging criteria",
            "Diabetic_Foot_Ulcers_(DFU)",
            top_k=4,
        )
    )

    assert len(FakeClientSession.calls) == 1
    tool_name, arguments = FakeClientSession.calls[0]
    assert tool_name == "search_and_fetch_pubmed"
    assert arguments["clinical_question"] == (
        "Diabetic_Foot_Ulcers_(DFU): diabetic foot ulcer Wagner grade 3 staging criteria"
    )
    assert arguments["max_results"] == 8

    assert captured_server_params["value"].command == "/usr/bin/python3"
    assert captured_server_params["value"].args == ["-m", "pubmed_clinical_mcp.server"]

    assert len(chunks) == 1
    assert chunks[0].doc_id == "PMID:12345"
    assert chunks[0].source == "https://pubmed.ncbi.nlm.nih.gov/12345/"
    assert chunks[0].condition_tag == "dfu"
    assert "osteomyelitis" in chunks[0].text.lower()


def test_retrieve_raises_when_pubmed_mcp_project_path_missing(tmp_path):
    missing_path = tmp_path / "does-not-exist"
    config = AgentConfig(
        model_provider="openai",
        agent_model="gpt-4o",
        verifier_model="gpt-4o",
        embedding_model="text-embedding-3-small",
        literature_backend="pubmed_mcp",
        pubmed_mcp_project_path=missing_path,
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
    )
    retriever = PubMedMCPRetriever(config)

    with pytest.raises(RuntimeError, match="PubMed MCP project not found"):
        asyncio.run(retriever.retrieve("query", "Diabetic_Foot_Ulcers_(DFU)"))
