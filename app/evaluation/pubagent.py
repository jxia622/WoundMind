from __future__ import annotations

import asyncio
import importlib
import sys
from typing import Any, Protocol

from app.agent.config import AgentConfig
from app.evaluation.schemas import (
    CitationMetadata,
    PubAgentEvidence,
    PubAgentResponse,
)


class PubAgentProtocol(Protocol):
    async def ask(self, question: str) -> PubAgentResponse: ...


class PubAgentAdapter:
    """Thin adapter over PubAgent's existing build_agent().run() public facade."""

    def __init__(self, config: AgentConfig | None = None, *, agent: Any = None) -> None:
        self.config = config or AgentConfig.from_env()
        self._agent = agent

    async def ask(self, question: str) -> PubAgentResponse:
        try:
            agent = self._agent or self._load_agent()
            self._agent = agent
            answer = await asyncio.to_thread(
                agent.run,
                question,
                per_source_limit=self.config.pubagent_per_source_limit,
            )
            return self._convert_answer(question, answer.to_dict())
        except Exception as exc:
            return PubAgentResponse(
                question=question,
                summary="PubAgent could not provide clinical evidence.",
                sufficient=False,
                evidence_quality_note="No external clinical evidence was added.",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _load_agent(self):
        project_path = self.config.pubagent_project_path.resolve()
        if not project_path.exists():
            raise FileNotFoundError(f"PubAgent project not found: {project_path}")
        project_string = str(project_path)
        if project_string not in sys.path:
            sys.path.insert(0, project_string)
        module = importlib.import_module("research_agent.main")
        return module.build_agent(max_iterations=self.config.pubagent_max_iterations)

    @staticmethod
    def _convert_answer(question: str, payload: dict[str, Any]) -> PubAgentResponse:
        citations = [_citation(item) for item in payload.get("citations", [])]
        evidence: list[PubAgentEvidence] = []
        for item in payload.get("ranked_evidence", []):
            article = item.get("article") or {}
            evidence.append(
                PubAgentEvidence(
                    quote=str(item.get("quote") or ""),
                    source_section=item.get("source_section"),
                    citation=_citation(article),
                )
            )
        return PubAgentResponse(
            question=question,
            summary=str(payload.get("summary") or ""),
            sufficient=bool(payload.get("sufficient", False)),
            evidence_quality_note=str(payload.get("evidence_quality_note") or ""),
            evidence=evidence,
            citations=citations,
        )


def _citation(payload: dict[str, Any]) -> CitationMetadata:
    return CitationMetadata(
        title=payload.get("title"),
        source=payload.get("source"),
        url=payload.get("oa_url") or payload.get("url"),
        authors=[str(item) for item in payload.get("authors", [])],
        journal=payload.get("journal"),
        year=str(payload["year"]) if payload.get("year") is not None else None,
        pmid=payload.get("pmid"),
        pmcid=payload.get("pmcid"),
        doi=payload.get("doi"),
    )
