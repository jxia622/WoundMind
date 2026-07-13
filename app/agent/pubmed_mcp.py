from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from app.agent.config import AgentConfig
from app.agent.memory import normalize_condition_tag
from app.agent.schemas import DocumentChunk


class PubMedMCPRetriever:
    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig.from_env()

    async def retrieve(
        self,
        query: str,
        condition: str | None,
        top_k: int = 4,
        allowed_doc_ids: list[str] | None = None,
    ) -> list[DocumentChunk]:
        if not self.config.pubmed_mcp_project_path.exists():
            raise RuntimeError(
                "PubMed MCP project not found at "
                f"{self.config.pubmed_mcp_project_path}. Set PUBMED_MCP_PROJECT_PATH."
            )

        payload = await asyncio.wait_for(
            self._call_search_and_fetch(query, condition, top_k),
            timeout=self.config.pubmed_mcp_timeout_seconds,
        )
        articles = payload.get("ranked_articles", [])[:top_k]
        return [
            self._article_to_chunk(article, condition, payload.get("query_used"))
            for article in articles
            if article.get("title") or article.get("abstract")
        ]

    async def _call_search_and_fetch(
        self,
        query: str,
        condition: str | None,
        top_k: int,
    ) -> dict[str, Any]:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            raise RuntimeError(
                "The mcp package is required for PubMed MCP retrieval. "
                "Install requirements_agent.txt or run `python3 -m pip install mcp`."
            ) from exc

        src_path = self.config.pubmed_mcp_project_path / "src"
        pythonpath_parts = [str(src_path)]
        if os.environ.get("PYTHONPATH"):
            pythonpath_parts.append(os.environ["PYTHONPATH"])
        server_env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(pythonpath_parts),
        }
        server_params = StdioServerParameters(
            command=self.config.pubmed_mcp_python,
            args=["-m", "pubmed_clinical_mcp.server"],
            env=server_env,
        )
        clinical_question = self._clinical_question(query, condition)
        max_results = max(top_k, min(self.config.pubmed_mcp_max_results, 25))

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "search_and_fetch_pubmed",
                    {
                        "clinical_question": clinical_question,
                        "filters": {"english_only": True},
                        "max_results": max_results,
                    },
                )
        return self._tool_result_to_dict(result)

    @staticmethod
    def _tool_result_to_dict(result: Any) -> dict[str, Any]:
        structured = getattr(result, "structuredContent", None)
        if isinstance(structured, dict):
            return structured
        for content in getattr(result, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                return json.loads(text)
        return {}

    @staticmethod
    def _clinical_question(query: str, condition: str | None) -> str:
        if condition:
            return f"{condition}: {query}"
        return query

    @staticmethod
    def _article_to_chunk(
        article: dict[str, Any],
        condition: str | None,
        query_used: str | None,
    ) -> DocumentChunk:
        citation_bits = [
            article.get("title"),
            article.get("journal"),
            article.get("year"),
            f"PMID {article.get('pmid')}" if article.get("pmid") else None,
            f"DOI {article.get('doi')}" if article.get("doi") else None,
        ]
        citation = "; ".join(str(bit) for bit in citation_bits if bit)
        reasons = ", ".join(article.get("relevance_reasons") or [])
        text_parts = [
            citation,
            article.get("abstract"),
            f"Publication types: {', '.join(article.get('publication_types') or [])}",
            f"MeSH terms: {', '.join(article.get('mesh_terms') or [])}",
            f"PubMed query used: {query_used}" if query_used else None,
            f"Ranking reasons: {reasons}" if reasons else None,
        ]
        pmid = article.get("pmid")
        return DocumentChunk(
            text="\n".join(str(part) for part in text_parts if part),
            source=article.get("url") or "PubMed",
            doc_id=f"PMID:{pmid}" if pmid else article.get("doi"),
            section="PubMed abstract",
            condition_tag=normalize_condition_tag(condition),
            score=article.get("relevance_score"),
        )
