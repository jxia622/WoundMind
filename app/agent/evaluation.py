from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from app.agent.config import AgentConfig
from app.agent.json_utils import parse_json_object
from app.agent.schemas import DocumentChunk, VerifierResult


RetrieveDocsFn = Callable[[str, str | None, int], Awaitable[list[DocumentChunk]]]


@dataclass
class EvaluationLoopResult:
    draft: dict
    doc_chunks: list[DocumentChunk]
    verifier_result: VerifierResult
    summary: dict
    trace_steps: list[dict] = field(default_factory=list)


class DiagnosticAgent:
    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig.from_env()

    async def propose(
        self,
        *,
        base_draft: dict,
        visual_evidence: dict,
        doc_chunks: list[DocumentChunk],
        evaluator_feedback: dict | None,
        iteration: int,
    ) -> dict:
        if not self.config.openai_enabled:
            return self._deterministic_proposal(base_draft, visual_evidence, evaluator_feedback)
        return await asyncio.to_thread(
            self._propose_with_openai,
            base_draft,
            visual_evidence,
            doc_chunks,
            evaluator_feedback,
            iteration,
        )

    def _propose_with_openai(
        self,
        base_draft: dict,
        visual_evidence: dict,
        doc_chunks: list[DocumentChunk],
        evaluator_feedback: dict | None,
        iteration: int,
    ) -> dict:
        try:
            from openai import OpenAI
        except ImportError:
            return self._deterministic_proposal(base_draft, visual_evidence, evaluator_feedback)

        prompt = {
            "role": "diagnostic_agent",
            "instruction": (
                "Build or revise a research staging assessment from model outputs, "
                "VLM visual evidence, and retrieved literature. Do not diagnose. "
                "Keep the condition and stage anchored to the supplied model output unless "
                "the evidence clearly contradicts it."
            ),
            "iteration": iteration,
            "base_model_draft": base_draft,
            "visual_evidence": visual_evidence,
            "retrieved_evidence": [_chunk_for_prompt(chunk) for chunk in doc_chunks],
            "evaluator_feedback": evaluator_feedback or {},
            "return_json_keys": [
                "condition",
                "severity_stage",
                "severity_confidence",
                "rationale",
                "visual_findings_used",
                "literature_findings_used",
                "uncertainties",
                "next_evidence_query",
            ],
        }
        try:
            response = OpenAI().responses.create(
                model=self.config.agent_model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are the diagnostic agent inside a wound research prototype. "
                            "Return JSON only. Do not provide medical advice."
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt, indent=2)},
                ],
            )
            payload = parse_json_object(response.output_text)
        except Exception as exc:
            payload = self._deterministic_proposal(base_draft, visual_evidence, evaluator_feedback)
            payload.setdefault("uncertainties", []).append(
                f"Diagnostic agent LLM failed: {type(exc).__name__}: {exc}"
            )

        payload.setdefault("condition", base_draft.get("condition"))
        payload.setdefault("severity_stage", base_draft.get("severity_stage"))
        payload.setdefault("severity_confidence", base_draft.get("severity_confidence"))
        payload.setdefault("rationale", base_draft.get("rationale", ""))
        _ensure_list(payload, "visual_findings_used")
        _ensure_list(payload, "literature_findings_used")
        _ensure_list(payload, "uncertainties")
        return payload

    @staticmethod
    def _deterministic_proposal(
        base_draft: dict,
        visual_evidence: dict,
        evaluator_feedback: dict | None,
    ) -> dict:
        findings = visual_evidence.get("staging_relevant_observations") or visual_evidence.get("findings") or []
        uncertainties = list(visual_evidence.get("limitations") or [])
        if evaluator_feedback and evaluator_feedback.get("flag"):
            uncertainties.append(str(evaluator_feedback["flag"]))
        return {
            **base_draft,
            "visual_findings_used": findings[:5],
            "literature_findings_used": [],
            "uncertainties": uncertainties,
            "next_evidence_query": None,
        }


class EvidenceEvaluator:
    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig.from_env()

    async def evaluate(
        self,
        *,
        draft: dict,
        visual_evidence: dict,
        doc_chunks: list[DocumentChunk],
        iteration: int,
    ) -> VerifierResult:
        if not self.config.openai_enabled:
            return self._evaluate_deterministic(draft, doc_chunks)
        return await asyncio.to_thread(
            self._evaluate_with_openai,
            draft,
            visual_evidence,
            doc_chunks,
            iteration,
        )

    def _evaluate_with_openai(
        self,
        draft: dict,
        visual_evidence: dict,
        doc_chunks: list[DocumentChunk],
        iteration: int,
    ) -> VerifierResult:
        try:
            from openai import OpenAI
        except ImportError:
            return VerifierResult(
                result="UNCERTAIN",
                flag="OPENAI_API_KEY is set but the openai package is not installed.",
            )

        prompt = {
            "role": "evaluator",
            "instruction": (
                "Evaluate whether the diagnostic agent's proposed wound stage is supported "
                "by the VLM visual evidence and retrieved literature. If evidence is thin but "
                "a targeted PubMed query could help, return needs_more_evidence=true and a "
                "followup_query. Do not diagnose."
            ),
            "iteration": iteration,
            "draft": draft,
            "visual_evidence": visual_evidence,
            "retrieved_evidence": [_chunk_for_prompt(chunk) for chunk in doc_chunks],
            "return_json_schema": {
                "result": "PASS, FAIL, or UNCERTAIN",
                "citation": "best supporting citation or null",
                "flag": "brief issue, escalation reason, or null",
                "needs_more_evidence": "boolean",
                "followup_query": "targeted PubMed query or null",
                "reasoning": "brief reason",
            },
        }
        try:
            response = OpenAI().responses.create(
                model=self.config.evaluation_model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You are the evaluator in a wound research agent loop. "
                            "Return JSON only with result, citation, flag, needs_more_evidence, "
                            "followup_query, and reasoning."
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt, indent=2)},
                ],
            )
            payload = parse_json_object(response.output_text)
        except Exception as exc:
            return VerifierResult(
                result="UNCERTAIN",
                flag=f"Evaluator LLM failed: {type(exc).__name__}: {exc}",
                needs_more_evidence=False,
            )

        result = str(payload.get("result", "UNCERTAIN")).upper()
        if result not in {"PASS", "FAIL", "UNCERTAIN"}:
            result = "UNCERTAIN"
        return VerifierResult(
            result=result,
            citation=_none_if_blank(payload.get("citation")),
            flag=_none_if_blank(payload.get("flag")),
            needs_more_evidence=bool(payload.get("needs_more_evidence", False)),
            followup_query=_none_if_blank(payload.get("followup_query")),
            reasoning=_none_if_blank(payload.get("reasoning")),
        )

    def _evaluate_deterministic(
        self,
        draft: dict,
        doc_chunks: list[DocumentChunk],
    ) -> VerifierResult:
        stage = draft.get("severity_stage")
        condition = (draft.get("condition") or "").lower()
        if not stage:
            return VerifierResult(
                result="UNCERTAIN",
                flag="No severity stage was drafted for evaluator review.",
            )

        stage_terms = _stage_terms(stage)
        for chunk in doc_chunks:
            text_lower = chunk.text.lower()
            if any(term in text_lower for term in stage_terms):
                return VerifierResult(
                    result="PASS",
                    citation=_short_citation(chunk.text, stage_terms),
                    flag="Deterministic local verifier used because OPENAI_API_KEY is not set.",
                    needs_more_evidence=False,
                )

        if "diabetic" in condition or "dfu" in condition or "pressure" in condition:
            return VerifierResult(
                result="UNCERTAIN",
                flag=f"No retrieved passage clearly matched {stage}.",
                needs_more_evidence=True,
                followup_query=f"{condition} {stage} staging criteria wound depth tissue involvement",
            )
        return VerifierResult(
            result="UNCERTAIN",
            flag="Severity verification is only supported for DFU and pressure injury.",
        )


class EvaluationAgentLoop:
    def __init__(
        self,
        config: AgentConfig | None = None,
        diagnostic_agent: DiagnosticAgent | None = None,
        evaluator: EvidenceEvaluator | None = None,
    ) -> None:
        self.config = config or AgentConfig.from_env()
        self.diagnostic_agent = diagnostic_agent or DiagnosticAgent(self.config)
        self.evaluator = evaluator or EvidenceEvaluator(self.config)

    async def run(
        self,
        *,
        base_draft: dict,
        visual_evidence: dict,
        doc_chunks: list[DocumentChunk],
        condition: str | None,
        retrieve_docs: RetrieveDocsFn,
    ) -> EvaluationLoopResult:
        chunks = list(doc_chunks)
        feedback: dict | None = None
        trace_steps: list[dict] = []
        final_draft = base_draft
        final_result = VerifierResult(result="UNCERTAIN", flag="Evaluation loop did not run.")
        max_iterations = max(1, min(self.config.max_iterations, self.config.max_verifier_retries + 1))

        for iteration in range(1, max_iterations + 1):
            final_draft = await self.diagnostic_agent.propose(
                base_draft=base_draft,
                visual_evidence=visual_evidence,
                doc_chunks=chunks,
                evaluator_feedback=feedback,
                iteration=iteration,
            )
            trace_steps.append(
                {
                    "tool": "diagnostic_agent",
                    "inputs_summary": {
                        "iteration": iteration,
                        "evidence_chunks": len(chunks),
                        "has_evaluator_feedback": feedback is not None,
                    },
                    "outputs_summary": _draft_summary(final_draft),
                }
            )

            final_result = await self.evaluator.evaluate(
                draft=final_draft,
                visual_evidence=visual_evidence,
                doc_chunks=chunks,
                iteration=iteration,
            )
            trace_steps.append(
                {
                    "tool": "evaluator",
                    "inputs_summary": {"iteration": iteration, "evidence_chunks": len(chunks)},
                    "outputs_summary": final_result.model_dump(),
                }
            )

            if (
                final_result.result in {"PASS", "FAIL"}
                or not final_result.needs_more_evidence
                or not final_result.followup_query
                or iteration >= max_iterations
            ):
                break

            new_chunks = await retrieve_docs(final_result.followup_query, condition, 4)
            before = len(chunks)
            chunks = _dedupe_chunks([*chunks, *new_chunks])
            trace_steps.append(
                {
                    "tool": "retrieve_docs",
                    "inputs_summary": {
                        "query": final_result.followup_query,
                        "condition": condition,
                        "requested_by": "evaluator",
                        "iteration": iteration,
                    },
                    "outputs_summary": {
                        "new_chunks": len(new_chunks),
                        "deduped_total": len(chunks),
                    },
                }
            )
            if len(chunks) == before:
                final_result.needs_more_evidence = False
                final_result.flag = final_result.flag or "Evaluator requested more evidence, but retrieval returned no new chunks."
                break
            feedback = final_result.model_dump()

        summary = {
            "iterations": len([step for step in trace_steps if step["tool"] == "evaluator"]),
            "final_result": final_result.result,
            "needs_more_evidence": final_result.needs_more_evidence,
            "final_reasoning": final_result.reasoning,
            "evidence_chunks": len(chunks),
        }
        return EvaluationLoopResult(
            draft=final_draft,
            doc_chunks=chunks,
            verifier_result=final_result,
            summary=summary,
            trace_steps=trace_steps,
        )


def _chunk_for_prompt(chunk: DocumentChunk) -> dict:
    return {
        "source": chunk.source,
        "doc_id": chunk.doc_id,
        "section": chunk.section,
        "condition_tag": chunk.condition_tag,
        "score": chunk.score,
        "text": chunk.text,
    }


def _draft_summary(draft: dict) -> dict:
    return {
        "condition": draft.get("condition"),
        "severity_stage": draft.get("severity_stage"),
        "severity_confidence": draft.get("severity_confidence"),
        "visual_findings_used": draft.get("visual_findings_used", []),
        "uncertainties": draft.get("uncertainties", []),
        "next_evidence_query": draft.get("next_evidence_query"),
    }


def _ensure_list(payload: dict, key: str) -> None:
    if not isinstance(payload.get(key), list):
        payload[key] = []


def _none_if_blank(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "n/a"}:
        return None
    return str(value)


def _dedupe_chunks(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    seen: set[tuple[str | None, str]] = set()
    deduped: list[DocumentChunk] = []
    for chunk in chunks:
        key = (chunk.doc_id, chunk.source)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped


def _stage_terms(stage: str) -> list[str]:
    lower = stage.lower()
    terms = [lower]
    numbers = re.findall(r"\d+", lower)
    roman_by_number = {"0": "0", "1": "i", "2": "ii", "3": "iii", "4": "iv", "5": "v"}
    for number in numbers:
        roman = roman_by_number.get(number)
        if "stage" in lower and roman:
            terms.append(f"stage {roman}")
            terms.append(f"category/stage {roman}")
        if "grade" in lower:
            terms.append(f"grade {number}")
            terms.append(f"wagner grade {number}")
    return terms


def _short_citation(text: str, terms: list[str], limit: int = 360) -> str:
    text_one_line = re.sub(r"\s+", " ", text).strip()
    lower = text_one_line.lower()
    start = 0
    for term in terms:
        index = lower.find(term)
        if index >= 0:
            start = max(0, index - 100)
            break
    return text_one_line[start : start + limit]
