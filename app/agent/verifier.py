from __future__ import annotations

import json
import re

from app.agent.config import AgentConfig
from app.agent.schemas import DocumentChunk, VerifierResult


ROMAN_BY_NUMBER = {
    "0": "0",
    "1": "i",
    "2": "ii",
    "3": "iii",
    "4": "iv",
    "5": "v",
}


class AssessmentVerifier:
    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig.from_env()

    def verify(self, draft: dict, doc_chunks: list[DocumentChunk]) -> VerifierResult:
        if self.config.openai_enabled:
            return self._verify_with_openai(draft, doc_chunks)
        return self._verify_deterministic(draft, doc_chunks)

    def _verify_with_openai(self, draft: dict, doc_chunks: list[DocumentChunk]) -> VerifierResult:
        try:
            from openai import OpenAI
        except ImportError:
            return VerifierResult(
                result="UNCERTAIN",
                flag="OPENAI_API_KEY is set but the openai package is not installed.",
            )

        client = OpenAI()
        chunk_payload = [
            {
                "source": chunk.source,
                "doc_id": chunk.doc_id,
                "section": chunk.section,
                "condition_tag": chunk.condition_tag,
                "text": chunk.text,
            }
            for chunk in doc_chunks
        ]
        prompt = (
            "Does the draft staging match the criteria in the provided documents? "
            "Return JSON only with keys result (PASS|FAIL|UNCERTAIN), citation, flag.\n\n"
            f"Draft assessment:\n{json.dumps(draft, indent=2)}\n\n"
            f"Retrieved document chunks:\n{json.dumps(chunk_payload, indent=2)}"
        )
        response = client.responses.create(
            model=self.config.verifier_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a narrow clinical-staging audit verifier for a research "
                        "prototype. Do not diagnose. Check only whether the draft stage "
                        "is supported by the provided passages."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        text = response.output_text
        try:
            payload = self._parse_json_object(text)
        except json.JSONDecodeError:
            return VerifierResult(
                result="UNCERTAIN",
                flag="Verifier returned non-JSON output.",
            )
        return VerifierResult(**payload)

    @staticmethod
    def _parse_json_object(text: str) -> dict:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                return json.loads(cleaned[start : end + 1])
            raise

    def _verify_deterministic(self, draft: dict, doc_chunks: list[DocumentChunk]) -> VerifierResult:
        stage = draft.get("severity_stage")
        condition = (draft.get("condition") or "").lower()
        if not stage:
            return VerifierResult(
                result="UNCERTAIN",
                flag="No severity stage was drafted for verifier review.",
            )

        stage_terms = self._stage_terms(stage)
        for chunk in doc_chunks:
            text_lower = chunk.text.lower()
            if any(term in text_lower for term in stage_terms):
                return VerifierResult(
                    result="PASS",
                    citation=self._short_citation(chunk.text, stage_terms),
                    flag="Deterministic local verifier used because OPENAI_API_KEY is not set.",
                )

        if "diabetic" in condition or "dfu" in condition or "pressure" in condition:
            return VerifierResult(
                result="UNCERTAIN",
                flag=f"No retrieved passage clearly matched {stage}.",
            )

        return VerifierResult(
            result="UNCERTAIN",
            flag="Severity verification is only supported for DFU and pressure injury.",
        )

    @staticmethod
    def _stage_terms(stage: str) -> list[str]:
        lower = stage.lower()
        terms = [lower]
        numbers = re.findall(r"\d+", lower)
        for number in numbers:
            roman = ROMAN_BY_NUMBER.get(number)
            if "stage" in lower and roman:
                terms.append(f"stage {roman}")
                terms.append(f"category/stage {roman}")
            if "grade" in lower:
                terms.append(f"grade {number}")
                terms.append(f"wagner grade {number}")
        return terms

    @staticmethod
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
