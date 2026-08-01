from __future__ import annotations

import base64
import io
import json
from typing import Any

from PIL import Image

from app.agent.config import AgentConfig
from app.agent.json_utils import parse_json_object


class VisualEvidenceExtractor:
    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig.from_env()

    def extract(
        self,
        image: Image.Image,
        *,
        condition: str | None,
        stage: str | None,
        mask_summary: dict[str, Any] | None = None,
        depth_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.config.openai_enabled:
            return {
                "vlm_used": False,
                "source": "not_run",
                "model": None,
                "findings": [],
                "staging_relevant_observations": [],
                "limitations": ["OPENAI_API_KEY is not set; VLM visual extraction was skipped."],
            }

        try:
            from openai import OpenAI
        except ImportError:
            return {
                "vlm_used": False,
                "source": "not_run",
                "model": None,
                "findings": [],
                "staging_relevant_observations": [],
                "limitations": ["The openai package is not installed."],
            }

        prompt = {
            "task": (
                "Extract visible wound findings for a blinded wound evaluator. Do not "
                "diagnose or infer a stage. Describe only visual evidence, supplied "
                "deterministic image outputs, and image-quality limits."
            ),
            "model_mask_summary": mask_summary or {},
            "model_depth_summary": depth_summary or {},
            "return_json_schema": {
                "vlm_used": "boolean",
                "source": "openai_vision",
                "model": "string",
                "wound_visible": "boolean",
                "image_quality": "short string",
                "findings": ["short visual findings"],
                "staging_relevant_observations": ["findings relevant to depth/tissue loss/infection/necrosis"],
                "contraindicating_observations": ["visual observations that contradict another supplied observation"],
                "limitations": ["uncertainties or things not visible from the image"],
            },
        }
        if condition is not None:
            prompt["candidate_condition"] = condition
        if stage is not None:
            prompt["candidate_stage"] = stage

        try:
            response = OpenAI().responses.create(
                model=self.config.vision_model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Return JSON only.\n" + json.dumps(prompt, indent=2),
                            },
                            {
                                "type": "input_image",
                                "image_url": _image_data_url(image),
                            },
                        ],
                    }
                ],
            )
            payload = parse_json_object(response.output_text)
        except Exception as exc:
            return {
                "vlm_used": False,
                "source": "openai_vision_error",
                "model": self.config.vision_model,
                "findings": [],
                "staging_relevant_observations": [],
                "limitations": [f"VLM extraction failed: {type(exc).__name__}: {exc}"],
            }

        payload["vlm_used"] = True
        payload["source"] = "openai_vision"
        if str(payload.get("model", "")).strip().lower() in {"", "n/a", "none", "null"}:
            payload["model"] = self.config.vision_model
        for key in (
            "findings",
            "staging_relevant_observations",
            "contraindicating_observations",
            "limitations",
        ):
            if not isinstance(payload.get(key), list):
                payload[key] = []
        return payload

    def answer_followup(
        self,
        image: Image.Image,
        *,
        question: str,
        condition: str | None,
        stage: str | None,
        current_visual_evidence: dict[str, Any] | None = None,
        mask_summary: dict[str, Any] | None = None,
        depth_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.config.openai_enabled:
            return {
                "vlm_used": False,
                "source": "not_run",
                "model": None,
                "question": question,
                "answer": None,
                "findings": [],
                "limitations": ["OPENAI_API_KEY is not set; focused VLM follow-up was skipped."],
            }

        try:
            from openai import OpenAI
        except ImportError:
            return {
                "vlm_used": False,
                "source": "not_run",
                "model": None,
                "question": question,
                "answer": None,
                "findings": [],
                "limitations": ["The openai package is not installed."],
            }

        prompt = {
            "task": (
                "Answer the blinded evaluator's focused wound-specific follow-up. Use "
                "only what is visible in the image and the supplied deterministic image "
                "summaries. Do not diagnose, stage, or provide medical advice."
            ),
            "question": question,
            "current_visual_evidence": current_visual_evidence or {},
            "model_mask_summary": mask_summary or {},
            "model_depth_summary": depth_summary or {},
            "return_json_schema": {
                "vlm_used": "boolean",
                "source": "openai_vision",
                "model": "string",
                "question": "string",
                "answer": "brief visual answer or null",
                "findings": ["short findings directly relevant to the question"],
                "supports_question_premise": "true, false, or null if not visually determinable",
                "limitations": ["uncertainties or things not visible from the image"],
            },
        }
        if condition is not None:
            prompt["candidate_condition"] = condition
        if stage is not None:
            prompt["candidate_stage"] = stage

        try:
            response = OpenAI().responses.create(
                model=self.config.vision_model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Return JSON only.\n" + json.dumps(prompt, indent=2),
                            },
                            {
                                "type": "input_image",
                                "image_url": _image_data_url(image),
                            },
                        ],
                    }
                ],
            )
            payload = parse_json_object(response.output_text)
        except Exception as exc:
            return {
                "vlm_used": False,
                "source": "openai_vision_error",
                "model": self.config.vision_model,
                "question": question,
                "answer": None,
                "findings": [],
                "limitations": [f"Focused VLM follow-up failed: {type(exc).__name__}: {exc}"],
            }

        payload["vlm_used"] = True
        payload["source"] = "openai_vision"
        payload["question"] = str(payload.get("question") or question)
        if str(payload.get("model", "")).strip().lower() in {"", "n/a", "none", "null"}:
            payload["model"] = self.config.vision_model
        for key in ("findings", "limitations"):
            if not isinstance(payload.get(key), list):
                payload[key] = []
        return payload


def _image_data_url(image: Image.Image, max_side: int = 1024) -> str:
    image_rgb = image.convert("RGB")
    image_rgb.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image_rgb.save(buffer, format="JPEG", quality=85, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
