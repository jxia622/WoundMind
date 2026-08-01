from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from app.agent.config import AgentConfig
from app.agent.memory import ClinicalMemory
from app.agent.pubmed_mcp import PubMedMCPRetriever
from app.agent.schemas import DocumentChunk
from app.agent.vision import VisualEvidenceExtractor


class AgentTools:
    def __init__(
        self,
        pipeline,
        config: AgentConfig | None = None,
        memory: ClinicalMemory | None = None,
        literature_retriever: PubMedMCPRetriever | None = None,
        visual_evidence_extractor: VisualEvidenceExtractor | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.config = config or AgentConfig.from_env()
        self.memory = memory
        self.literature_retriever = literature_retriever or PubMedMCPRetriever(self.config)
        self.visual_evidence_extractor = visual_evidence_extractor or VisualEvidenceExtractor(
            self.config
        )

    async def segment_wound(self, image: Image.Image) -> dict[str, Any]:
        result = await asyncio.to_thread(self.pipeline.segmentation_model.predict, image)
        mask = result["mask"] if result.get("mask_available") else None
        return {
            "raw": result,
            "mask": mask,
            "summary": self._mask_summary(mask, result),
        }

    async def depth_map(self, image: Image.Image, mask: Image.Image | None = None) -> dict[str, Any]:
        result = await asyncio.to_thread(self.pipeline.depth_model.predict, image, mask)
        preview = result.get("depth_full_preview") or result.get("depth_canvas_preview")
        if preview is None and result.get("depth") is not None:
            preview = _colorize_depth_preview(result["depth"])
        return {
            "raw": result,
            "depth": result.get("depth"),
            "preview": preview,
            "summary": {
                "depth_available": result.get("depth_available", False),
                "depth_source": result.get("depth_source"),
                "roi": result.get("roi"),
                "warnings": result.get("warnings", []),
            },
        }

    async def classify_condition(self, image: Image.Image) -> dict:
        return await asyncio.to_thread(self.pipeline.predict_condition, image)

    async def predict_severity(
        self,
        image: Image.Image,
        condition: str,
        mask: Image.Image | None,
        depth: Image.Image | None,
    ) -> dict:
        mask_status = "accepted" if mask is not None else "none"
        depth_status = "accepted" if depth is not None else "rejected"
        return await asyncio.to_thread(
            self.pipeline.predict_severity,
            image,
            condition,
            "model_accepted",
            mask_status,
            depth_status,
            mask,
            depth,
        )

    async def extract_visual_evidence(
        self,
        image: Image.Image,
        condition: str | None,
        stage: str | None,
        mask_summary: dict[str, Any] | None,
        depth_summary: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self.visual_evidence_extractor.extract,
            image,
            condition=condition,
            stage=stage,
            mask_summary=mask_summary,
            depth_summary=depth_summary,
        )

    async def answer_visual_followup(
        self,
        image: Image.Image,
        *,
        question: str,
        condition: str | None,
        stage: str | None,
        current_visual_evidence: dict[str, Any] | None,
        mask_summary: dict[str, Any] | None,
        depth_summary: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self.visual_evidence_extractor.answer_followup,
            image,
            question=question,
            condition=condition,
            stage=stage,
            current_visual_evidence=current_visual_evidence,
            mask_summary=mask_summary,
            depth_summary=depth_summary,
        )

    async def retrieve_docs(
        self,
        query: str,
        condition: str | None,
        top_k: int = 4,
        allowed_doc_ids: list[str] | None = None,
    ) -> list[DocumentChunk]:
        if self.config.literature_backend == "local":
            memory = self.memory or ClinicalMemory(self.config)
            return await asyncio.to_thread(
                memory.retrieve,
                query,
                condition,
                top_k,
                allowed_doc_ids,
            )
        if self.config.literature_backend != "pubmed_mcp":
            raise RuntimeError(
                f"Unsupported literature retrieval backend: {self.config.literature_backend}"
            )
        # PubMed MCP is the runtime evidence source. Policy doc IDs are local
        # corpus hints, so they are intentionally not applied to live PubMed.
        _ = allowed_doc_ids
        return await self.literature_retriever.retrieve(query, condition, top_k)

    async def ask_clinician(self, question: str, patient_context: dict[str, Any]) -> str:
        lower = question.lower()
        if "diabetic" in lower and "is_diabetic" in patient_context:
            return str(patient_context["is_diabetic"])
        if "long" in lower and "wound_age_days" in patient_context:
            return f"{patient_context['wound_age_days']} days"
        if "treatment" in lower and "prior_treatment" in patient_context:
            return str(patient_context["prior_treatment"])
        if "bone" in lower and "exposed_bone_or_tendon" in patient_context:
            return str(patient_context["exposed_bone_or_tendon"])
        return "Unknown from provided context."

    @staticmethod
    def _mask_summary(mask: Image.Image | None, result: dict) -> dict[str, Any]:
        if mask is None:
            return {
                "mask_available": False,
                "mask_source": result.get("mask_source"),
                "warnings": result.get("warnings", []),
            }
        mask_array = np.asarray(mask.convert("L"), dtype=np.uint8)
        area_pixels = int(np.count_nonzero(mask_array >= 128))
        mask_fraction = float(area_pixels / mask_array.size) if mask_array.size else 0.0
        return {
            "mask_available": True,
            "mask_source": result.get("mask_source"),
            "mask_area_pixels": area_pixels,
            "mask_fraction": mask_fraction,
            "warnings": result.get("warnings", []),
        }


def _colorize_depth_preview(depth: Image.Image, colormap: str = "inferno") -> Image.Image:
    depth_l = depth.convert("L")
    depth_array = np.asarray(depth_l, dtype=np.float32) / 255.0

    try:
        import matplotlib.colormaps as colormaps

        cmap = colormaps[colormap]
        rgb = (cmap(depth_array)[:, :, :3] * 255).round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(rgb, mode="RGB")
    except Exception:
        return ImageOps.colorize(
            depth_l,
            black="#000004",
            mid="#bc3754",
            white="#fcffa4",
        )


TOOL_SCHEMAS = [
    {
        "name": "segment_wound",
        "description": "Run the existing U-Net++ wound segmentation model and return a soft mask.",
    },
    {
        "name": "depth_map",
        "description": "Run Depth Anything V2 and return a relative depth map plus preview.",
    },
    {
        "name": "classify_condition",
        "description": "Run the existing 18-class ConvNeXt-Tiny condition classifier.",
    },
    {
        "name": "predict_severity",
        "description": "Route and run DFU or PI severity ConvNeXt variants.",
    },
    {
        "name": "extract_visual_evidence",
        "description": "Run an OpenAI vision model to extract visible wound findings for agent evaluation.",
    },
    {
        "name": "answer_visual_followup",
        "description": (
            "Ask the OpenAI vision model a targeted evaluator follow-up question "
            "about the wound image."
        ),
    },
    {
        "name": "retrieve_docs",
        "description": "Retrieve PubMed literature through the PubMed clinical MCP for staging verification.",
    },
    {
        "name": "ask_clinician",
        "description": "Ask a targeted clarifying question or return simulated context in demo mode.",
    },
]
