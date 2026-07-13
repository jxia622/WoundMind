from uuid import uuid4

import numpy as np
from PIL import Image

from app.config import DISCLAIMER, MODEL_VERSION
from app.models import ConditionModel, DepthModel, SegmentationModel, SeverityModelManager
from app.preprocessing.validation import validate_image_basic
from app.routing.severity_router import route_severity_model
from app.schemas import PipelineState
from app.utils.encoding import pil_to_base64_png


def blue_mask_overlay(image: Image.Image, mask: Image.Image, alpha: float = 0.42) -> Image.Image:
    image_rgb = image.convert("RGB")
    mask_l = mask.convert("L")
    if mask_l.size != image_rgb.size:
        mask_l = mask_l.resize(image_rgb.size)

    overlay = Image.new("RGB", image_rgb.size, (37, 99, 235))
    mask_alpha = mask_l.point(lambda px: int(px * alpha))
    return Image.composite(overlay, image_rgb, mask_alpha)


class WoundAnalysisPipeline:
    """
    Coordinates model wrappers and staged human-in-the-loop decisions.
    """

    def __init__(self) -> None:
        self.condition_model = ConditionModel()
        self.segmentation_model = SegmentationModel(device=self.condition_model.device)
        self.depth_model = DepthModel(device=self.condition_model.device)
        self.severity_models = SeverityModelManager(device=self.condition_model.device)

    @staticmethod
    def new_case_id() -> str:
        return str(uuid4())

    def validate_image(self, image: Image.Image) -> dict:
        return validate_image_basic(image).model_dump()

    def predict_condition(self, image: Image.Image, case_id: str | None = None) -> dict:
        result = self.condition_model.predict(image, top_k=3)
        return {
            "case_id": case_id or self.new_case_id(),
            "top3": result["top3"],
            "top1_label": result["top1_label"],
            "confidence": result["confidence"],
            "confidence_level": result["confidence_level"],
            "uncertainty_level": result["uncertainty_level"],
            "top2_probability_margin": result["top2_probability_margin"],
            "normalized_entropy": result["normalized_entropy"],
            "model_version": MODEL_VERSION,
            "disclaimer": DISCLAIMER,
        }

    def confirm_condition(
        self,
        case_id: str,
        model_condition: str,
        user_selected_condition: str | None,
        abort: bool = False,
    ) -> dict:
        if abort:
            return PipelineState(
                case_id=case_id,
                condition_model_top1=model_condition,
                user_selected_condition=user_selected_condition,
                analysis_status="aborted",
            ).model_dump()

        selected_condition = user_selected_condition or model_condition
        condition_source = "model_accepted" if selected_condition == model_condition else "user_override"

        return PipelineState(
            case_id=case_id,
            condition_model_top1=model_condition,
            user_selected_condition=selected_condition,
            condition_source=condition_source,
            analysis_status="in_progress",
        ).model_dump()

    def generate_mask(self, image: Image.Image, case_id: str | None = None) -> dict:
        result = self.segmentation_model.predict(image)
        mask_array = (
            np.asarray(result["mask"].convert("L"), dtype=np.uint8)
            if result["mask_available"]
            else None
        )
        mask_area_pixels = (
            int(np.count_nonzero(mask_array >= 128))
            if mask_array is not None
            else None
        )
        mask_fraction = (
            float(mask_area_pixels / mask_array.size)
            if mask_array is not None and mask_array.size
            else None
        )
        preview = pil_to_base64_png(result["mask"]) if result["mask_available"] else None
        overlay_preview = (
            pil_to_base64_png(blue_mask_overlay(image, result["mask"]))
            if result["mask_available"]
            else None
        )

        return {
            "case_id": case_id or self.new_case_id(),
            "mask_available": result["mask_available"],
            "mask_preview": preview,
            "mask_overlay_preview": overlay_preview,
            "mask_status": None,
            "mask_source": "model" if result["mask_available"] else None,
            "mask_area_pixels": mask_area_pixels,
            "mask_fraction": mask_fraction,
            "warnings": result.get("warnings", []),
            "disclaimer": DISCLAIMER,
        }

    def generate_depth(self, image: Image.Image, case_id: str | None = None) -> dict:
        mask_result = self.segmentation_model.predict(image)
        mask = mask_result["mask"] if mask_result["mask_available"] else None
        result = self.depth_model.predict(image, mask=mask)
        preview_image = (
            result.get("depth_full_preview")
            or result.get("depth_canvas_preview")
            or result.get("depth")
        )
        preview = pil_to_base64_png(preview_image) if result["depth_available"] else None

        return {
            "case_id": case_id or self.new_case_id(),
            "depth_available": result["depth_available"],
            "depth_preview": preview,
            "depth_status": None,
            "warnings": result.get("warnings", []),
            "depth_source": result.get("depth_source"),
            "depth_scale": result.get("depth_scale"),
            "roi": result.get("roi"),
            "preview_note": "The UI preview is generated from the full image. Severity inference continues to use the segmentation-guided square ROI depth input.",
            "disclaimer": DISCLAIMER,
        }

    def predict_severity(
        self,
        image: Image.Image,
        selected_condition: str,
        condition_source: str,
        mask_status: str,
        depth_status: str,
        mask: Image.Image | None = None,
        depth: Image.Image | None = None,
    ) -> dict:
        route = route_severity_model(
            condition=selected_condition,
            mask_status=mask_status,
            depth_status=depth_status,
        )

        if not route["severity_available"]:
            return route

        if mask_status == "custom" and mask is None:
            raise ValueError("mask_file is required when mask_status is 'custom'.")

        if mask_status == "accepted" and mask is None:
            mask_result = self.segmentation_model.predict(image)
            mask = mask_result["mask"] if mask_result["mask_available"] else None

        if depth_status == "accepted" and depth is None:
            depth_result = self.depth_model.predict(
                image,
                mask=mask,
                include_full_preview=False,
            )
            depth = depth_result["depth"] if depth_result["depth_available"] else None

        model_result = self.severity_models.predict(
            image=image,
            selected_condition=selected_condition,
            model_used=route["model_used"],
            input_channels_used=route["input_channels_used"],
            mask=mask,
            depth=depth,
        )

        return {
            **route,
            "condition_source": condition_source,
            "mask_status": mask_status,
            "depth_status": depth_status,
            "severity_prediction": model_result["severity_prediction"],
            "severity_confidence": model_result["severity_confidence"],
            "class_probabilities": model_result["class_probabilities"],
            "mock_output": model_result.get("mock_output", False),
            "warning": model_result.get("warning"),
        }

    def analyze_default(self, image: Image.Image, case_id: str | None = None) -> dict:
        case_id = case_id or self.new_case_id()
        validation = self.validate_image(image)
        condition = self.predict_condition(image, case_id=case_id)

        selected_condition = condition["top1_label"]
        mask = self.generate_mask(image, case_id=case_id)
        depth = self.generate_depth(image, case_id=case_id)

        severity = self.predict_severity(
            image=image,
            selected_condition=selected_condition,
            condition_source="model_accepted",
            mask_status="accepted" if mask["mask_available"] else "none",
            depth_status="accepted" if depth["depth_available"] else "rejected",
        )

        return {
            "case_id": case_id,
            "validation": validation,
            "condition": condition,
            "mask": mask,
            "depth": depth,
            "severity": severity,
            "analysis_status": "completed" if severity.get("severity_available") else "in_progress",
            "disclaimer": DISCLAIMER,
        }
