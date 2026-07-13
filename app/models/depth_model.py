from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from app.config import DEPTH_MODEL_ID

MASK_THRESHOLD = 0.5
DEPTH_COLORMAP = "inferno"


def depth_array_from_result(result: dict) -> np.ndarray:
    predicted_depth = result.get("predicted_depth")
    if predicted_depth is not None:
        if torch.is_tensor(predicted_depth):
            predicted_depth = predicted_depth.detach().float().cpu().squeeze().numpy()
        depth = np.asarray(predicted_depth, dtype=np.float32).squeeze()
    else:
        depth = np.asarray(result["depth"], dtype=np.float32)

    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth map, got shape {depth.shape}")
    return depth


def normalize_depth_array(depth: np.ndarray) -> tuple[np.ndarray, dict]:
    deep_value = float(np.nanmin(depth))
    shallow_value = float(np.nanmax(depth))
    depth_norm = (depth - deep_value) / (shallow_value - deep_value + 1e-8)
    scale = {
        "shallow_relative": shallow_value,
        "deep_relative": deep_value,
        "units": "relative",
        "lighter_is_shallower": True,
    }
    return depth_norm, scale


def load_depth_model(
    model_id: str = DEPTH_MODEL_ID,
    device: torch.device | None = None,
):
    try:
        from transformers import pipeline
    except ImportError as e:
        raise ImportError("transformers is required for the depth model.") from e

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline_device = 0 if device.type == "cuda" else -1
    return pipeline(task="depth-estimation", model=model_id, device=pipeline_device)


def adaptive_margin_fraction(mask_fraction: float) -> float:
    if mask_fraction <= 0.005:
        return 1.00
    if mask_fraction <= 0.02:
        return 0.75
    if mask_fraction <= 0.05:
        return 0.50
    if mask_fraction <= 0.10:
        return 0.35
    if mask_fraction <= 0.20:
        return 0.25
    return 0.15


def square_roi_from_mask(mask: Image.Image, threshold: float = MASK_THRESHOLD) -> dict:
    mask_prob = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    height, width = mask_prob.shape
    mask_binary = mask_prob > threshold

    if not mask_binary.any():
        return {
            "box": (0, 0, width, height),
            "raw_box": None,
            "mask_fraction": 0.0,
            "margin_fraction": None,
            "used_full_image": True,
        }

    ys, xs = np.where(mask_binary)
    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1

    raw_w = x2 - x1
    raw_h = y2 - y1
    mask_fraction = float(mask_binary.mean())
    margin_fraction = adaptive_margin_fraction(mask_fraction)

    side = max(raw_w, raw_h)
    side = int(round(side * (1.0 + 2.0 * margin_fraction)))
    side = max(side, raw_w, raw_h, 1)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    sx1 = int(round(cx - side / 2.0))
    sy1 = int(round(cy - side / 2.0))
    sx2 = sx1 + side
    sy2 = sy1 + side

    if sx1 < 0:
        sx2 -= sx1
        sx1 = 0
    if sy1 < 0:
        sy2 -= sy1
        sy1 = 0
    if sx2 > width:
        sx1 -= sx2 - width
        sx2 = width
    if sy2 > height:
        sy1 -= sy2 - height
        sy2 = height

    sx1 = max(0, sx1)
    sy1 = max(0, sy1)
    sx2 = min(width, sx2)
    sy2 = min(height, sy2)

    if sx2 <= sx1 or sy2 <= sy1:
        return {
            "box": (0, 0, width, height),
            "raw_box": (x1, y1, x2, y2),
            "mask_fraction": mask_fraction,
            "margin_fraction": margin_fraction,
            "used_full_image": True,
        }

    return {
        "box": (sx1, sy1, sx2, sy2),
        "raw_box": (x1, y1, x2, y2),
        "mask_fraction": mask_fraction,
        "margin_fraction": margin_fraction,
        "used_full_image": False,
    }


class DepthModel:
    """
    Experimental depth-estimation wrapper.

    Depth is an auxiliary research signal only and is not clinically validated.
    """

    def __init__(
        self,
        model_id: str = DEPTH_MODEL_ID,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.pipeline = None
        self.available = False
        self.warnings: list[str] = []

        try:
            self.pipeline = load_depth_model(model_id=model_id, device=self.device)
            self.available = True
        except Exception as e:
            self.warnings.append(
                f"Depth model unavailable; using placeholder depth map. Reason: {e}"
            )

    def predict(
        self,
        image: Image.Image,
        mask: Image.Image | None = None,
        include_full_preview: bool = True,
    ) -> dict:
        if self.pipeline is None:
            depth = self._placeholder_depth(image)
            preview = colorize_depth(depth)
            return {
                "depth": depth,
                "depth_preview": preview,
                "depth_canvas_preview": preview,
                "depth_full_preview": preview,
                "depth_roi": depth,
                "depth_available": True,
                "depth_source": "placeholder",
                "depth_scale": {
                    "shallow_relative": 1.0,
                    "deep_relative": 0.0,
                    "units": "relative",
                    "lighter_is_shallower": True,
                },
                "warnings": self.warnings,
                "model_id": self.model_id,
                "roi": None,
            }

        depth, depth_roi, roi, roi_scale = self._predict_segmentation_guided_depth(
            image=image,
            mask=mask,
        )
        roi_preview = colorize_depth(depth_roi)
        canvas_preview = colorize_depth(depth)
        full_preview = None
        depth_scale = None
        if include_full_preview:
            if roi["used_full_image"]:
                full_preview = canvas_preview
                depth_scale = roi_scale
            else:
                full_depth, depth_scale = self._predict_full_image_depth(image)
                full_preview = colorize_depth(full_depth)
        return {
            "depth": depth,
            "depth_preview": roi_preview,
            "depth_canvas_preview": canvas_preview,
            "depth_full_preview": full_preview,
            "depth_roi": depth_roi,
            "depth_available": True,
            "depth_source": "model",
            "depth_scale": depth_scale,
            "warnings": [],
            "model_id": self.model_id,
            "roi": roi,
        }

    def _predict_segmentation_guided_depth(
        self,
        image: Image.Image,
        mask: Image.Image | None = None,
    ) -> tuple[Image.Image, Image.Image, dict, dict]:
        image_rgb = image.convert("RGB")
        width, height = image_rgb.size

        if mask is None:
            roi = {
                "box": (0, 0, width, height),
                "raw_box": None,
                "mask_fraction": None,
                "margin_fraction": None,
                "used_full_image": True,
            }
        else:
            if mask.size != image_rgb.size:
                mask = mask.resize(image_rgb.size, resample=Image.BILINEAR)
            roi = square_roi_from_mask(mask)

        x1, y1, x2, y2 = roi["box"]
        crop = image_rgb.crop((x1, y1, x2, y2))
        result = self.pipeline(crop)
        depth = depth_array_from_result(result)
        depth_norm, depth_scale = normalize_depth_array(depth)

        depth_crop = Image.fromarray(
            (depth_norm * 255).round().clip(0, 255).astype(np.uint8),
            mode="L",
        )
        depth_roi = depth_crop.copy()
        depth_crop = depth_crop.resize((x2 - x1, y2 - y1), resample=Image.BILINEAR)

        depth_canvas = Image.new("L", image_rgb.size, 0)
        depth_canvas.paste(depth_crop, (x1, y1))
        return depth_canvas, depth_roi, roi, depth_scale

    def _predict_full_image_depth(self, image: Image.Image) -> tuple[Image.Image, dict]:
        """
        Generate a full-frame depth image for UI preview only.

        Severity inference still uses the segmentation-guided depth canvas
        returned by _predict_segmentation_guided_depth.
        """
        result = self.pipeline(image.convert("RGB"))
        depth = depth_array_from_result(result)
        depth_norm, depth_scale = normalize_depth_array(depth)
        depth_image = Image.fromarray(
            (depth_norm * 255).round().clip(0, 255).astype(np.uint8),
            mode="L",
        ).resize(image.size, resample=Image.BILINEAR)
        return depth_image, depth_scale

    @staticmethod
    def _placeholder_depth(image: Image.Image) -> Image.Image:
        width, height = image.size
        gradient = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
        return Image.fromarray(gradient, mode="L")


def colorize_depth(depth: Image.Image, colormap: str = DEPTH_COLORMAP) -> Image.Image:
    """
    Colorize a normalized depth image for preview only.

    The severity models still receive the grayscale depth channel.
    """
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
