from pathlib import Path

import torch
from torch import nn
from torchvision import models

from app.config import (
    DFU_RGB_CHECKPOINT_PATH,
    DFU_RGB_DEPTH_CHECKPOINT_PATH,
    DFU_RGB_DEPTH_MASK_CHECKPOINT_PATH,
    DFU_RGB_MASK_CHECKPOINT_PATH,
    DFU_SEVERITY_LABELS,
    PI_RGB_CHECKPOINT_PATH,
    PI_RGB_DEPTH_CHECKPOINT_PATH,
    PI_RGB_DEPTH_MASK_CHECKPOINT_PATH,
    PI_RGB_MASK_CHECKPOINT_PATH,
    PI_SEVERITY_LABELS,
)
from app.preprocessing.transforms import preprocess_rgb_mask_depth


def _expand_conv2d(old_conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )

    with torch.no_grad():
        new_conv.weight[:, :3].copy_(old_conv.weight)
        mean_rgb = old_conv.weight.mean(dim=1, keepdim=True)
        for channel_idx in range(3, in_channels):
            new_conv.weight[:, channel_idx:channel_idx + 1].copy_(mean_rgb)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    return new_conv


def _build_convnext_classifier(in_channels: int, num_classes: int = 5) -> nn.Module:
    model = models.convnext_tiny(weights=None)

    if in_channels != 3:
        model.features[0][0] = _expand_conv2d(model.features[0][0], in_channels)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def _load_state_dict_compatible(model: nn.Module, state_dict: dict) -> None:
    state_dict = dict(state_dict)

    if "classifier.2.weight" in state_dict and "classifier.2.1.weight" not in state_dict:
        state_dict["classifier.2.1.weight"] = state_dict.pop("classifier.2.weight")
        state_dict["classifier.2.1.bias"] = state_dict.pop("classifier.2.bias")

    model.load_state_dict(state_dict)


class SeverityModelManager:
    """
    Routes severity inference to ConvNeXt DFU/PI model variants.

    Missing models fall back to mock output so the API contract remains stable
    during frontend integration.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: dict[str, nn.Module] = {}
        self.model_errors: dict[str, str] = {}

        self._register_model("dfu_rgb", DFU_RGB_CHECKPOINT_PATH, in_channels=3, num_classes=5)
        self._register_model("dfu_rgb_mask", DFU_RGB_MASK_CHECKPOINT_PATH, in_channels=4, num_classes=5)
        self._register_model("dfu_rgb_depth", DFU_RGB_DEPTH_CHECKPOINT_PATH, in_channels=4, num_classes=5)
        self._register_model("dfu_rgb_depth_mask", DFU_RGB_DEPTH_MASK_CHECKPOINT_PATH, in_channels=5, num_classes=5)
        self._register_model("pi_rgb", PI_RGB_CHECKPOINT_PATH, in_channels=3, num_classes=6)
        self._register_model("pi_rgb_mask", PI_RGB_MASK_CHECKPOINT_PATH, in_channels=4, num_classes=6)
        self._register_model("pi_rgb_depth", PI_RGB_DEPTH_CHECKPOINT_PATH, in_channels=4, num_classes=6)
        self._register_model("pi_rgb_depth_mask", PI_RGB_DEPTH_MASK_CHECKPOINT_PATH, in_channels=5, num_classes=6)

    def _register_model(
        self,
        model_name: str,
        checkpoint_path: Path,
        in_channels: int,
        num_classes: int,
    ) -> None:
        try:
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)

            model = _build_convnext_classifier(in_channels=in_channels, num_classes=num_classes)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
            _load_state_dict_compatible(model, state_dict)
            model.to(self.device)
            model.eval()
            self.models[model_name] = model
        except Exception as e:
            self.model_errors[model_name] = str(e)

    def predict(
        self,
        image,
        selected_condition: str,
        model_used: str,
        input_channels_used: list[str],
        mask=None,
        depth=None,
    ) -> dict:
        labels = PI_SEVERITY_LABELS if "pressure" in selected_condition.lower() else DFU_SEVERITY_LABELS
        model = self.models.get(model_used)

        if model is None:
            probabilities = self._mock_probabilities(labels)
            return self._format_result(
                labels=labels,
                probabilities=probabilities,
                model_used=model_used,
                input_channels_used=input_channels_used,
                mock=True,
                warning=f"Model weights not loaded for {model_used}: {self.model_errors.get(model_used, 'unknown reason')}",
            )

        input_mask = mask if "mask" in input_channels_used else None
        input_depth = depth if "depth" in input_channels_used else None
        tensor = preprocess_rgb_mask_depth(image, mask=input_mask, depth=input_depth).to(self.device)

        with torch.no_grad():
            logits = model(tensor)
            probabilities_tensor = torch.softmax(logits, dim=1).squeeze(0).cpu()

        probabilities = probabilities_tensor.tolist()
        return self._format_result(
            labels=labels,
            probabilities=probabilities,
            model_used=model_used,
            input_channels_used=input_channels_used,
            mock=False,
            warning=None,
        )

    @staticmethod
    def _mock_probabilities(labels: list[str]) -> list[float]:
        if len(labels) == 6:
            return [0.05, 0.12, 0.18, 0.40, 0.18, 0.07]

        base = [0.08, 0.16, 0.52, 0.18, 0.06]
        return base[:len(labels)]

    @staticmethod
    def _format_result(
        labels: list[str],
        probabilities: list[float],
        model_used: str,
        input_channels_used: list[str],
        mock: bool,
        warning: str | None,
    ) -> dict:
        best_index = max(range(len(probabilities)), key=lambda idx: probabilities[idx])
        result = {
            "severity_prediction": labels[best_index],
            "severity_confidence": float(probabilities[best_index]),
            "model_used": model_used,
            "input_channels_used": input_channels_used,
            "class_probabilities": {
                label: float(prob) for label, prob in zip(labels, probabilities)
            },
            "mock_output": mock,
        }
        if warning:
            result["warning"] = warning
        return result
