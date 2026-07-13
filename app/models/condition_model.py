from pathlib import Path
import json
import time

import torch
from torch import nn
from torchvision import models

from app.config import CONDITION_CHECKPOINT_PATH, CONDITION_CLASS_MAP_PATH, CONDITION_TEMPERATURE_PATH
from app.utils.confidence import confidence_from_probabilities, uncertainty_level_from_entropy
from app.preprocessing.transforms import preprocess_image

NUM_CLASSES = 18
DROPOUT_P = 0.4


def load_class_map(class_map_path: Path = CONDITION_CLASS_MAP_PATH) -> dict[int, str]:
    if not class_map_path.exists():
        raise FileNotFoundError(f"Class map not found: {class_map_path}")

    with open(class_map_path, "r") as f:
        raw_class_map = json.load(f)

    return {int(k): v for k, v in raw_class_map.items()}


def build_condition_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=DROPOUT_P),
        nn.Linear(in_features, num_classes),
    )
    return model


def _load_state_dict_compatible(model: nn.Module, state_dict: dict) -> None:
    state_dict = dict(state_dict)

    if "classifier.2.weight" in state_dict and "classifier.2.1.weight" not in state_dict:
        state_dict["classifier.2.1.weight"] = state_dict.pop("classifier.2.weight")
        state_dict["classifier.2.1.bias"] = state_dict.pop("classifier.2.bias")

    model.load_state_dict(state_dict)


def load_condition_model(
    checkpoint_path: Path = CONDITION_CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Condition checkpoint not found: {checkpoint_path}")

    model = build_condition_model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    _load_state_dict_compatible(model, state_dict)

    model.to(device)
    model.eval()
    return model


def load_temperature(default_temperature: float = 1.0) -> float:
    if not CONDITION_TEMPERATURE_PATH.exists():
        return default_temperature

    with open(CONDITION_TEMPERATURE_PATH, "r") as f:
        calibration = json.load(f)

    temperature = float(calibration.get("temperature", default_temperature))
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    return temperature


class ConditionModel:
    """
    ConvNeXt-Tiny condition classifier.
    """

    def __init__(
        self,
        checkpoint_path: Path = CONDITION_CHECKPOINT_PATH,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.idx_to_class = load_class_map()
        self.temperature = load_temperature()
        self.model = load_condition_model(checkpoint_path=checkpoint_path, device=self.device)

    def predict(self, image, top_k: int = 3) -> dict:
        start = time.perf_counter()
        image_tensor = preprocess_image(image).to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits / self.temperature, dim=1)

        elapsed = time.perf_counter() - start
        top_probs, top_indices = torch.topk(
            probabilities,
            k=min(top_k, len(self.idx_to_class)),
            dim=1,
        )

        probabilities_list = probabilities.squeeze(0).cpu().tolist()
        confidence_level, margin, normalized_entropy = confidence_from_probabilities(probabilities_list)

        top_predictions = []
        for rank, (idx, prob) in enumerate(
            zip(top_indices.squeeze(0).cpu().tolist(), top_probs.squeeze(0).cpu().tolist()),
            start=1,
        ):
            top_predictions.append({
                "rank": rank,
                "class_index": int(idx),
                "class_name": self.idx_to_class[int(idx)],
                "probability": float(prob),
            })

        top1 = top_predictions[0]
        return {
            "top3": top_predictions,
            "top1_label": top1["class_name"],
            "top1_index": top1["class_index"],
            "confidence": top1["probability"],
            "confidence_level": confidence_level,
            "uncertainty_level": uncertainty_level_from_entropy(normalized_entropy),
            "top2_probability_margin": margin,
            "normalized_entropy": normalized_entropy,
            "prediction_time_seconds": elapsed,
            "temperature": self.temperature,
        }
