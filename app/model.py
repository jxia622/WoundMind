from pathlib import Path
import json

import torch
from torch import nn
from torchvision import models


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "convnext_tiny_best.pt"
CLASS_MAP_PATH = PROJECT_ROOT / "app" / "condition_class_map.json"

NUM_CLASSES = 18
DROPOUT_P = 0.4


def load_class_map(class_map_path: Path = CLASS_MAP_PATH) -> dict[int, str]:
    """
    Load class index to class name mapping.

    JSON stores keys as strings, so we convert them back to integers.
    """
    if not class_map_path.exists():
        raise FileNotFoundError(f"Class map not found: {class_map_path}")

    with open(class_map_path, "r") as f:
        raw_class_map = json.load(f)

    idx_to_class = {int(k): v for k, v in raw_class_map.items()}

    expected_indices = list(range(len(idx_to_class)))
    actual_indices = sorted(idx_to_class.keys())

    if actual_indices != expected_indices:
        raise ValueError(
            f"Class map indices must be continuous from 0 to {len(idx_to_class) - 1}. "
            f"Found: {actual_indices}"
        )

    return idx_to_class


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    """
    Rebuild the exact ConvNeXt-Tiny architecture used during training.
    """
    model = models.convnext_tiny(weights=None)

    in_features = model.classifier[2].in_features

    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=DROPOUT_P),
        nn.Linear(in_features, num_classes),
    )

    return model


def load_model(
    checkpoint_path: Path = CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Load the trained ConvNeXt-Tiny model for inference.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(num_classes=NUM_CLASSES)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model
