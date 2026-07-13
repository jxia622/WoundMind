from pathlib import Path
import json
import math
import time

import torch

from app.model import load_model, load_class_map
from app.preprocessing import preprocess_image_path


def load_temperature(default_temperature: float = 1.0) -> float:
    """
    Load temperature calibration value if available.

    If the file does not exist, use default_temperature.
    """
    calibration_path = Path(__file__).resolve().parent / "temperature_calibration.json"

    if not calibration_path.exists():
        return default_temperature

    with open(calibration_path, "r") as f:
        calibration = json.load(f)

    temperature = float(calibration.get("temperature", default_temperature))

    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    return temperature


def confidence_from_probabilities(probabilities: torch.Tensor) -> tuple[str, float, float]:
    """
    Return High/Moderate/Low using top-2 margin plus normalized entropy override.
    """
    sorted_probs, _ = torch.sort(probabilities, descending=True)
    p1 = sorted_probs[0].item()
    p2 = sorted_probs[1].item() if len(sorted_probs) > 1 else 0.0
    margin = p1 - p2

    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum().item()
    normalized_entropy = entropy / math.log(len(probabilities))

    if p1 >= 0.80 and margin >= 0.30:
        confidence = "High"
    elif p1 >= 0.60 and margin >= 0.15:
        confidence = "Moderate"
    else:
        confidence = "Low"

    if normalized_entropy > 0.75:
        confidence = "Low"

    return confidence, margin, normalized_entropy


def predict_image(
    image_path: str | Path,
    model: torch.nn.Module,
    idx_to_class: dict[int, str],
    device: torch.device,
    top_k: int = 3,
    temperature: float = 1.0,
) -> dict:
    """
    Run single-image inference.

    Returns:
        Dictionary containing top prediction, calibrated confidence level, runtime,
        top-2 margin, normalized entropy, and top-k predictions.
    """
    prediction_start_time = time.perf_counter()

    image_tensor = preprocess_image_path(image_path)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits / temperature, dim=1)

    prediction_elapsed_seconds = time.perf_counter() - prediction_start_time

    top_probs, top_indices = torch.topk(
        probabilities,
        k=min(top_k, len(idx_to_class)),
        dim=1,
    )
    probabilities_1d = probabilities.squeeze(0).cpu()
    confidence, margin, normalized_entropy = confidence_from_probabilities(probabilities_1d)

    top_probs = top_probs.squeeze(0).cpu().tolist()
    top_indices = top_indices.squeeze(0).cpu().tolist()

    top_predictions = []

    for rank, (idx, prob) in enumerate(zip(top_indices, top_probs), start=1):
        top_predictions.append({
            "rank": rank,
            "class_index": idx,
            "class_name": idx_to_class[idx],
            "probability": prob,
        })

    best_prediction = top_predictions[0]

    return {
        "predicted_class_index": best_prediction["class_index"],
        "predicted_class_name": best_prediction["class_name"],
        "predicted_probability": best_prediction["probability"],
        "confidence_level": confidence,
        "prediction_time_seconds": prediction_elapsed_seconds,
        "top2_probability_margin": margin,
        "normalized_entropy": normalized_entropy,
        "temperature": temperature,
        "top_predictions": top_predictions,
    }


def load_inference_components():
    """
    Load model, class map, device, and temperature once.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx_to_class = load_class_map()
    model = load_model(device=device)
    temperature = load_temperature()

    return model, idx_to_class, device, temperature
