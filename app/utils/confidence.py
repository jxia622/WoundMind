import math


def confidence_from_probabilities(probabilities: list[float]) -> tuple[str, float, float]:
    sorted_probs = sorted(probabilities, reverse=True)
    p1 = sorted_probs[0] if sorted_probs else 0.0
    p2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    margin = p1 - p2

    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probabilities)
    normalized_entropy = entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0

    if p1 >= 0.80 and margin >= 0.30:
        confidence_level = "High"
    elif p1 >= 0.60 and margin >= 0.15:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"

    if normalized_entropy > 0.75:
        confidence_level = "Low"

    return confidence_level, margin, normalized_entropy


def uncertainty_level_from_entropy(normalized_entropy: float) -> str:
    if normalized_entropy >= 0.75:
        return "high"
    if normalized_entropy >= 0.45:
        return "moderate"
    return "low"
