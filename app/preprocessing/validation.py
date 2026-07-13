import numpy as np
from PIL import Image

from app.config import BLUR_WARNING_THRESHOLD
from app.schemas import ValidationResult


def _grayscale_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32)


def laplacian_variance(image: Image.Image) -> float:
    gray = _grayscale_array(image)

    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0

    center = gray[1:-1, 1:-1] * -4.0
    up = gray[:-2, 1:-1]
    down = gray[2:, 1:-1]
    left = gray[1:-1, :-2]
    right = gray[1:-1, 2:]
    laplacian = center + up + down + left + right

    return float(laplacian.var())


def validate_image_basic(image: Image.Image) -> ValidationResult:
    warnings = []
    width, height = image.size
    passed = True

    blur_score = laplacian_variance(image)
    if blur_score < BLUR_WARNING_THRESHOLD:
        warnings.append(
            f"Image may be blurry. Laplacian variance {blur_score:.2f} is below threshold {BLUR_WARNING_THRESHOLD:.2f}."
        )

    return ValidationResult(
        passed=passed,
        warnings=warnings,
        blur_score=blur_score,
        width=width,
        height=height,
    )
