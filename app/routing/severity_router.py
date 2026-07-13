from app.config import (
    DISCLAIMER,
    SUPPORTED_DFU_CONDITION_ALIASES,
    SUPPORTED_PRESSURE_INJURY_ALIASES,
    SUPPORTED_SEVERITY_CONDITION_ALIASES,
)


def normalize_condition(condition: str) -> str:
    return condition.strip().lower().replace(" ", "_")


def is_supported_severity_condition(condition: str) -> bool:
    normalized = normalize_condition(condition)
    readable = condition.strip().lower()
    return normalized in SUPPORTED_SEVERITY_CONDITION_ALIASES or readable in SUPPORTED_SEVERITY_CONDITION_ALIASES


def severity_family(condition: str) -> str | None:
    normalized = normalize_condition(condition)
    readable = condition.strip().lower()

    if normalized in SUPPORTED_DFU_CONDITION_ALIASES or readable in SUPPORTED_DFU_CONDITION_ALIASES:
        return "dfu"

    if normalized in SUPPORTED_PRESSURE_INJURY_ALIASES or readable in SUPPORTED_PRESSURE_INJURY_ALIASES:
        return "pressure_injury"

    return None


def route_severity_model(condition: str, mask_status: str, depth_status: str) -> dict:
    family = severity_family(condition)

    if family is None:
        return {
            "severity_available": False,
            "selected_condition": condition,
            "message": "Severity model for this condition is still under development.",
            "disclaimer": DISCLAIMER,
        }

    if mask_status == "aborted" or depth_status == "aborted":
        return {
            "severity_available": False,
            "selected_condition": condition,
            "analysis_status": "aborted",
            "message": "Analysis was aborted before severity prediction.",
            "disclaimer": DISCLAIMER,
        }

    mask_used = mask_status in {"accepted", "custom"}
    depth_used = depth_status == "accepted"

    input_channels_used = ["rgb"]
    if mask_used:
        input_channels_used.append("mask")
    if depth_used:
        input_channels_used.append("depth")

    if family == "dfu":
        if mask_used and depth_used:
            model_used = "dfu_rgb_depth_mask"
            explanation = "User accepted both segmentation mask and depth map, so the RGB+mask+depth severity model was used."
        elif mask_used and not depth_used:
            model_used = "dfu_rgb_mask"
            explanation = "User accepted or supplied a mask and rejected depth, so the RGB+mask severity model was used."
        elif not mask_used and depth_used:
            model_used = "dfu_rgb_depth"
            explanation = "User chose no mask and accepted depth, so the RGB+depth severity model was used."
        else:
            model_used = "dfu_rgb"
            explanation = "User chose no mask and rejected depth, so the RGB-only severity model was used."
    else:
        # TODO: Replace with PI-specific model names once PI severity checkpoints are packaged.
        if mask_used and depth_used:
            model_used = "pi_rgb_depth_mask"
        elif mask_used and not depth_used:
            model_used = "pi_rgb_mask"
        elif not mask_used and depth_used:
            model_used = "pi_rgb_depth"
        else:
            model_used = "pi_rgb"
        explanation = "Pressure injury severity routing is scaffolded; plug in PI-specific severity weights before production use."

    return {
        "severity_available": True,
        "selected_condition": condition,
        "model_used": model_used,
        "input_channels_used": input_channels_used,
        "routing_explanation": explanation,
        "disclaimer": DISCLAIMER,
    }
