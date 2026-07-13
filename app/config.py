from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
FEEDBACK_LOG_PATH = PROJECT_ROOT / "feedback.jsonl"

API_TITLE = "WoundMind Diagnostic Assistant API"
API_VERSION = "1.0.0"
MODEL_VERSION = "wound-multistage-prototype-v1"
DISCLAIMER = "Research prototype. Not for clinical diagnosis."

ALLOWED_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MIN_IMAGE_WIDTH = 224
MIN_IMAGE_HEIGHT = 224
BLUR_WARNING_THRESHOLD = 80.0

CONDITION_CLASS_MAP_PATH = APP_DIR / "condition_class_map.json"
CONDITION_TEMPERATURE_PATH = APP_DIR / "temperature_calibration.json"
CONDITION_CHECKPOINT_PATH = CHECKPOINT_DIR / "convnext_tiny_best.pt"

SEGMENTATION_CHECKPOINT_PATH = CHECKPOINT_DIR / "unetpp_efficientnet_b0_512.pt"
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Base-hf"

DFU_RGB_CHECKPOINT_PATH = CHECKPOINT_DIR / "DFU_convnext_tiny" / "convnext_tiny_best.pt"
DFU_RGB_MASK_CHECKPOINT_PATH = CHECKPOINT_DIR / "DFU_convnext_tiny_mask4" / "convnext_tiny_mask4_best.pt"
DFU_RGB_DEPTH_CHECKPOINT_PATH = CHECKPOINT_DIR / "DFU_convnext_tiny_depth4" / "convnext_tiny_depth4_best.pt"
DFU_RGB_DEPTH_MASK_CHECKPOINT_PATH = CHECKPOINT_DIR / "DFU_convnext_tiny_depth_mask5" / "convnext_tiny_depth_mask5_best.pt"

PI_RGB_CHECKPOINT_PATH = CHECKPOINT_DIR / "PI_convnext_tiny" / "convnext_tiny_pi_best.pt"
PI_RGB_MASK_CHECKPOINT_PATH = CHECKPOINT_DIR / "PI_convnext_tiny_mask4" / "convnext_tiny_mask4_pi_best.pt"
PI_RGB_DEPTH_CHECKPOINT_PATH = CHECKPOINT_DIR / "PI_convnext_tiny_depth4" / "convnext_tiny_depth4_pi_best.pt"
PI_RGB_DEPTH_MASK_CHECKPOINT_PATH = CHECKPOINT_DIR / "PI_convnext_tiny_depth_mask5" / "convnext_tiny_depth_mask5_pi_best.pt"

SUPPORTED_DFU_CONDITION_ALIASES = {
    "diabetic foot ulcer",
    "diabetic foot ulcers",
    "diabetic_foot_ulcers_(dfu)",
    "dfu",
}

SUPPORTED_PRESSURE_INJURY_ALIASES = {
    "pressure injury",
    "pressure injuries",
    "pressure_injury_(pi)",
    "pi",
}

SUPPORTED_SEVERITY_CONDITION_ALIASES = (
    SUPPORTED_DFU_CONDITION_ALIASES | SUPPORTED_PRESSURE_INJURY_ALIASES
)

DFU_SEVERITY_LABELS = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"]
PI_SEVERITY_LABELS = ["Stage 0", "Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5"]
