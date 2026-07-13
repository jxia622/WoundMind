from typing import Any, Literal

from pydantic import BaseModel, Field


AnalysisStatus = Literal["in_progress", "completed", "aborted"]
ConditionSource = Literal["model_accepted", "user_override"]
MaskStatus = Literal["accepted", "custom", "none", "aborted"]
DepthStatus = Literal["accepted", "rejected", "aborted"]


class ValidationResult(BaseModel):
    passed: bool
    warnings: list[str] = Field(default_factory=list)
    blur_score: float | None = None
    width: int | None = None
    height: int | None = None


class PredictionItem(BaseModel):
    rank: int
    class_index: int
    class_name: str
    probability: float


class ConditionPredictionResponse(BaseModel):
    case_id: str
    top3: list[PredictionItem]
    top1_label: str
    confidence: float
    confidence_level: str | None = None
    uncertainty_level: str | None = None
    top2_probability_margin: float | None = None
    normalized_entropy: float | None = None
    model_version: str
    disclaimer: str


class PreviewImage(BaseModel):
    mime_type: str = "image/png"
    data: str
    width: int
    height: int


class MaskGenerationResponse(BaseModel):
    case_id: str
    mask_available: bool
    mask_preview: PreviewImage | None = None
    mask_status: MaskStatus | None = None
    mask_source: Literal["model", "user"] | None = None
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str


class DepthGenerationResponse(BaseModel):
    case_id: str
    depth_available: bool
    depth_preview: PreviewImage | None = None
    depth_status: DepthStatus | None = None
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str


class SeverityPredictionResponse(BaseModel):
    severity_available: bool
    selected_condition: str
    severity_prediction: str | None = None
    severity_confidence: float | None = None
    model_used: str | None = None
    input_channels_used: list[str] = Field(default_factory=list)
    mask_status: MaskStatus | None = None
    depth_status: DepthStatus | None = None
    routing_explanation: str | None = None
    class_probabilities: dict[str, float] | None = None
    message: str | None = None
    disclaimer: str


class PipelineState(BaseModel):
    case_id: str
    validation_result: ValidationResult | None = None
    condition_model_top3: list[PredictionItem] = Field(default_factory=list)
    condition_model_top1: str | None = None
    user_selected_condition: str | None = None
    condition_source: ConditionSource | None = None
    mask_status: MaskStatus | None = None
    depth_status: DepthStatus | None = None
    severity_model_used: str | None = None
    severity_prediction: str | None = None
    analysis_status: AnalysisStatus = "in_progress"


class FeedbackPayload(BaseModel):
    case_id: str
    model_condition: str | None = None
    user_condition: str | None = None
    mask_choice: str | None = None
    depth_choice: str | None = None
    severity_prediction: str | None = None
    user_corrected_severity: str | None = None
    comment: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
