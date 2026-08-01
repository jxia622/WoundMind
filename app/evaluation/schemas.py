from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


EvaluationRoute = Literal[
    "ASK_VERBALIZER",
    "ASK_PUBAGENT",
    "COMMIT_ASSESSMENT",
]
FinalStatus = Literal["SUPPORTED", "FLAGGED", "INSUFFICIENT_EVIDENCE"]


class HiddenModelOutput(BaseModel):
    condition: str | None = None
    stage: str | None = None
    confidence: float | None = None
    condition_confidence: float | None = None
    stage_confidence: float | None = None


class InitialVerbalization(BaseModel):
    description: str
    directly_observed: list[str] = Field(default_factory=list)
    deterministically_derived: list[str] = Field(default_factory=list)
    model_predicted: list[str] = Field(default_factory=list)
    unavailable: list[str] = Field(default_factory=list)


class VerbalizerAnswer(BaseModel):
    question: str
    answer: str
    directly_observed: list[str] = Field(default_factory=list)
    deterministically_derived: list[str] = Field(default_factory=list)
    model_predicted: list[str] = Field(default_factory=list)
    unavailable: bool = False
    unavailable_reason: str | None = None


class CitationMetadata(BaseModel):
    title: str | None = None
    source: str | None = None
    url: str | None = None
    authors: list[str] = Field(default_factory=list)
    journal: str | None = None
    year: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    doi: str | None = None


class PubAgentEvidence(BaseModel):
    quote: str
    source_section: str | None = None
    citation: CitationMetadata


class PubAgentResponse(BaseModel):
    question: str
    summary: str
    sufficient: bool = False
    evidence_quality_note: str = ""
    evidence: list[PubAgentEvidence] = Field(default_factory=list)
    citations: list[CitationMetadata] = Field(default_factory=list)
    error: str | None = None


class EvaluationAction(BaseModel):
    action: EvaluationRoute
    question: str | None = None
    current_condition_hypothesis: str | None = None
    current_stage_hypothesis: str | None = None
    independent_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    alternative_hypotheses: list[str] = Field(default_factory=list)
    rationale: str

    @model_validator(mode="after")
    def validate_route(self) -> "EvaluationAction":
        if self.action in {"ASK_VERBALIZER", "ASK_PUBAGENT"} and not (
            self.question and self.question.strip()
        ):
            raise ValueError(f"{self.action} requires a focused question.")
        return self


class EvidenceLedger(BaseModel):
    current_condition_hypothesis: str | None = None
    current_stage_hypothesis: str | None = None
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    clinical_evidence: list[PubAgentResponse] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    alternative_hypotheses: list[str] = Field(default_factory=list)
    next_action: EvaluationRoute | None = None
    next_question: str | None = None


class IndependentAssessment(BaseModel):
    condition: str | None = None
    stage: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    key_supporting_evidence: list[str] = Field(default_factory=list)
    key_contradicting_evidence: list[str] = Field(default_factory=list)
    remaining_uncertainties: list[str] = Field(default_factory=list)
    clinical_evidence_used: list[PubAgentResponse] = Field(default_factory=list)


class AdjudicationDecision(BaseModel):
    condition_agreement: bool | None = None
    stage_agreement: bool | None = None
    discrepancies: list[str] = Field(default_factory=list)
    evidence_consistent: bool | None = None
    clinically_meaningful_disagreement: bool | None = None
    status: FinalStatus
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class ComparisonResult(BaseModel):
    condition_agreement: bool | None = None
    stage_agreement: bool | None = None
    evidence_consistent: bool | None = None
    clinically_meaningful_disagreement: bool | None = None
    discrepancies: list[str] = Field(default_factory=list)


class FinalEvaluation(BaseModel):
    status: FinalStatus
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class EvaluationEvidence(BaseModel):
    supporting: list[str] = Field(default_factory=list)
    contradicting: list[str] = Field(default_factory=list)
    clinical_evidence: list[PubAgentResponse] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)


class AuditEvent(BaseModel):
    sequence: int
    node: str
    iteration: int
    route: str | None = None
    question: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    case_id: str
    model_output: HiddenModelOutput
    independent_evaluation: IndependentAssessment
    comparison: ComparisonResult
    evidence: EvaluationEvidence
    final_evaluation: FinalEvaluation
    iterations: int
    max_iterations: int
    verbalizer_history: list[VerbalizerAnswer] = Field(default_factory=list)
    pubagent_history: list[PubAgentResponse] = Field(default_factory=list)
    audit_log: list[AuditEvent] = Field(default_factory=list)


class BlindedVerbalizerContext(BaseModel):
    case_evidence: dict[str, Any]
    initial_verbalization: InitialVerbalization | None = None
    prior_answers: list[VerbalizerAnswer] = Field(default_factory=list)


class BlindedOrchestratorContext(BaseModel):
    case_evidence: dict[str, Any]
    initial_verbalization: InitialVerbalization
    verbalizer_history: list[VerbalizerAnswer] = Field(default_factory=list)
    pubagent_history: list[PubAgentResponse] = Field(default_factory=list)
    evidence_ledger: EvidenceLedger
    iteration: int
    max_iterations: int


class AdjudicationContext(BaseModel):
    independent_assessment: IndependentAssessment
    model_output: HiddenModelOutput
    evidence_ledger: EvidenceLedger
