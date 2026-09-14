"""
Pydantic models for the public API surface. These are intentionally
separate from the internal LangGraph state (app/agents/state.py) and the
DB row shapes (app/db/ledger.py) -- the wire contract is allowed to be more
conservative than the internal representation, and decoupling them means an
internal refactor never becomes a breaking API change by accident.
"""
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


# =============================================================================
# Ingestion request -- POST /v1/assessments
# =============================================================================

class AssessmentRequest(BaseModel):
    # No `requested_by` field: who made this request is derived from the
    # authenticated API key (see api/auth.py), never from a self-reported
    # string in the body -- a caller can no longer claim to be a different
    # system in a ledger meant to be legally defensible.
    external_reference: str = Field(description="Caller's own case/loan/engagement ID")
    subject_type: Literal["counterparty", "issuer", "vendor"]
    subject_id: str = Field(description="Internal entity identifier (e.g. LEI)")
    request_type: Literal["esg_score", "fraud_risk", "combined"]
    fiscal_year: int = Field(ge=1990, le=2100)

    model_config = {
        "json_schema_extra": {
            "example": {
                "external_reference": "LOAN-2026-04831",
                "subject_type": "counterparty",
                "subject_id": "LEI-549300ABCXYZ1234567",
                "request_type": "combined",
                "fiscal_year": 2025,
            }
        }
    }


class AssessmentAcceptedResponse(BaseModel):
    assessment_id: UUID
    status: Literal["pending", "in_progress", "escalated"]
    poll_url: str


# =============================================================================
# Final decision -- GET /v1/assessments/{id}
# =============================================================================

class DecisionOutput(BaseModel):
    esg_score: float | None = None
    esg_rating: str | None = None
    fraud_risk_score: float | None = None
    fraud_risk_level: Literal["low", "medium", "high", "critical"] | None = None
    distress_risk_score: float | None = None
    distress_risk_level: Literal["safe", "grey", "distress"] | None = None
    verdict: Literal["approved", "approved_with_flags", "escalated", "rejected"]
    rationale: str
    computed_by: list[dict[str, Any]]
    reviewed_by_human: bool
    human_reviewer: str | None = None
    conflict_rounds: int


class PendingHumanReview(BaseModel):
    """Surfaced while an assessment is paused on a human_escalation
    interrupt -- what a reviewer's console needs to render the decision."""
    reason: str
    disputed_fields: list[str]
    conflict_round: int
    financial_analysis: dict[str, Any] | None = None


class AssessmentStatusResponse(BaseModel):
    assessment_id: UUID
    external_reference: str
    status: Literal["pending", "in_progress", "escalated", "completed", "failed"]
    decision: DecisionOutput | None = None
    pending_human_review: PendingHumanReview | None = None


# =============================================================================
# Human resume -- POST /v1/assessments/{id}/resume
# =============================================================================

class ResumeDecisionRequest(BaseModel):
    # No `reviewer` field: who resolved the escalation comes from the
    # authenticated 'reviewer'-scoped API key, not a self-reported string --
    # otherwise anyone holding any valid key could resolve an escalation
    # while claiming to be a named human reviewer in the permanent record.
    decision: Literal["approve", "reject"]
    notes: str = ""


# =============================================================================
# Audit trace -- GET /v1/assessments/{id}/trace
# =============================================================================

class AuditEventOut(BaseModel):
    seq: int
    event_type: str
    node_name: str
    agent_name: str
    tool_name: str | None
    reasoning: str | None
    input_summary: dict[str, Any]
    output_summary: dict[str, Any]
    record_hash: str
    created_at: datetime


class AssessmentTraceResponse(BaseModel):
    assessment_id: UUID
    events: list[AuditEventOut]
