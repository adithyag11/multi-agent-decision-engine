"""
The public HTTP surface. Three things worth noting about its shape:

  1. Every route requires an API key (X-API-Key header), scoped to either
     'caller' (submit/read assessments) or 'reviewer' (additionally resolve
     a human escalation) -- see api/auth.py. Before this existed, anyone
     who could reach the service could resolve a human escalation while
     claiming to be any reviewer name they typed into the request body,
     which is a real gap in a system whose whole premise is a defensible
     audit trail.
  2. Every route is a thin producer onto `run_queue`, never a direct caller
     of the LangGraph graph. POST /assessments and POST /.../resume both
     INSERT a row and return 202 immediately; one or more `app/worker.py`
     processes actually invoke the graph. A multi-agent, multi-retry LLM
     pipeline -- one that can also pause for an indefinite, human-timescale
     amount of time on an escalation -- cannot live inside one HTTP
     request/response cycle without risking gateway timeouts, and a crashed
     worker process shouldn't be able to silently orphan a run (see
     worker.py's module docstring for how the queue plus the LangGraph
     checkpoint together make that recoverable).
  3. Status is recovered from Postgres and, when relevant, from LangGraph's
     own checkpointed state (`graph.aget_state`) -- never from in-memory
     Python state, which would vanish on a restart or be wrong behind a
     load balancer with more than one replica.
"""
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from app.agents.graph import get_compiled_graph
from app.api.auth import require_scope
from app.api.schemas import (
    AssessmentAcceptedResponse,
    AssessmentRequest,
    AssessmentStatusResponse,
    AssessmentTraceResponse,
    DecisionOutput,
    PendingHumanReview,
    ResumeDecisionRequest,
)
from app.config import ApiKeyIdentity, settings
from app.db import ledger
from app.db.pool import acquire

router = APIRouter(prefix="/v1/assessments", tags=["assessments"])


def _graph_config(assessment_id: UUID) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": str(assessment_id)},
        # Generous but finite: (max_conflict_retries + 1) full loops through
        # data_engineer -> financial_analyst -> risk_critic, plus the
        # supervisor/finalize/escalation nodes. A run that exceeds this
        # raises GraphRecursionError instead of spinning forever on an
        # unforeseen routing bug. Only actually consulted by worker.py, but
        # kept here (not duplicated) since GET reads state with the same
        # config shape.
        "recursion_limit": (settings.max_conflict_retries + 1) * 3 + 6,
    }


@router.post("", response_model=AssessmentAcceptedResponse, status_code=202)
async def create_assessment(
    payload: AssessmentRequest,
    identity: ApiKeyIdentity = Depends(require_scope("caller")),
) -> AssessmentAcceptedResponse:
    initial_state: dict[str, Any] = {
        "assessment_id": None,  # filled in below once the row exists
        "subject_id": payload.subject_id,
        "subject_type": payload.subject_type,
        "request_type": payload.request_type,
        "fiscal_year": payload.fiscal_year,
        "raw_request": payload.model_dump(),
        "conflict_round": 0,
        "event_seq": 0,
        "status": "pending",
        "audit_trail": [],
    }

    async with acquire() as conn:
        async with conn.transaction():
            assessment_id = await ledger.create_assessment(
                conn,
                external_reference=payload.external_reference,
                subject_type=payload.subject_type,
                subject_id=payload.subject_id,
                request_type=payload.request_type,
                requested_by=identity.name,  # authenticated identity, not a client-supplied string
                request_payload=payload.model_dump(),
            )
            await ledger.set_checkpoint_thread(conn, assessment_id, str(assessment_id))

            initial_state["assessment_id"] = str(assessment_id)
            await conn.execute(
                "INSERT INTO run_queue (assessment_id, kind, payload) VALUES ($1, 'start', $2::jsonb)",
                assessment_id, initial_state,
            )

    return AssessmentAcceptedResponse(
        assessment_id=assessment_id,
        status="pending",
        poll_url=f"/v1/assessments/{assessment_id}",
    )


@router.get("/{assessment_id}", response_model=AssessmentStatusResponse)
async def get_assessment(
    assessment_id: UUID,
    identity: ApiKeyIdentity = Depends(require_scope("caller")),
) -> AssessmentStatusResponse:
    async with acquire() as conn:
        assessment_row = await conn.fetchrow("SELECT * FROM assessments WHERE id = $1", assessment_id)
        if assessment_row is None:
            raise HTTPException(status_code=404, detail="assessment not found")

        decision_row = await conn.fetchrow(
            "SELECT * FROM decisions WHERE assessment_id = $1 ORDER BY version DESC LIMIT 1",
            assessment_id,
        )

    pending_review = None
    if assessment_row["status"] == "escalated":
        graph = get_compiled_graph()
        snapshot = await graph.aget_state(_graph_config(assessment_id))
        for task in snapshot.tasks:
            for intr in task.interrupts:
                payload = intr.value
                pending_review = PendingHumanReview(
                    reason=payload["reason"],
                    disputed_fields=payload["disputed_fields"],
                    conflict_round=payload["conflict_round"],
                    financial_analysis=payload.get("financial_analysis"),
                )

    return AssessmentStatusResponse(
        assessment_id=assessment_row["id"],
        external_reference=assessment_row["external_reference"],
        status=assessment_row["status"],
        decision=DecisionOutput(**dict(decision_row)) if decision_row else None,
        pending_human_review=pending_review,
    )


@router.post("/{assessment_id}/resume", response_model=AssessmentAcceptedResponse, status_code=202)
async def resume_assessment(
    assessment_id: UUID,
    payload: ResumeDecisionRequest,
    identity: ApiKeyIdentity = Depends(require_scope("reviewer")),
) -> AssessmentAcceptedResponse:
    resume_payload = {"decision": payload.decision, "reviewer": identity.name, "notes": payload.notes}

    async with acquire() as conn:
        assessment_row = await conn.fetchrow("SELECT status FROM assessments WHERE id = $1", assessment_id)
        if assessment_row is None:
            raise HTTPException(status_code=404, detail="assessment not found")
        if assessment_row["status"] != "escalated":
            raise HTTPException(
                status_code=409,
                detail=f"assessment is '{assessment_row['status']}', not awaiting human review",
            )
        await conn.execute(
            "INSERT INTO run_queue (assessment_id, kind, payload) VALUES ($1, 'resume', $2::jsonb)",
            assessment_id, resume_payload,
        )

    return AssessmentAcceptedResponse(
        assessment_id=assessment_id,
        status="escalated",
        poll_url=f"/v1/assessments/{assessment_id}",
    )


@router.get("/{assessment_id}/trace", response_model=AssessmentTraceResponse)
async def get_assessment_trace(
    assessment_id: UUID,
    identity: ApiKeyIdentity = Depends(require_scope("caller")),
) -> AssessmentTraceResponse:
    """The full, ordered, hash-chained reasoning trace -- what gets handed
    to an auditor or rendered in a case-review UI."""
    async with acquire() as conn:
        exists = await conn.fetchval("SELECT 1 FROM assessments WHERE id = $1", assessment_id)
        if not exists:
            raise HTTPException(status_code=404, detail="assessment not found")
        events = await ledger.fetch_trace(conn, assessment_id)

    return AssessmentTraceResponse(assessment_id=assessment_id, events=events)
