"""
Thin, explicit data-access layer for the audit ledger. Every write here maps
1:1 to a table in db/schema.sql. Deliberately NOT an ORM: the ledger's value
comes from the DB-enforced hash chain and immutability triggers (see
schema.sql), and hand-written parameterized SQL keeps that guarantee visible
and auditable rather than hidden behind an abstraction layer.

Every function takes an already-acquired asyncpg connection (or a
transaction) rather than a pool, so callers control transaction boundaries
-- e.g. the Supervisor node writes a `routing_decision` event and updates
`assessments.status` atomically in one transaction.
"""
import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import asyncpg


async def create_assessment(
    conn: asyncpg.Connection,
    *,
    external_reference: str,
    subject_type: str,
    subject_id: str,
    request_type: str,
    requested_by: str,
    request_payload: dict[str, Any],
) -> UUID:
    row = await conn.fetchrow(
        """
        INSERT INTO assessments
            (external_reference, subject_type, subject_id, request_type, requested_by, request_payload)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
        RETURNING id
        """,
        external_reference,
        subject_type,
        subject_id,
        request_type,
        requested_by,
        json.dumps(request_payload),
    )
    return row["id"]


async def set_checkpoint_thread(conn: asyncpg.Connection, assessment_id: UUID, thread_id: str) -> None:
    await conn.execute(
        "UPDATE assessments SET checkpoint_thread_id = $2, updated_at = now() WHERE id = $1",
        assessment_id,
        thread_id,
    )


async def update_assessment_status(conn: asyncpg.Connection, assessment_id: UUID, status: str) -> None:
    await conn.execute(
        "UPDATE assessments SET status = $2, updated_at = now() WHERE id = $1",
        assessment_id,
        status,
    )


async def record_event(
    conn: asyncpg.Connection,
    *,
    assessment_id: UUID,
    seq: int,
    event_type: str,
    node_name: str,
    agent_name: str,
    input_summary: dict[str, Any],
    output_summary: dict[str, Any],
    reasoning: str | None = None,
    model_name: str | None = None,
    tool_name: str | None = None,
    latency_ms: int | None = None,
) -> int:
    """Insert one immutable ledger row. `seq` is caller-assigned (from the
    graph state's running counter) so ordering is deterministic even though
    Postgres also stamps a wall-clock created_at."""
    row = await conn.fetchrow(
        """
        INSERT INTO audit_events
            (assessment_id, seq, event_type, node_name, agent_name, model_name,
             tool_name, input_summary, output_summary, reasoning, latency_ms,
             prev_hash, record_hash, created_at)
        VALUES
            ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb, $10, $11,
             repeat('0', 64), repeat('0', 64), $12)
        RETURNING id
        """,
        # prev_hash / record_hash placeholders above are overwritten by the
        # BEFORE INSERT trigger `audit_events_chain_hash` in schema.sql --
        # they're passed here only to satisfy NOT NULL before the trigger runs.
        assessment_id,
        seq,
        event_type,
        node_name,
        agent_name,
        model_name,
        tool_name,
        json.dumps(input_summary, default=str),
        json.dumps(output_summary, default=str),
        reasoning,
        latency_ms,
        datetime.now(timezone.utc),
    )
    return row["id"]


async def record_conflict(
    conn: asyncpg.Connection,
    *,
    assessment_id: UUID,
    audit_event_id: int,
    round_number: int,
    raised_by_agent: str,
    disputed_agent: str,
    disputed_fields: dict[str, Any],
    reason: str,
) -> UUID:
    row = await conn.fetchrow(
        """
        INSERT INTO conflicts
            (assessment_id, audit_event_id, round, raised_by_agent, disputed_agent, disputed_fields, reason)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
        RETURNING id
        """,
        assessment_id,
        audit_event_id,
        round_number,
        raised_by_agent,
        disputed_agent,
        json.dumps(disputed_fields),
        reason,
    )
    return row["id"]


async def resolve_conflict(conn: asyncpg.Connection, conflict_id: UUID, resolution: str) -> None:
    await conn.execute("UPDATE conflicts SET resolution = $2 WHERE id = $1", conflict_id, resolution)


async def record_decision(
    conn: asyncpg.Connection,
    *,
    assessment_id: UUID,
    version: int,
    esg_score: float | None,
    esg_rating: str | None,
    fraud_risk_score: float | None,
    fraud_risk_level: str | None,
    distress_risk_score: float | None,
    distress_risk_level: str | None,
    verdict: str,
    rationale: str,
    computed_by: dict[str, Any],
    reviewed_by_human: bool,
    human_reviewer: str | None,
    conflict_rounds: int,
) -> UUID:
    row = await conn.fetchrow(
        """
        INSERT INTO decisions
            (assessment_id, version, esg_score, esg_rating, fraud_risk_score, fraud_risk_level,
             distress_risk_score, distress_risk_level,
             verdict, rationale, computed_by, reviewed_by_human, human_reviewer, conflict_rounds)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12, $13, $14)
        RETURNING id
        """,
        assessment_id,
        version,
        esg_score,
        esg_rating,
        fraud_risk_score,
        fraud_risk_level,
        distress_risk_score,
        distress_risk_level,
        verdict,
        rationale,
        json.dumps(computed_by, default=str),
        reviewed_by_human,
        human_reviewer,
        conflict_rounds,
    )
    return row["id"]


async def fetch_trace(conn: asyncpg.Connection, assessment_id: UUID) -> list[dict[str, Any]]:
    """Full ordered reasoning trace for one assessment -- what the API's
    GET /assessments/{id}/trace endpoint and any auditor tooling reads."""
    rows = await conn.fetch(
        "SELECT * FROM assessment_trace WHERE assessment_id = $1 ORDER BY seq",
        assessment_id,
    )
    return [dict(r) for r in rows]
