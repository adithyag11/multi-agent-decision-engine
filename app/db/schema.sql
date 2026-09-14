-- =============================================================================
-- Audit Ledger Schema -- Multi-Agent Business Decision Engine
-- Target: PostgreSQL 15+
--
-- Design goals (in priority order, because they occasionally trade off):
--   1. Regulatory defensibility: a third-party auditor (or Wells Fargo's
--      model-risk-management function, or a PwC/EY engagement team) must be
--      able to reconstruct *exactly* which agent made which call, with what
--      inputs, and how any disagreement between agents was resolved --
--      without trusting the application layer's honesty.
--   2. Tamper evidence: audit rows are insert-only at the database-grant
--      level (not just "we don't call UPDATE in the code"), and each row is
--      hash-chained to its predecessor so a row deleted or edited out-of-band
--      breaks the chain and is detectable.
--   3. Queryability for compliance reporting (e.g. "list every assessment
--      this quarter where the Risk agent overruled the Financial Analyst").
--
-- We use CHECK constraints instead of native Postgres ENUM types for status
-- columns. ENUMs are marginally cheaper on disk, but altering one (adding a
-- new status) takes an ACCESS EXCLUSIVE lock and a migration; a CHECK
-- constraint is a normal, reviewable DDL diff. In a system whose vocabulary
-- (agent names, event types) will grow as new specialist agents are added,
-- that flexibility is worth more than the marginal storage cost.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- for digest() / gen_random_uuid()

-- -----------------------------------------------------------------------------
-- Roles: engine_app (the orchestration service; INSERT/SELECT on the ledger,
-- never UPDATE/DELETE) and engine_readonly (the Data Engineer agent's SQL
-- tool; SELECT-only on financial_statements/esg_disclosures, nothing else).
-- Run roles.sql in this directory AFTER this file, once the tables it grants
-- against actually exist. Kept in a separate file rather than inlined here
-- because it needs real passwords supplied at deploy time -- see roles.sql's
-- header for how -- and shouldn't be re-run by every `psql -f schema.sql`.
-- -----------------------------------------------------------------------------

-- =============================================================================
-- 1. assessments
--    One row per inbound decision request (the "case file"). Mutable --
--    this is the working-state row, not the audit trail itself.
-- =============================================================================
CREATE TABLE IF NOT EXISTS assessments (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    external_reference  TEXT NOT NULL,              -- caller's own case/loan ID
    subject_type        TEXT NOT NULL CHECK (subject_type IN ('counterparty', 'issuer', 'vendor')),
    subject_id          TEXT NOT NULL,               -- e.g. LEI or internal entity ID
    request_type        TEXT NOT NULL CHECK (request_type IN ('esg_score', 'fraud_risk', 'combined')),
    status              TEXT NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'in_progress', 'escalated', 'completed', 'failed')),
    requested_by        TEXT NOT NULL,               -- calling system / user identity
    request_payload     JSONB NOT NULL,               -- verbatim ingested request
    checkpoint_thread_id TEXT,                        -- LangGraph checkpointer thread_id, for replay
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_assessments_subject ON assessments (subject_type, subject_id);
CREATE INDEX IF NOT EXISTS idx_assessments_status ON assessments (status);

-- =============================================================================
-- 2. audit_events
--    The immutable, hash-chained ledger. One row per agent action, tool
--    call, routing decision, or conflict -- the append-only spine that lets
--    a reviewer replay the entire graph execution in order.
-- =============================================================================
CREATE TABLE IF NOT EXISTS audit_events (
    id              BIGSERIAL PRIMARY KEY,           -- global monotonic order
    assessment_id   UUID NOT NULL REFERENCES assessments (id),
    seq             INTEGER NOT NULL,                -- per-assessment order (1, 2, 3, ...)
    event_type      TEXT NOT NULL CHECK (event_type IN (
                        'routing_decision',  -- supervisor chose the next node
                        'agent_call',        -- an agent (LLM) reasoning step
                        'tool_call',         -- a deterministic tool invocation
                        'conflict',          -- critic rejected another agent's output
                        'human_decision',    -- a human resolved an escalation
                        'final_decision'     -- the terminal, published decision
                    )),
    node_name       TEXT NOT NULL,                   -- LangGraph node that emitted this event
    agent_name      TEXT NOT NULL,                   -- 'supervisor' | 'data_engineer' | 'financial_analyst' | 'risk_critic' | 'human'
    model_name      TEXT,                            -- e.g. 'claude-sonnet-5'; NULL for pure tool_call rows
    tool_name       TEXT,                            -- e.g. 'sql_query:get_financial_statements'
    input_summary   JSONB NOT NULL,                  -- redacted/structured inputs to this step
    output_summary  JSONB NOT NULL,                  -- structured outputs of this step
    reasoning       TEXT,                            -- human-readable rationale (LLM-drafted or rule-derived)
    latency_ms      INTEGER,
    prev_hash       CHAR(64) NOT NULL,                -- hash of the previous event in this assessment's chain
    record_hash     CHAR(64) NOT NULL,                -- sha256(prev_hash || this row's canonical content)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (assessment_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_audit_events_assessment ON audit_events (assessment_id, seq);
CREATE INDEX IF NOT EXISTS idx_audit_events_type ON audit_events (event_type);

-- --- Hash-chaining trigger -----------------------------------------------
-- Computes record_hash server-side so the application cannot forge a chain
-- by pre-computing a hash client-side and having it accepted verbatim.
-- Genesis row per assessment chains from a well-known zero hash.
CREATE OR REPLACE FUNCTION audit_events_chain_hash() RETURNS TRIGGER AS $$
DECLARE
    last_hash CHAR(64);
BEGIN
    SELECT record_hash INTO last_hash
    FROM audit_events
    WHERE assessment_id = NEW.assessment_id
    ORDER BY seq DESC
    LIMIT 1;

    NEW.prev_hash := COALESCE(last_hash, repeat('0', 64));

    NEW.record_hash := encode(
        digest(
            NEW.prev_hash ||
            NEW.assessment_id::text ||
            NEW.seq::text ||
            NEW.event_type ||
            NEW.node_name ||
            NEW.agent_name ||
            COALESCE(NEW.tool_name, '') ||
            NEW.input_summary::text ||
            NEW.output_summary::text ||
            COALESCE(NEW.reasoning, '') ||
            NEW.created_at::text,
            'sha256'
        ),
        'hex'
    );

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_audit_events_chain_hash ON audit_events;
CREATE TRIGGER trg_audit_events_chain_hash
    BEFORE INSERT ON audit_events
    FOR EACH ROW EXECUTE FUNCTION audit_events_chain_hash();

-- --- Immutability guard ----------------------------------------------------
-- Defense in depth: even if a role somehow retains UPDATE/DELETE grants
-- (misconfiguration, a future migration that forgets to re-revoke), this
-- trigger makes mutation impossible at the database layer, not just by
-- convention in application code.
CREATE OR REPLACE FUNCTION reject_ledger_mutation() RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'audit_events is append-only: % is not permitted', TG_OP;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_audit_events_no_update ON audit_events;
CREATE TRIGGER trg_audit_events_no_update
    BEFORE UPDATE OR DELETE ON audit_events
    FOR EACH ROW EXECUTE FUNCTION reject_ledger_mutation();

-- Grants for this table are in roles.sql (REVOKE UPDATE/DELETE, GRANT
-- INSERT/SELECT to engine_app) -- the trigger above is defense-in-depth on
-- top of that grant, not a replacement for it.

-- =============================================================================
-- 3. conflicts
--    Denormalized, queryable view of every agent-vs-agent disagreement, for
--    compliance reporting ("how often does Risk overrule Financial Analyst,
--    and on what grounds?"). Each row also has a matching event_type =
--    'conflict' row in audit_events -- this table is a reporting index over
--    that immutable fact, not a separate source of truth.
-- =============================================================================
CREATE TABLE IF NOT EXISTS conflicts (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id       UUID NOT NULL REFERENCES assessments (id),
    audit_event_id      BIGINT NOT NULL REFERENCES audit_events (id),
    round               INTEGER NOT NULL,            -- which retry attempt (1, 2, ...)
    raised_by_agent     TEXT NOT NULL,                -- 'risk_critic'
    disputed_agent      TEXT NOT NULL,                -- 'data_engineer' | 'financial_analyst'
    disputed_fields     JSONB NOT NULL,               -- which output fields are in question
    reason              TEXT NOT NULL,
    resolution          TEXT NOT NULL DEFAULT 'pending'
                            CHECK (resolution IN ('pending', 'retried', 'escalated_to_human', 'overridden')),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conflicts_assessment ON conflicts (assessment_id);

-- =============================================================================
-- 4. decisions
--    The terminal, published output of an assessment. Insert-only: if an
--    assessment is re-run (new data, appeal, correction), it produces a new
--    versioned row rather than mutating the original -- the prior decision
--    remains in the historical record, which matters for model-risk audits
--    of "what did we tell the business on date X."
-- =============================================================================
CREATE TABLE IF NOT EXISTS decisions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id       UUID NOT NULL REFERENCES assessments (id),
    version             INTEGER NOT NULL DEFAULT 1,
    esg_score           NUMERIC(5, 2),                -- 0-100 composite, NULL if request_type = 'fraud_risk'
    esg_rating          TEXT,                          -- 'AAA'...'CCC' style bucket
    fraud_risk_score    NUMERIC(5, 2),                -- 0-100, NULL if request_type = 'esg_score'
    fraud_risk_level    TEXT CHECK (fraud_risk_level IN ('low', 'medium', 'high', 'critical')),
    distress_risk_score  NUMERIC(6, 3),                -- Altman Z''-score, NULL if request_type = 'esg_score'
    distress_risk_level  TEXT CHECK (distress_risk_level IN ('safe', 'grey', 'distress')),
    verdict             TEXT NOT NULL CHECK (verdict IN ('approved', 'approved_with_flags', 'escalated', 'rejected')),
    rationale           TEXT NOT NULL,                -- narrative summary (LLM-drafted, cites the numeric trace)
    computed_by         JSONB NOT NULL,                -- {"formula": "...", "tool_call_event_ids": [...]} traceability map
    reviewed_by_human    BOOLEAN NOT NULL DEFAULT FALSE,
    human_reviewer       TEXT,
    conflict_rounds      INTEGER NOT NULL DEFAULT 0,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (assessment_id, version)
);

CREATE INDEX IF NOT EXISTS idx_decisions_assessment ON decisions (assessment_id);

DROP TRIGGER IF EXISTS trg_decisions_no_update ON decisions;
CREATE TRIGGER trg_decisions_no_update
    BEFORE UPDATE OR DELETE ON decisions
    FOR EACH ROW EXECUTE FUNCTION reject_ledger_mutation();

-- =============================================================================
-- 5. run_queue
--    The durable work queue that decouples the API from graph execution.
--    POST /assessments and POST /.../resume both just INSERT a row here and
--    return immediately; one or more `app/worker.py` processes claim rows
--    with `FOR UPDATE SKIP LOCKED` (safe under multiple concurrent workers)
--    and actually invoke the graph. If a worker crashes mid-run, the
--    LangGraph checkpoint already has the last-completed-node state, so
--    another worker reclaiming a stale 'claimed' row (see
--    reclaim_stale_run_queue_jobs below) can safely resume it -- durability
--    comes from the checkpoint; this table is what makes sure *something*
--    actually goes and resumes it after a crash, rather than the run
--    silently dying with whatever in-process task happened to be holding it.
-- =============================================================================
CREATE TABLE IF NOT EXISTS run_queue (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id   UUID NOT NULL REFERENCES assessments (id),
    kind            TEXT NOT NULL CHECK (kind IN ('start', 'resume')),
    payload         JSONB NOT NULL DEFAULT '{}'::jsonb,   -- initial graph state (kind='start') or resume payload (kind='resume')
    status          TEXT NOT NULL DEFAULT 'queued'
                        CHECK (status IN ('queued', 'claimed', 'done', 'failed')),
    claimed_by      TEXT,                                  -- worker instance id, for observability
    claimed_at      TIMESTAMPTZ,
    attempts        INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_run_queue_status ON run_queue (status, created_at);

-- Reclaims jobs stuck 'claimed' past a staleness threshold (a worker that
-- died mid-run). Call periodically from the worker's own loop.
CREATE OR REPLACE FUNCTION reclaim_stale_run_queue_jobs(stale_after INTERVAL DEFAULT '5 minutes')
RETURNS INTEGER AS $$
DECLARE
    reclaimed INTEGER;
BEGIN
    UPDATE run_queue
    SET status = 'queued', claimed_by = NULL, claimed_at = NULL, updated_at = now()
    WHERE status = 'claimed' AND claimed_at < now() - stale_after;
    GET DIAGNOSTICS reclaimed = ROW_COUNT;
    RETURN reclaimed;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- 6. Convenience view: full reasoning trace for one assessment, in order,
--    ready to hand to an auditor or render in a review UI.
-- =============================================================================
CREATE OR REPLACE VIEW assessment_trace AS
SELECT
    a.id AS assessment_id,
    a.external_reference,
    a.status AS assessment_status,
    e.seq,
    e.event_type,
    e.node_name,
    e.agent_name,
    e.tool_name,
    e.reasoning,
    e.input_summary,
    e.output_summary,
    e.record_hash,
    e.created_at
FROM assessments a
JOIN audit_events e ON e.assessment_id = a.id
ORDER BY a.id, e.seq;

-- =============================================================================
-- 7. Source-of-truth tables the Data Engineer agent's SQL tool reads from.
--    Minimal illustrative shape -- in a real deployment these are almost
--    certainly views over an existing data warehouse, not owned by this
--    service. Queried by the `engine_readonly` role (see roles.sql) --
--    never by `engine_app`, and never write access for either.
-- =============================================================================
CREATE TABLE IF NOT EXISTS financial_statements (
    subject_id          TEXT NOT NULL,
    fiscal_year         INTEGER NOT NULL,
    revenue             NUMERIC,
    cogs                NUMERIC,                     -- cost of goods sold, for gross-margin analysis
    net_income           NUMERIC,
    total_assets        NUMERIC,
    current_assets       NUMERIC,
    current_liabilities   NUMERIC,                    -- for Altman working-capital ratio
    ppe_net              NUMERIC,                     -- net property, plant & equipment
    total_liabilities   NUMERIC,
    retained_earnings    NUMERIC,                     -- for Altman X2
    ebit                 NUMERIC,                     -- earnings before interest & tax, for Altman X3
    receivables         NUMERIC,
    depreciation        NUMERIC,
    sga_expense         NUMERIC,
    operating_cash_flow  NUMERIC,
    source_document_id  TEXT,
    PRIMARY KEY (subject_id, fiscal_year)
);

CREATE TABLE IF NOT EXISTS esg_disclosures (
    subject_id                  TEXT NOT NULL,
    fiscal_year                 INTEGER NOT NULL,
    scope1_emissions_tco2e      NUMERIC,
    scope2_emissions_tco2e      NUMERIC,
    board_independence_pct      NUMERIC,
    workforce_injury_rate       NUMERIC,
    controversy_flag_count      INTEGER,
    source_document_id          TEXT,
    PRIMARY KEY (subject_id, fiscal_year)
);
