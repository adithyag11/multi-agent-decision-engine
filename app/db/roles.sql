-- =============================================================================
-- Role separation for the Multi-Agent Business Decision Engine.
-- Run once, AFTER schema.sql, against the target database:
--
--     psql "$DATABASE_URL" \
--       -v engine_app_pw="$(openssl rand -base64 24)" \
--       -v engine_readonly_pw="$(openssl rand -base64 24)" \
--       -f app/db/roles.sql
--
-- Passwords are passed as psql variables rather than hardcoded here so this
-- file is safe to keep in version control. In a real deployment those
-- values come from a secrets manager (AWS Secrets Manager, Vault, etc.),
-- not a shell one-liner -- `openssl rand` above is only for local/dev use.
--
-- Why two roles instead of one:
--   engine_app       -- what the orchestration service (FastAPI + worker)
--                       connects to Postgres as for everything EXCEPT the
--                       Data Engineer's data-warehouse reads. Can write new
--                       ledger rows but cannot UPDATE or DELETE them -- so
--                       even a full compromise of the application's DB
--                       credentials can't be used to tamper with history.
--   engine_readonly  -- what the Data Engineer agent's SQL tool
--                       (tools/data_warehouse.py) connects as. SELECT-only,
--                       and only on the two source-of-truth tables it's
--                       allowed to query. It cannot read the audit ledger,
--                       decisions, or conflicts, and it cannot write
--                       anything anywhere -- so a bug or a successful
--                       prompt-injection against the Data Engineer's query
--                       selection can, at worst, read financial data it was
--                       already going to read; it categorically cannot
--                       forge a ledger entry or alter a filed decision.
-- =============================================================================

-- psql substitutes `:'var'` client-side, BEFORE the statement is sent to
-- the server -- but it deliberately does NOT substitute inside a
-- dollar-quoted ($$...$$) string, to avoid exactly the kind of confusion
-- that would cause. So the password variables have to be interpolated in a
-- plain top-level statement, not inside a DO block's body; \gexec is the
-- idiomatic way to make that statement's result conditional (idempotent
-- against a role that already exists) while still substituting client-side.
SELECT format('CREATE ROLE engine_app LOGIN PASSWORD %L', :'engine_app_pw')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'engine_app')
\gexec

SELECT format('CREATE ROLE engine_readonly LOGIN PASSWORD %L', :'engine_readonly_pw')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'engine_readonly')
\gexec

-- GRANT ... ON DATABASE takes an identifier, not an expression, so
-- current_database() can't be spliced in directly the way it can inside a
-- plain SELECT -- route it through dynamic SQL in an ordinary (non
-- password-bearing) DO block instead.
DO $$
BEGIN
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO engine_app', current_database());
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO engine_readonly', current_database());
END
$$;

-- --- engine_app: ledger tables -----------------------------------------
GRANT USAGE ON SCHEMA public TO engine_app;

GRANT SELECT, INSERT ON assessments TO engine_app;
GRANT UPDATE (status, checkpoint_thread_id, updated_at) ON assessments TO engine_app;  -- the mutable case-file row; NOT the ledger

GRANT SELECT, INSERT ON audit_events TO engine_app;
REVOKE UPDATE, DELETE ON audit_events FROM engine_app;
GRANT USAGE, SELECT ON SEQUENCE audit_events_id_seq TO engine_app;

GRANT SELECT, INSERT ON conflicts TO engine_app;
GRANT UPDATE (resolution) ON conflicts TO engine_app;  -- resolving a conflict's status, not rewriting its facts

GRANT SELECT, INSERT ON decisions TO engine_app;
REVOKE UPDATE, DELETE ON decisions FROM engine_app;

GRANT SELECT, INSERT, UPDATE, DELETE ON run_queue TO engine_app;  -- the work queue is ordinary mutable state, not the ledger

-- Views need their OWN explicit SELECT grant -- Postgres does not infer it
-- from the querying role's grants on the underlying tables (found the hard
-- way: engine_app has SELECT on both assessments and audit_events, which
-- assessment_trace is built from, and still got permission-denied on the
-- view itself until this line was added).
GRANT SELECT ON assessment_trace TO engine_app;

-- LangGraph's own checkpoint tables (checkpoints, checkpoint_blobs,
-- checkpoint_writes, checkpoint_migrations) don't exist yet the first time
-- this script runs -- they're created by `python -m app.db.migrate`, which
-- also grants engine_app DML on them. Run that AFTER this script, not
-- before (the tables have to exist before you can GRANT on them), and see
-- that module's docstring for why creating them is a separate admin step
-- rather than something the app does at boot.

-- engine_app must NOT be able to read or write the source-of-truth tables
-- directly -- all financial/ESG data access goes through engine_readonly's
-- narrower grant, via the Data Engineer's tool boundary.
REVOKE ALL ON financial_statements, esg_disclosures FROM engine_app;

-- --- engine_readonly: source-of-truth tables only -----------------------
GRANT USAGE ON SCHEMA public TO engine_readonly;

GRANT SELECT ON financial_statements, esg_disclosures TO engine_readonly;

-- Everything else is implicitly denied (Postgres default-deny) -- this
-- final block just makes the intent explicit and defends against a future
-- migration accidentally granting something broader by default.
REVOKE ALL ON assessments, audit_events, conflicts, decisions, run_queue FROM engine_readonly;
