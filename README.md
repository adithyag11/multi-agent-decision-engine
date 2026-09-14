# Multi-Agent Business Decision Engine

A headless, API-first orchestration layer that deploys specialized, autonomous
agents to produce **auditable** ESG scores and financial fraud-risk
assessments for enterprise counterparties. Built for the standard a global
consultancy or bank model-risk function actually enforces: every number
traces to a deterministic function, every agent action is logged before the
next agent runs, the log cannot be edited after the fact, every request is
attributable to an authenticated identity, and no single signal or database
credential can unilaterally clear or convict a counterparty.

**Try the scoring core live:** [Formula Bench](https://claude.ai/code/artifact/05eaa4f8-afcf-42a0-a92b-09583de2a50d)
is an interactive, browser-based port of the Beneish M-Score and ESG
composite functions from `tools/formulas.py` — no server, no API key. It
predates the Altman Z''-Score signal added below, so treat it as a demo of
the scoring-core *pattern* (deterministic math, no LLM arithmetic), not a
complete mirror of the current backend.

## 1. System Architecture

```mermaid
flowchart TD
    Client["Client System<br/>(e.g. loan underwriting platform)"]

    Client -->|"POST /v1/assessments<br/>X-API-Key: caller scope"| Gateway["FastAPI Gateway"]
    Gateway -->|"401 / 403 on bad or under-scoped key"| Client
    Gateway -->|"202 Accepted<br/>(fire-and-forget)"| Client
    Client -->|"GET /v1/assessments/{id}"| Gateway
    Client -->|"POST .../resume<br/>X-API-Key: reviewer scope"| Gateway

    Gateway -->|"INSERT run_queue row<br/>(kind='start' / 'resume')"| Queue[("run_queue<br/>durable work queue")]
    Worker["app/worker.py<br/>(N replicas, FOR UPDATE SKIP LOCKED)"] -->|claims + runs| Queue
    Worker --> Supervisor

    subgraph Orchestration["LangGraph Orchestration Engine (per-assessment run)"]
        Supervisor["Supervisor Agent<br/>intake + delegation<br/>(deterministic routing)"]
        DataEng["Data Engineer Agent<br/>parameterized retrieval"]
        FinAnalyst["Financial Analyst Agent<br/>deterministic scoring"]
        RiskCritic["Risk / Critic Agent<br/>policy gates + judgment"]
        HumanEsc["Human Escalation<br/>interrupt() / resume"]
        Finalize["Finalize Decision"]

        Supervisor --> DataEng
        DataEng --> FinAnalyst
        FinAnalyst --> RiskCritic
        RiskCritic -->|approve| Finalize
        RiskCritic -->|"reject (retries remain)"| DataEng
        RiskCritic -->|"reject (retries exhausted)<br/>or escalate"| HumanEsc
        HumanEsc --> Finalize
    end

    DataEng -->|"allowlisted parameterized SQL<br/>engine_readonly role"| DW[("Data Warehouse<br/>financial_statements<br/>esg_disclosures")]
    FinAnalyst -->|formula registry call<br/>no LLM arithmetic| Tools["Deterministic Tools<br/>Beneish M-Score /<br/>Altman Z''-Score /<br/>ESG Composite Rubric"]

    DataEng -.structured-output LLM call<br/>untrusted-data wrapped.-> LLM[("Anthropic Claude<br/>claude-sonnet-5")]
    FinAnalyst -.narrative drafting only.-> LLM
    RiskCritic -.judgment-zone cases only.-> LLM

    Supervisor -->|"routing_decision<br/>engine_app role"| Ledger
    DataEng -->|tool_call| Ledger
    FinAnalyst -->|tool_call| Ledger
    RiskCritic -->|"agent_call + conflict"| Ledger
    HumanEsc -->|human_decision| Ledger
    Finalize -->|final_decision + decisions row| Ledger[("Postgres Audit Ledger<br/>hash-chained, insert-only<br/>audit_events / conflicts / decisions")]

    Gateway -->|"GET .../trace"| Ledger
    Orchestration -.every node write persisted.-> Checkpointer[("LangGraph Checkpointer<br/>durable execution state<br/>(PostgresSaver)")]
```

**Why this shape.** The Supervisor's job is dispatch, not judgment — deciding
"a new request starts with the Data Engineer" is a flowchart fact, not
something that benefits from an LLM re-deriving it on every run. The one
real decision point in the whole graph is **after the Risk/Critic agent**:
approve, send the Data Engineer back for a redo, or escalate to a human.
Concentrating all branching there means the entire set of paths this system
can take is small enough to enumerate in a model-risk review. The Gateway
never talks to the graph directly — it only ever writes a row to
`run_queue`; a separately-scaled fleet of `app/worker.py` processes is what
actually drives LangGraph, so a crashed worker or a slow LLM provider can
never turn into a hung HTTP request.

## 2. Multi-Agent Orchestration (LangGraph)

| File | Responsibility |
|---|---|
| [app/agents/state.py](app/agents/state.py) | Shared graph state (`AssessmentState` TypedDict) |
| [app/agents/schemas.py](app/agents/schemas.py) | Structured-output schemas for every LLM call |
| [app/agents/nodes.py](app/agents/nodes.py) | The five node implementations + the routing function |
| [app/agents/graph.py](app/agents/graph.py) | `StateGraph` wiring, compiled with a Postgres checkpointer |

**Where LLM calls happen, and where they deliberately don't:**

- **Supervisor** — no LLM call. Routing is plain Python.
- **Data Engineer** — one structured-output call to pick a fiscal-year window
  (or, on a remediation loop, to translate the Critic's rejection into a
  revised fetch plan). The actual data access goes through
  `tools/data_warehouse.run_query` over a connection from
  `app.db.pool.acquire_dw()` — a genuinely separate Postgres role
  (`engine_readonly`, see §4) from the one everything else uses, so the LLM
  selecting a bad `query_id` or a compromised Data Engineer prompt cannot
  produce a write or a ledger read, full stop, regardless of what the code
  asks for.
- **Financial Analyst** — **zero** LLM calls for the numbers.
  `FinancialPeriod` and `EsgDisclosure` rows go straight into
  `tools/formulas.py` (`compute_beneish_m_score`, `compute_altman_zscore`,
  `compute_esg_composite`), pure deterministic Python. **Two independent**
  fraud/risk signals are computed, not one: Beneish asks "does this year's
  accounting look manipulated relative to last year's" (needs 2 periods);
  Altman asks "is this company financially distressed right now" (needs
  only 1, so it still runs even when Beneish can't). One LLM call afterward
  drafts a plain-English narrative *over* the already-computed results.
- **Risk/Critic** — deterministic policy gates run first: data-completeness,
  then a hard "Beneish in the critical band → mandatory escalation" rule,
  then a hard "Altman in the distress zone → mandatory escalation" rule —
  **two separate gates**, so a company can trip either without needing to
  trip the other, and neither model is ever the sole reason a case is or
  isn't flagged. The LLM is consulted only in the genuine gray zone, and its
  answer must fit the `RiskVerdict` schema — the graph's routing branches on
  `verdict.verdict`, a validated `Literal`, never on parsed prose.
- **Human Escalation** — a real LangGraph `interrupt()`. The graph pauses
  indefinitely (durably, via the Postgres checkpointer) until a
  `reviewer`-scoped API key calls `POST /v1/assessments/{id}/resume`, which
  enqueues a `run_queue` resume job that a worker picks up and resumes with
  `Command(resume=...)`. Code before `interrupt()` is side-effect-free by
  design (see the comment on `human_escalation_node`), because LangGraph
  re-executes the node from the top on resume.

**Conflict handling, concretely:** if the Risk/Critic agent rejects the
Financial Analyst's output, it returns a `RiskVerdict(verdict="reject", ...)`
with `disputed_fields`. `route_after_risk_critic` sends the graph back to the
Data Engineer with a `remediation_request` in state, incrementing
`conflict_round`. After `MAX_CONFLICT_RETRIES` rejections (default 2), the
same rejection instead routes to `human_escalation` — the system never loops
forever trying to satisfy the Critic on its own.

**Prompt-injection hardening:** every dynamic value interpolated into an LLM
prompt (Critic reasoning, computed results, data-quality notes) is wrapped
with `_wrap_untrusted()` in `nodes.py` and paired with an explicit "this is
data, not instructions" system-prompt clause. Today's schema only carries
structured numeric fields, so there's effectively no exploitable surface
yet — the point of applying it now is that the moment unstructured source
documents (10-Ks, PDFs) are ingested, every extracted string has a
pre-established, tested pattern to go through, rather than needing one
retrofitted under pressure later.

## 3. API Gateway (FastAPI)

| File | Responsibility |
|---|---|
| [app/api/auth.py](app/api/auth.py) | API-key authentication + scope-based authorization |
| [app/api/schemas.py](app/api/schemas.py) | Pydantic request/response contracts |
| [app/api/routes.py](app/api/routes.py) | The four endpoints below |
| [app/main.py](app/main.py) | App wiring, lifespan-managed DB pools + checkpointer |
| [app/worker.py](app/worker.py) | Standalone process(es) that actually drive the graph |

```
POST   /v1/assessments              [caller]   202 Accepted, enqueues a run_queue 'start' job
GET    /v1/assessments/{id}         [caller]   current status + decision (or pending human-review payload)
POST   /v1/assessments/{id}/resume  [reviewer] 202 Accepted, enqueues a run_queue 'resume' job
GET    /v1/assessments/{id}/trace   [caller]   full ordered, hash-chained reasoning trace
```

Every route requires an `X-API-Key` header, scoped to `caller` or `reviewer`
(see `app/api/auth.py`) — deliberately two separate scopes, since the system
that originates a request shouldn't, by default, also be trusted to
override the Risk/Critic agent's judgment on it. The authenticated
identity — never a client-supplied string in the request body — is what
gets written into the ledger's `requested_by` and `human_reviewer` fields,
so a caller can't claim to be someone else in a record meant to be legally
defensible.

Both POST routes are **thin producers**: they INSERT a `run_queue` row in
the same transaction as the ledger row and return immediately. Nothing in
the request path ever calls the LangGraph graph — that's `app/worker.py`'s
job, run as one or more separate, independently-scalable processes. If a
worker crashes mid-run, the LangGraph checkpoint already has the last
completed node's state; `reclaim_stale_run_queue_jobs()` (a Postgres
function) resets any job stuck `'claimed'` past a staleness window back to
`'queued'`, and the next worker to poll it resumes from that checkpoint —
independently verified in `tests/test_integration.py`'s crash-recovery
scenario, not just asserted.

## 4. Audit Ledger (PostgreSQL)

See [app/db/schema.sql](app/db/schema.sql) for the DDL, [app/db/roles.sql](app/db/roles.sql)
for role/grant setup, and [app/db/migrate.py](app/db/migrate.py) for the
one-time LangGraph-checkpoint-table migration — all three verified against
a real, live PostgreSQL 16 instance, not just parsed. Deploy in that order:

```bash
psql "$ADMIN_DATABASE_URL" -f app/db/schema.sql
psql "$ADMIN_DATABASE_URL" -v engine_app_pw=... -v engine_readonly_pw=... -f app/db/roles.sql
ADMIN_DATABASE_URL=... python -m app.db.migrate
```

Tables:

- **`assessments`** — the mutable case file (one row per request).
- **`audit_events`** — the immutable ledger. A `BEFORE INSERT` trigger
  computes a `sha256(prev_hash || this row's content)` hash chain server-side
  (so the app can't forge it), and a `BEFORE UPDATE OR DELETE` trigger raises
  on any mutation attempt — enforced at the database layer, not just by
  application convention. `REVOKE UPDATE, DELETE` from `engine_app` (see
  below) is the second, independent layer, proven with a direct grant test,
  not just asserted in a comment.
- **`conflicts`** — a queryable index over every Risk-vs-other-agent
  disagreement, for compliance reporting ("how often does Risk overrule the
  Financial Analyst, and why?").
- **`decisions`** — the terminal, versioned output, now including
  `distress_risk_score` / `distress_risk_level` alongside the fraud and ESG
  fields. Also insert-only: a re-run produces a new version rather than
  mutating history.
- **`run_queue`** — the durable work queue described in §3.

`assessment_trace` is a convenience view joining `assessments` +
`audit_events` in order — what `GET /v1/assessments/{id}/trace` reads.
(Views need their own explicit `GRANT SELECT`, independent of grants on the
tables behind them — see the "verification" section below for how that was
found.)

**Role separation, actually enforced, not just described:**

- **`engine_app`** — what the API and worker connect as. `SELECT`/`INSERT`
  on the ledger, never `UPDATE`/`DELETE`; ordinary DML on `run_queue`; zero
  grants on `financial_statements`/`esg_disclosures`.
- **`engine_readonly`** — what `tools/data_warehouse.py` connects as via
  `acquire_dw()`. `SELECT`-only on the two source-of-truth tables, nothing
  else — it cannot read the audit ledger, `decisions`, or `conflicts`.

Both were proven with a direct SQL script, connected as each role in turn:
`engine_readonly` attempting an `INSERT` into `financial_statements` or a
`SELECT` from `audit_events` is rejected with `permission denied`;
`engine_app` attempting an `UPDATE` on `audit_events` or a direct `SELECT`
on `financial_statements` is likewise rejected. `tests/test_integration.py`
now runs against these exact restricted roles by default, not a superuser —
if a grant were ever wrong, the whole suite fails with a permission error
rather than silently passing under elevated test credentials.

## Design decisions worth flagging explicitly

- **No general-purpose Python REPL / `exec()` tool.** Truly sandboxing
  arbitrary LLM-authored code needs microVM-class isolation this service has
  no reason to own, and even with perfect isolation an auditor can't
  pre-certify code that's regenerated on every run. Agents instead call
  named, versioned, unit-tested functions from a fixed registry
  (`FORMULA_REGISTRY` in `tools/formulas.py`). Same logic for SQL:
  `QUERY_REGISTRY` in `tools/data_warehouse.py` is a fixed, parameterized
  set of templates — SQL injection is closed off by construction, not by
  sanitization.
- **Two independent risk signals, not one standing in for the whole
  story.** Relying on a single 1999 academic heuristic (Beneish) for a
  decision with real financial consequences is a single point of failure a
  real risk function would reject. Altman's Z''-score is a second,
  differently-shaped model (point-in-time solvency vs. year-over-year
  manipulation) with its own independent escalation gate.
- **The ESG rubric is explicitly *not* claimed as an industry-standard
  methodology.** Unlike Beneish and Altman, there's no single published "the
  ESG formula" — commercial raters don't publish theirs. `EsgRubricConfig` in
  `tools/formulas.py` bundles every weight and threshold into one swappable,
  documented object specifically so a real ESG policy team's methodology (or
  licensed third-party scores) can replace it without touching
  `compute_esg_composite`'s logic.
- **Hash-chained, trigger-enforced immutability** on `audit_events` and
  `decisions`, backed by real Postgres grants (see §4), rather than trusting
  the application layer not to call `UPDATE`.
- **Two Postgres roles, not one**, enforced by real `GRANT`/`REVOKE`
  statements and proven by direct testing, not just documented as an
  aspiration.
- **A durable work queue, not an in-process background task**, so a crashed
  worker process can never silently orphan a run — proven with an actual
  crash-recovery test, not just claimed.
- **Every request is authenticated and scoped**, and the ledger's identity
  fields come from that authentication, never from a client-supplied string.
- **`temperature=0`** on every LLM call. An audit trail benefits from the
  same inputs producing the same reasoning text on a retry.
- **Deterministic policy gates run before the LLM** in the Risk/Critic node.
  A firm rule like "critical fraud signal → mandatory human escalation"
  should not be something an LLM can reason its way around.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

createdb decision_engine
psql decision_engine -f app/db/schema.sql
psql decision_engine -v engine_app_pw=... -v engine_readonly_pw=... -f app/db/roles.sql
ADMIN_DATABASE_URL=postgresql://<admin>@localhost:5432/decision_engine python -m app.db.migrate

cp .env.example .env   # fill in DATABASE_URL, DATA_WAREHOUSE_URL, ANTHROPIC_API_KEY, API_KEY_REGISTRY, etc.

# Two processes, independently scalable:
uvicorn app.main:app --reload
python -m app.worker
```

```bash
curl -X POST localhost:8000/v1/assessments \
  -H 'content-type: application/json' \
  -H 'X-API-Key: sk-caller-changeme' \
  -d '{
        "external_reference": "LOAN-2026-04831",
        "subject_type": "counterparty",
        "subject_id": "LEI-549300ABCXYZ1234567",
        "request_type": "combined",
        "fiscal_year": 2025
      }'
# -> {"assessment_id": "...", "status": "pending", "poll_url": "/v1/assessments/..."}

curl -H 'X-API-Key: sk-caller-changeme' localhost:8000/v1/assessments/<id>
curl -H 'X-API-Key: sk-caller-changeme' localhost:8000/v1/assessments/<id>/trace
curl -X POST -H 'content-type: application/json' -H 'X-API-Key: sk-reviewer-changeme' \
  -d '{"decision":"approve","notes":"cleared after manual review"}' \
  localhost:8000/v1/assessments/<id>/resume
```

## Testing

```bash
createdb decision_engine_test
psql decision_engine_test -f app/db/schema.sql
psql decision_engine_test -v engine_app_pw=app_test_pw -v engine_readonly_pw=ro_test_pw -f app/db/roles.sql
ADMIN_DATABASE_URL=postgresql://<admin>@localhost:5432/decision_engine_test python -m app.db.migrate
pip install -r requirements-dev.txt

DATABASE_URL=postgresql://engine_app:app_test_pw@localhost:5432/decision_engine_test \
DATA_WAREHOUSE_URL=postgresql://engine_readonly:ro_test_pw@localhost:5432/decision_engine_test \
TEST_ADMIN_DATABASE_URL=postgresql://<admin>@localhost:5432/decision_engine_test \
python tests/test_integration.py
```

[tests/test_integration.py](tests/test_integration.py) drives the **real**
FastAPI app (real API-key auth), the **real** durable `run_queue` + worker
loop, the **real** LangGraph graph, and a **real** Postgres ledger —
connected as the actual restricted `engine_app` / `engine_readonly` roles,
not a superuser. `TEST_ADMIN_DATABASE_URL` is a separate, more-privileged
connection used only to seed the source-of-truth fixture tables, which is
itself part of the point: neither `engine_app` nor `engine_readonly` is
privileged to write `financial_statements`/`esg_disclosures`, exactly as
intended — seeding that data is an administrative/ETL job in a real
deployment too, not something the orchestration service does. The only
thing mocked is the Anthropic call boundary (`app.agents.nodes._llm`), so
this runs with no API key and no network calls.

10 scenarios: auth (401 no key, 401 bad key, 403 under-scoped resume);
immediate approval; two Risk/Critic rejections then approval; rejections
exhausting retries and escalating to a human, resumed both ways
(approve/reject); an already-resolved assessment correctly refusing a
second resume (409); a critical fraud signal auto-escalating without ever
reaching the LLM judgment branch; an unfixably incomplete data case
terminating (not hanging) at a human escalation; ESG-only / fraud-only
request scopes; a financial-distress case escalating on the **Altman** gate
specifically (never touching the fraud gate, proving the two signals are
independent); and a crash-recovery scenario that inserts a job already
`'claimed'` and backdated, reclaims it, and confirms the live worker
actually completes it. A closing pass checks hash-chain integrity and
"no orphaned state" invariants across every assessment the run created.

## Verification performed while building this

- `app/db/schema.sql`, `roles.sql`, and the checkpoint-table grants in
  `migrate.py` were applied to and exercised against a real local
  PostgreSQL 16 instance — not just parsed. A direct SQL script exercised
  the hash-chain trigger and the immutability triggers on both
  `audit_events` and `decisions`; a second script connected as each
  restricted role in turn and confirmed every intended denial actually
  denies (`engine_readonly` writing or reading the ledger; `engine_app`
  updating the ledger or reading source-of-truth tables directly).
- `tests/test_integration.py` — 58 checks across 10 scenarios — was run 7
  times in a row against the real restricted roles: 406 individual
  assertions, zero failures, zero flakiness.
- `compute_beneish_m_score` and `compute_esg_composite` were checked by hand
  against clean/manipulation-signature and strong/weak-ESG cases, which
  caught a real bug early on: an early `_band_score` helper took a
  `higher_is_better` flag and reversed the threshold list for "lower is
  better" metrics, double-applying the direction and silently inverting
  every environmental/social score. Fixed by making the helper
  direction-agnostic.
- The Formula Bench interactive artifact was cross-checked against the
  Python source across 800 randomized cases / 12,000 field-level
  comparisons — exact match on every field, every case (predates the
  Altman addition; see the note at the top of this file).

**Bugs caught only by actually running the system, not by static checks or
import-level smoke tests** (six across the full build; the two below are
new in this round):

1. `app/agents/graph.py` did `from app.db.pool import checkpointer` at
   module load time — a stale snapshot of `None` taken before
   `init_pools()` ever ran, which would have made every real request fail
   permanently. Fixed by reading `pool.checkpointer` at call time.
2. `human_escalation_node` wrote its status-change *after* `interrupt()`,
   which never runs on the pausing call — so `assessments.status` never
   actually reached `'escalated'`, and a real caller polling the API would
   have hung forever. Fixed by moving the (idempotent) write to before
   `interrupt()`.
3. **`AsyncPostgresSaver.setup()` needs `CREATE` privilege**, which the
   deliberately-locked-down `engine_app` role correctly does not have —
   discovered only by actually running the app against the real restricted
   role instead of a superuser. Fixed by splitting checkpoint-table
   creation into a separate, one-time admin migration (`app/db/migrate.py`)
   that also grants `engine_app` ordinary DML on the resulting tables,
   rather than having the app attempt DDL at every boot.
4. **Postgres views need their own explicit `SELECT` grant**, independent
   of the querying role's grants on the tables behind them —
   `engine_app` had `SELECT` on both `assessments` and `audit_events` but
   still got `permission denied` on the `assessment_trace` view built from
   them, until `GRANT SELECT ON assessment_trace TO engine_app` was added.
5. The test harness's own `poll_until_terminal` treated `'escalated'` as a
   terminal status unconditionally — correct for the *first* poll on a
   fresh assessment, but after a resume the assessment starts back in
   `'escalated'` before the worker has processed the resume job, so the
   poll returned immediately with a stale status instead of waiting for the
   real outcome. A test-harness bug, not an app bug, but worth naming: it
   would have produced a false "pass" (or, as it happened, a `TypeError`
   crash) that had nothing to do with whether resume actually worked.

Not independently verified: no live LLM call was made against
`claude-sonnet-5` (the Anthropic call boundary is mocked in the test), no
load/concurrency testing has been done against the connection pools
(`min_size=2, max_size=10` for `engine_app`, smaller for `engine_readonly`
— sized for this reference scale, untested at real production concurrency),
and the API-key registry remains a static env var rather than a rotatable,
database-backed identity store.
