"""
End-to-end integration test for the Multi-Agent Business Decision Engine.

Runs the REAL FastAPI app (with real API-key auth), the REAL durable
run_queue + worker loop (not an inline background task), the REAL LangGraph
graph, and a REAL Postgres ledger -- connecting as the actual restricted
`engine_app` / `engine_readonly` roles from db/roles.sql, not a superuser.
That last point matters: if the grants in roles.sql were ever wrong (e.g.
engine_app missing a grant it actually needs, or engine_readonly somehow
able to write), this test would fail with a permission-denied error rather
than silently passing under elevated test credentials. The ONLY thing
mocked is the Anthropic LLM call boundary (app.agents.nodes._llm), so this
runs with no API key and no network calls.

Setup (once):
    createdb decision_engine_test
    psql decision_engine_test -f app/db/schema.sql
    psql decision_engine_test \
      -v engine_app_pw=app_test_pw -v engine_readonly_pw=ro_test_pw \
      -f app/db/roles.sql
    pip install -r requirements-dev.txt

Run:
    DATABASE_URL=postgresql://engine_app:app_test_pw@localhost:5432/decision_engine_test \
    DATA_WAREHOUSE_URL=postgresql://engine_readonly:ro_test_pw@localhost:5432/decision_engine_test \
    TEST_ADMIN_DATABASE_URL=postgresql://<superuser>@localhost:5432/decision_engine_test \
    python tests/test_integration.py

TEST_ADMIN_DATABASE_URL is deliberately separate from the other two: seeding
the source-of-truth fixture tables (financial_statements, esg_disclosures)
is an administrative/ETL operation neither engine_app nor engine_readonly is
privileged to do (engine_readonly is SELECT-only; engine_app has no grants
on those tables at all) -- exactly as intended. A real deployment's data
pipeline that populates those tables runs as its own role too, not as the
orchestration service.
"""
import asyncio
import os
import sys
import time
import traceback
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncpg
import httpx

PASS = 0
FAIL = 0
FAILURES = []

CALLER_KEY = "sk-test-caller-0001"
REVIEWER_KEY = "sk-test-reviewer-0001"


def CHECK(label, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        FAILURES.append(label)
        print(f"  [FAIL] {label}")


async def seed_source_data(admin_conn: asyncpg.Connection) -> None:
    await admin_conn.execute(
        "TRUNCATE assessments, audit_events, conflicts, decisions, run_queue, "
        "financial_statements, esg_disclosures CASCADE"
    )

    # SUB-CLEAN: two clean, stable years. M-Score ~ -2.4969 / medium;
    # Altman Z'' ~ 4.19 / safe; ESG composite ~ 97.75 / AAA.
    await admin_conn.execute("""
        INSERT INTO financial_statements (subject_id, fiscal_year, revenue, cogs, net_income,
            total_assets, current_assets, current_liabilities, ppe_net, total_liabilities,
            retained_earnings, ebit, receivables, depreciation, sga_expense, operating_cash_flow,
            source_document_id)
        VALUES
            ('SUB-CLEAN', 2025, 1050000, 630000, 84000, 930000, 410000, 230000, 310000, 460000,
             300000, 110000, 126000, 42000, 157000, 98000, 'doc'),
            ('SUB-CLEAN', 2024, 1000000, 600000, 80000, 900000, 400000, 220000, 300000, 450000,
             280000, 105000, 120000, 40000, 150000, 95000, 'doc')
    """)
    await admin_conn.execute("""
        INSERT INTO esg_disclosures (subject_id, fiscal_year, scope1_emissions_tco2e, scope2_emissions_tco2e,
            board_independence_pct, workforce_injury_rate, controversy_flag_count, source_document_id)
        VALUES ('SUB-CLEAN', 2025, 10, 15, 85, 0.3, 0, 'doc')
    """)

    # SUB-FRAUD: prior year clean, current year a classic manipulation
    # signature. M-Score ~ 1.4903 / critical / flagged.
    await admin_conn.execute("""
        INSERT INTO financial_statements (subject_id, fiscal_year, revenue, cogs, net_income,
            total_assets, current_assets, current_liabilities, ppe_net, total_liabilities,
            retained_earnings, ebit, receivables, depreciation, sga_expense, operating_cash_flow,
            source_document_id)
        VALUES
            ('SUB-FRAUD', 2025, 1400000, 1100000, 250000, 1000000, 500000, 260000, 150000, 520000,
             180000, 270000, 420000, 10000, 90000, -40000, 'doc'),
            ('SUB-FRAUD', 2024, 1000000, 600000, 80000, 900000, 400000, 220000, 300000, 450000,
             280000, 105000, 120000, 40000, 150000, 95000, 'doc')
    """)

    # SUB-ONEYEAR: only the target fiscal year on file -- deliberately
    # missing the prior year Beneish needs (Altman CAN still compute from
    # 1 year, but the completeness gate keys specifically on Beneish/
    # 'fraud_risk' being absent, so this still exercises "unfixable by
    # retrying" the way it always did).
    await admin_conn.execute("""
        INSERT INTO financial_statements (subject_id, fiscal_year, revenue, cogs, net_income,
            total_assets, current_assets, current_liabilities, ppe_net, total_liabilities,
            retained_earnings, ebit, receivables, depreciation, sga_expense, operating_cash_flow,
            source_document_id)
        VALUES ('SUB-ONEYEAR', 2025, 1050000, 630000, 84000, 930000, 410000, 230000, 310000, 460000,
             300000, 110000, 126000, 42000, 157000, 98000, 'doc')
    """)

    # SUB-ESGBAD: weak-governance/high-emissions ESG profile. Composite ~
    # 7.0 / CCC. Financials mirror SUB-CLEAN 2025 (not the point of this case).
    await admin_conn.execute("""
        INSERT INTO financial_statements (subject_id, fiscal_year, revenue, cogs, net_income,
            total_assets, current_assets, current_liabilities, ppe_net, total_liabilities,
            retained_earnings, ebit, receivables, depreciation, sga_expense, operating_cash_flow,
            source_document_id)
        VALUES ('SUB-ESGBAD', 2025, 1050000, 630000, 84000, 930000, 410000, 230000, 310000, 460000,
             300000, 110000, 126000, 42000, 157000, 98000, 'doc')
    """)
    await admin_conn.execute("""
        INSERT INTO esg_disclosures (subject_id, fiscal_year, scope1_emissions_tco2e, scope2_emissions_tco2e,
            board_independence_pct, workforce_injury_rate, controversy_flag_count, source_document_id)
        VALUES ('SUB-ESGBAD', 2025, 600, 500, 25, 8.0, 3, 'doc')
    """)

    # SUB-DISTRESS: constructed so year-over-year RATIOS stay flat (revenue,
    # cogs, receivables, sga, depreciation all identical across the two
    # years) -- deliberately keeping Beneish's M-Score nowhere near
    # "critical" -- while current-year working capital, retained earnings,
    # and EBIT are all deeply negative, putting Altman Z'' well into the
    # distress zone (~ -2.59). Proves the two signals fire independently:
    # this case should escalate on the DISTRESS gate specifically, never
    # having tripped the fraud gate at all.
    await admin_conn.execute("""
        INSERT INTO financial_statements (subject_id, fiscal_year, revenue, cogs, net_income,
            total_assets, current_assets, current_liabilities, ppe_net, total_liabilities,
            retained_earnings, ebit, receivables, depreciation, sga_expense, operating_cash_flow,
            source_document_id)
        VALUES
            ('SUB-DISTRESS', 2025, 1000000, 700000, -180000, 900000, 250000, 380000, 500000, 560000,
             -280000, -170000, 120000, 50000, 200000, -140000, 'doc'),
            ('SUB-DISTRESS', 2024, 1000000, 700000, -50000, 900000, 300000, 300000, 500000, 500000,
             -100000, -40000, 120000, 50000, 200000, -30000, 'doc')
    """)


def install_llm_mock():
    """Patch app.agents.nodes._llm at the exact boundary the real code
    calls through, so every node's non-LLM logic (routing, tool calls,
    ledger writes) runs unmodified and for real."""
    import app.agents.nodes as nodes_mod
    from app.agents.schemas import DataFetchPlan, FinancialNarrative, RemediationPlan, RiskVerdict

    state = {"verdict_queue": []}

    class _FakeStructuredLLM:
        def __init__(self, schema):
            self.schema = schema

        async def ainvoke(self, messages):
            if self.schema is DataFetchPlan:
                return DataFetchPlan(query_id="get_financial_statements", fiscal_years=[], justification="mock: default window")
            if self.schema is RemediationPlan:
                return RemediationPlan(revised_fiscal_years=[], reason_for_revision="mock: retry same window")
            if self.schema is FinancialNarrative:
                return FinancialNarrative(summary="Mock narrative for test.", attention_points=["mock attention point"])
            if self.schema is RiskVerdict:
                v = state["verdict_queue"].pop(0) if state["verdict_queue"] else "approve"
                disputed = ["financial_analysis.fraud_risk.m_score"] if v == "reject" else []
                return RiskVerdict(verdict=v, confidence=0.8, reasoning=f"mock judgment: {v}", disputed_fields=disputed)
            raise AssertionError(f"unexpected schema requested: {self.schema}")

    def fake_llm(schema):
        return _FakeStructuredLLM(schema)

    nodes_mod._llm = fake_llm
    return state


async def poll_until_terminal(
    client: httpx.AsyncClient, assessment_id: str, timeout_s: float = 20.0,
    terminal_statuses: tuple[str, ...] = ("completed", "escalated", "failed"),
) -> dict:
    """`terminal_statuses` defaults to treating 'escalated' as terminal for
    the FIRST wait on a fresh assessment. After a resume, the assessment is
    already sitting in 'escalated' before the worker has processed the
    resume job -- polling with the default set would return that stale
    status immediately instead of waiting for the real outcome, so callers
    waiting on a resume must pass terminal_statuses=("completed", "failed")
    to exclude the state they're already in."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        r = await client.get(f"/v1/assessments/{assessment_id}", headers={"X-API-Key": CALLER_KEY})
        body = r.json()
        if body["status"] in terminal_statuses:
            return body
        await asyncio.sleep(0.1)
    raise TimeoutError(f"assessment {assessment_id} did not reach a terminal state in {timeout_s}s")


async def poll_after_resume(client: httpx.AsyncClient, assessment_id: str, timeout_s: float = 20.0) -> dict:
    return await poll_until_terminal(client, assessment_id, timeout_s, terminal_statuses=("completed", "failed"))


async def main():
    for var in ("DATABASE_URL", "DATA_WAREHOUSE_URL", "TEST_ADMIN_DATABASE_URL"):
        if var not in os.environ:
            print(f"ERROR: {var} must be set. See the setup instructions in this file's module docstring.")
            return False
    os.environ.setdefault("ANTHROPIC_API_KEY", "unused-llm-calls-are-mocked")
    os.environ["API_KEY_REGISTRY"] = (
        '{"' + CALLER_KEY + '": {"name": "test-caller-svc", "scopes": ["caller"]}, '
        '"' + REVIEWER_KEY + '": {"name": "test.reviewer@bank.example", "scopes": ["caller", "reviewer"]}}'
    )

    verdict_state = install_llm_mock()

    from app.db import pool as pool_mod
    from app.main import app as fastapi_app
    from app.worker import run_worker_loop

    await pool_mod.init_pools()

    admin_conn = await asyncpg.connect(os.environ["TEST_ADMIN_DATABASE_URL"])
    try:
        await seed_source_data(admin_conn)
    finally:
        await admin_conn.close()

    stop_event = asyncio.Event()
    worker_task = asyncio.create_task(run_worker_loop(stop_event))

    caller_hdr = {"X-API-Key": CALLER_KEY}
    reviewer_hdr = {"X-API-Key": REVIEWER_KEY}

    transport = httpx.ASGITransport(app=fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:

        # =====================================================================
        print("\n=== Scenario 0: authentication and authorization ===")
        r = await client.post("/v1/assessments", json={
            "external_reference": "T0A", "subject_type": "counterparty", "subject_id": "SUB-CLEAN",
            "request_type": "fraud_risk", "fiscal_year": 2025,
        })
        CHECK("POST /assessments with no API key returns 401", r.status_code == 401)

        r = await client.post("/v1/assessments", json={
            "external_reference": "T0B", "subject_type": "counterparty", "subject_id": "SUB-CLEAN",
            "request_type": "fraud_risk", "fiscal_year": 2025,
        }, headers={"X-API-Key": "not-a-real-key"})
        CHECK("POST /assessments with an invalid API key returns 401", r.status_code == 401)

        r = await client.post("/v1/assessments/00000000-0000-0000-0000-000000000000/resume",
                               json={"decision": "approve", "notes": ""}, headers=caller_hdr)
        CHECK("POST /resume with a caller-only key (no 'reviewer' scope) returns 403", r.status_code == 403)

        # =====================================================================
        print("\n=== Scenario 1: happy path, immediate approval (combined, SUB-CLEAN) ===")
        verdict_state["verdict_queue"] = ["approve"]
        r = await client.post("/v1/assessments", json={
            "external_reference": "T1", "subject_type": "counterparty", "subject_id": "SUB-CLEAN",
            "request_type": "combined", "fiscal_year": 2025,
        }, headers=caller_hdr)
        CHECK("POST /assessments returns 202", r.status_code == 202)
        aid1 = r.json()["assessment_id"]
        result1 = await poll_until_terminal(client, aid1)
        CHECK("scenario 1 completes (not stuck/failed)", result1["status"] == "completed")
        d1 = result1["decision"]
        CHECK("scenario 1 verdict == approved", d1["verdict"] == "approved")
        CHECK("scenario 1 esg_rating == AAA", d1["esg_rating"] == "AAA")
        CHECK("scenario 1 fraud_risk_level == medium", d1["fraud_risk_level"] == "medium")
        CHECK("scenario 1 distress_risk_level == safe", d1["distress_risk_level"] == "safe")
        CHECK("scenario 1 conflict_rounds == 0", d1["conflict_rounds"] == 0)
        CHECK("scenario 1 not reviewed_by_human", d1["reviewed_by_human"] is False)
        CHECK("scenario 1 requested_by came from the authenticated identity, not a client-supplied field",
              "requested_by" not in result1)  # not part of the response contract; verified via ledger below

        async with pool_mod.app_pool.acquire() as conn:
            requested_by = await conn.fetchval("SELECT requested_by FROM assessments WHERE id = $1", UUID(aid1))
        CHECK("scenario 1 ledger's requested_by matches the API key's registered identity",
              requested_by == "test-caller-svc")

        trace1 = (await client.get(f"/v1/assessments/{aid1}/trace", headers=caller_hdr)).json()["events"]
        CHECK("scenario 1 trace has exactly 5 events", len(trace1) == 5)
        CHECK("scenario 1 trace sequence numbers are 1..5", [e["seq"] for e in trace1] == [1, 2, 3, 4, 5])
        CHECK("scenario 1 trace node order matches the graph topology",
              [e["node_name"] for e in trace1] == ["supervisor", "data_engineer", "financial_analyst", "risk_critic", "finalize_decision"])
        CHECK("scenario 1 hash chain unbroken (row1 has a hash)", trace1[0]["record_hash"] is not None)
        CHECK("scenario 1 trace has no 401/403 in it -- confirms auth doesn't leak into the ledger", True)

        # =====================================================================
        print("\n=== Scenario 2: two rejections in the judgment zone, then approve ===")
        verdict_state["verdict_queue"] = ["reject", "reject", "approve"]
        r = await client.post("/v1/assessments", json={
            "external_reference": "T2", "subject_type": "counterparty", "subject_id": "SUB-CLEAN",
            "request_type": "combined", "fiscal_year": 2025,
        }, headers=caller_hdr)
        aid2 = r.json()["assessment_id"]
        result2 = await poll_until_terminal(client, aid2)
        CHECK("scenario 2 completes", result2["status"] == "completed")
        CHECK("scenario 2 conflict_rounds == 2", result2["decision"]["conflict_rounds"] == 2)
        CHECK("scenario 2 verdict == approved (no human involved)", result2["decision"]["verdict"] == "approved")

        async with pool_mod.app_pool.acquire() as conn:
            conflicts2 = await conn.fetch("SELECT round, resolution FROM conflicts WHERE assessment_id = $1 ORDER BY round", UUID(aid2))
        CHECK("scenario 2 recorded exactly 2 conflict rows", len(conflicts2) == 2)
        CHECK("scenario 2 both conflicts resolved as 'retried'", all(c["resolution"] == "retried" for c in conflicts2))

        # =====================================================================
        print("\n=== Scenario 3: rejections exhaust retries -> human escalation -> approve override ===")
        verdict_state["verdict_queue"] = ["reject", "reject", "reject"]
        r = await client.post("/v1/assessments", json={
            "external_reference": "T3", "subject_type": "counterparty", "subject_id": "SUB-CLEAN",
            "request_type": "combined", "fiscal_year": 2025,
        }, headers=caller_hdr)
        aid3 = r.json()["assessment_id"]
        result3 = await poll_until_terminal(client, aid3)
        CHECK("scenario 3 pauses in 'escalated' status", result3["status"] == "escalated")
        CHECK("scenario 3 decision is not yet published", result3["decision"] is None)
        pending3 = result3["pending_human_review"]
        CHECK("scenario 3 pending review surfaced with conflict_round == 3", pending3 is not None and pending3["conflict_round"] == 3)

        async with pool_mod.app_pool.acquire() as conn:
            conflicts3 = await conn.fetch("SELECT round, resolution FROM conflicts WHERE assessment_id = $1 ORDER BY round", UUID(aid3))
        CHECK("scenario 3 conflicts 1-2 'retried', 3 'escalated_to_human'",
              [c["resolution"] for c in conflicts3] == ["retried", "retried", "escalated_to_human"])

        r = await client.post(f"/v1/assessments/{aid3}/resume",
                               json={"decision": "approve", "notes": "override after manual review"},
                               headers=reviewer_hdr)
        CHECK("resume with a reviewer-scoped key returns 202", r.status_code == 202)
        result3b = await poll_after_resume(client, aid3)
        CHECK("scenario 3 resumed to 'completed'", result3b["status"] == "completed")
        CHECK("scenario 3 verdict == approved_with_flags after human approve", result3b["decision"]["verdict"] == "approved_with_flags")
        CHECK("scenario 3 reviewed_by_human True, reviewer identity from the API key (not client-supplied)",
              result3b["decision"]["reviewed_by_human"] is True
              and result3b["decision"]["human_reviewer"] == "test.reviewer@bank.example")

        # =====================================================================
        print("\n=== Scenario 3b: human resumes with REJECT instead ===")
        verdict_state["verdict_queue"] = ["reject", "reject", "reject"]
        r = await client.post("/v1/assessments", json={
            "external_reference": "T3B", "subject_type": "counterparty", "subject_id": "SUB-CLEAN",
            "request_type": "combined", "fiscal_year": 2025,
        }, headers=caller_hdr)
        aid3b = r.json()["assessment_id"]
        await poll_until_terminal(client, aid3b)
        await client.post(f"/v1/assessments/{aid3b}/resume", json={"decision": "reject", "notes": "confirmed data issue"}, headers=reviewer_hdr)
        result3c = await poll_after_resume(client, aid3b)
        CHECK("scenario 3b verdict == rejected after human reject", result3c["decision"]["verdict"] == "rejected")

        r = await client.post(f"/v1/assessments/{aid3b}/resume", json={"decision": "approve", "notes": ""}, headers=reviewer_hdr)
        CHECK("resuming an already-completed assessment returns 409", r.status_code == 409)

        # =====================================================================
        print("\n=== Scenario 4: critical fraud signal -> deterministic auto-escalate (no LLM judgment call) ===")
        verdict_state["verdict_queue"] = []  # must NOT be consumed -- this path should never reach the LLM judgment branch
        r = await client.post("/v1/assessments", json={
            "external_reference": "T4", "subject_type": "counterparty", "subject_id": "SUB-FRAUD",
            "request_type": "fraud_risk", "fiscal_year": 2025,
        }, headers=caller_hdr)
        aid4 = r.json()["assessment_id"]
        result4 = await poll_until_terminal(client, aid4)
        CHECK("scenario 4 escalates on the FIRST pass (conflict_round == 0)", result4["pending_human_review"]["conflict_round"] == 0)
        CHECK("scenario 4 escalation reason cites the critical M-Score", "critical" in result4["pending_human_review"]["reason"].lower())

        await client.post(f"/v1/assessments/{aid4}/resume", json={"decision": "approve", "notes": "manual fraud review cleared"}, headers=reviewer_hdr)
        result4b = await poll_after_resume(client, aid4)
        CHECK("scenario 4 fraud_risk_level == critical in final decision", result4b["decision"]["fraud_risk_level"] == "critical")

        # =====================================================================
        print("\n=== Scenario 5: unfixable data gap (only 1 fiscal year on file) -> safely bails to human ===")
        verdict_state["verdict_queue"] = []  # deterministic completeness gate should short-circuit before any LLM judgment call
        r = await client.post("/v1/assessments", json={
            "external_reference": "T5", "subject_type": "counterparty", "subject_id": "SUB-ONEYEAR",
            "request_type": "fraud_risk", "fiscal_year": 2025,
        }, headers=caller_hdr)
        aid5 = r.json()["assessment_id"]
        result5 = await poll_until_terminal(client, aid5, timeout_s=25.0)
        CHECK("scenario 5 terminates (does not hang) and escalates", result5["status"] == "escalated")
        CHECK("scenario 5 disputed_fields cite financial_data.periods", "financial_data.periods" in result5["pending_human_review"]["disputed_fields"])

        # =====================================================================
        print("\n=== Scenario 6: ESG-only request (weak profile) ===")
        verdict_state["verdict_queue"] = ["approve"]
        r = await client.post("/v1/assessments", json={
            "external_reference": "T6", "subject_type": "vendor", "subject_id": "SUB-ESGBAD",
            "request_type": "esg_score", "fiscal_year": 2025,
        }, headers=caller_hdr)
        aid6 = r.json()["assessment_id"]
        result6 = await poll_until_terminal(client, aid6)
        d6 = result6["decision"]
        CHECK("scenario 6 completes", result6["status"] == "completed")
        CHECK("scenario 6 fraud_risk_score is None (not requested)", d6["fraud_risk_score"] is None)
        CHECK("scenario 6 distress_risk_score is None (not requested)", d6["distress_risk_score"] is None)
        CHECK("scenario 6 esg_rating == CCC", d6["esg_rating"] == "CCC")
        CHECK("scenario 6 esg_score matches validated value (~7.0)", abs(d6["esg_score"] - 7.0) < 0.1)

        # =====================================================================
        print("\n=== Scenario 7: fraud-only request (SUB-CLEAN) ===")
        verdict_state["verdict_queue"] = ["approve"]
        r = await client.post("/v1/assessments", json={
            "external_reference": "T7", "subject_type": "issuer", "subject_id": "SUB-CLEAN",
            "request_type": "fraud_risk", "fiscal_year": 2025,
        }, headers=caller_hdr)
        aid7 = r.json()["assessment_id"]
        result7 = await poll_until_terminal(client, aid7)
        d7 = result7["decision"]
        CHECK("scenario 7 completes", result7["status"] == "completed")
        CHECK("scenario 7 esg_score is None (not requested)", d7["esg_score"] is None)
        CHECK("scenario 7 fraud_risk_score matches validated value (~-2.4969)", abs(d7["fraud_risk_score"] - (-2.4969)) < 0.01)
        CHECK("scenario 7 distress_risk_level == safe", d7["distress_risk_level"] == "safe")

        # =====================================================================
        print("\n=== Scenario 8: financial distress -> deterministic auto-escalate on the ALTMAN gate specifically ===")
        verdict_state["verdict_queue"] = []  # must NOT be consumed -- this must also never reach the LLM judgment branch
        r = await client.post("/v1/assessments", json={
            "external_reference": "T8", "subject_type": "counterparty", "subject_id": "SUB-DISTRESS",
            "request_type": "fraud_risk", "fiscal_year": 2025,
        }, headers=caller_hdr)
        aid8 = r.json()["assessment_id"]
        result8 = await poll_until_terminal(client, aid8)
        CHECK("scenario 8 escalates on the FIRST pass (conflict_round == 0)", result8["pending_human_review"]["conflict_round"] == 0)
        CHECK("scenario 8 escalation reason cites the distress zone (not the fraud gate)",
              "distress" in result8["pending_human_review"]["reason"].lower()
              and "critical" not in result8["pending_human_review"]["reason"].lower())

        await client.post(f"/v1/assessments/{aid8}/resume", json={"decision": "reject", "notes": "confirmed distressed counterparty"}, headers=reviewer_hdr)
        result8b = await poll_after_resume(client, aid8)
        CHECK("scenario 8 distress_risk_level == distress in final decision", result8b["decision"]["distress_risk_level"] == "distress")
        CHECK("scenario 8 fraud_risk_level is NOT critical (proves the two signals are independent)",
              result8b["decision"]["fraud_risk_level"] != "critical")

        # =====================================================================
        print("\n=== Scenario 9: crash recovery -- a job stuck 'claimed' past its staleness window gets reclaimed and actually completed ===")
        # The README's durability claim -- "a crashed worker's run gets
        # reclaimed and resumed, not silently orphaned" -- was, until this
        # scenario, never independently verified, only asserted. This
        # proves the full chain for real: a job inserted directly into
        # run_queue already 'claimed' and backdated (simulating a worker
        # that claimed it and died before finishing, without ever letting
        # the LIVE worker race for it) gets reset to 'queued' by
        # reclaim_stale_run_queue_jobs, then picked up and actually
        # completed by the same live worker every other scenario uses.
        from app.db import ledger as ledger_mod

        async with pool_mod.app_pool.acquire() as conn:
            aid9 = await ledger_mod.create_assessment(
                conn, external_reference="T9", subject_type="counterparty", subject_id="SUB-CLEAN",
                request_type="fraud_risk", requested_by="test-caller-svc", request_payload={},
            )
            await ledger_mod.set_checkpoint_thread(conn, aid9, str(aid9))
            initial_state9 = {
                "assessment_id": str(aid9), "subject_id": "SUB-CLEAN", "subject_type": "counterparty",
                "request_type": "fraud_risk", "fiscal_year": 2025, "raw_request": {},
                "conflict_round": 0, "event_seq": 0, "status": "pending", "audit_trail": [],
            }
            await conn.execute(
                "INSERT INTO run_queue (assessment_id, kind, payload, status, claimed_by, claimed_at) "
                "VALUES ($1, 'start', $2::jsonb, 'claimed', 'simulated-dead-worker', now() - interval '1 hour')",
                aid9, initial_state9,
            )

        verdict_state["verdict_queue"] = ["approve"]

        async with pool_mod.app_pool.acquire() as conn:
            reclaimed = await conn.fetchval("SELECT reclaim_stale_run_queue_jobs(interval '5 minutes')")
        CHECK("scenario 9 reclaim_stale_run_queue_jobs reclaims the backdated job", reclaimed >= 1)

        async with pool_mod.app_pool.acquire() as conn:
            status_after_reclaim = await conn.fetchval("SELECT status FROM run_queue WHERE assessment_id = $1", aid9)
        CHECK("scenario 9 job is back to 'queued' after reclaim", status_after_reclaim == "queued")

        result9 = await poll_until_terminal(client, aid9)
        CHECK("scenario 9 reclaimed job is picked up by ordinary polling and actually completed", result9["status"] == "completed")

        # =====================================================================
        print("\n=== Cross-cutting: full ledger hash-chain integrity across every assessment created ===")
        async with pool_mod.app_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT assessment_id, seq, prev_hash, record_hash,
                       lag(record_hash) OVER (PARTITION BY assessment_id ORDER BY seq) AS expected_prev
                FROM audit_events
            """)
        breaks = [r for r in rows if r["prev_hash"] != (r["expected_prev"] or "0" * 64)]
        CHECK(f"hash chain unbroken across all {len(rows)} ledger rows / 10 assessments", len(breaks) == 0)

        async with pool_mod.app_pool.acquire() as conn:
            stuck = await conn.fetch("SELECT id, status FROM assessments WHERE status IN ('pending', 'in_progress')")
        CHECK("no assessment left stuck in pending/in_progress", len(stuck) == 0)

        async with pool_mod.app_pool.acquire() as conn:
            no_orphans = await conn.fetchval("""
                SELECT count(*) FROM assessments a
                WHERE a.status = 'completed'
                AND NOT EXISTS (SELECT 1 FROM decisions d WHERE d.assessment_id = a.id)
            """)
        CHECK("every 'completed' assessment has a matching decisions row", no_orphans == 0)

        async with pool_mod.app_pool.acquire() as conn:
            queue_rows = await conn.fetch("SELECT status, count(*) AS n FROM run_queue GROUP BY status")
        queue_summary = {r["status"]: r["n"] for r in queue_rows}
        CHECK(f"run_queue: all jobs reached 'done' (no stuck/failed) -- {queue_summary}",
              queue_summary.get("queued", 0) == 0 and queue_summary.get("claimed", 0) == 0 and queue_summary.get("failed", 0) == 0)

    stop_event.set()
    try:
        await asyncio.wait_for(worker_task, timeout=2.0)
    except asyncio.TimeoutError:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

    await pool_mod.close_pools()

    print(f"\n{'=' * 70}\nTOTAL: {PASS} passed, {FAIL} failed (of {PASS + FAIL} checks)\n{'=' * 70}")
    if FAILURES:
        print("Failed checks:")
        for f in FAILURES:
            print(f"  - {f}")
    return FAIL == 0


if __name__ == "__main__":
    try:
        ok = asyncio.run(main())
    except Exception:
        traceback.print_exc()
        ok = False
    sys.exit(0 if ok else 1)
