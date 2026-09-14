"""
The Data Engineer agent's only way to touch the database: a small,
allowlisted set of parameterized query templates, each with a Pydantic
schema for its arguments. The LLM never writes SQL -- it picks a `query_id`
by name and supplies structured parameters, which are validated before
they're bound (not interpolated) into the query. This closes off SQL
injection by construction rather than by sanitization, and it means every
query this service can ever run is enumerable and reviewable in one place
-- exactly what a data-governance sign-off needs.

The connection used here must come from `app.db.pool.acquire_dw()`, which
connects as the `engine_readonly` role (see db/roles.sql) -- a SEPARATE
Postgres role from the one the rest of the app uses, with SELECT-only
grants on financial_statements and esg_disclosures and no grants anywhere
else. That's a second, independent layer of protection enforced by
Postgres itself: even a bug in this file's allowlist, or a successful
prompt-injection against the Data Engineer agent's query selection, cannot
produce a write, a read of the audit ledger, or a cross-schema read --
those are permission-denied at the database layer regardless of what SQL
this code asks for.
"""
from typing import Any, Callable

import asyncpg
from pydantic import BaseModel, Field


class FinancialStatementsQueryParams(BaseModel):
    subject_id: str
    fiscal_years: list[int] = Field(min_length=1, max_length=5)


class EsgDisclosureQueryParams(BaseModel):
    subject_id: str
    fiscal_year: int


async def _get_financial_statements(conn: asyncpg.Connection, params: FinancialStatementsQueryParams) -> list[dict[str, Any]]:
    rows = await conn.fetch(
        """
        SELECT fiscal_year, revenue, cogs, net_income, total_assets, current_assets,
               current_liabilities, ppe_net, total_liabilities, retained_earnings, ebit,
               receivables, depreciation, sga_expense, operating_cash_flow, source_document_id
        FROM financial_statements
        WHERE subject_id = $1 AND fiscal_year = ANY($2::int[])
        ORDER BY fiscal_year DESC
        """,
        params.subject_id,
        params.fiscal_years,
    )
    return [dict(r) for r in rows]


async def _get_esg_disclosures(conn: asyncpg.Connection, params: EsgDisclosureQueryParams) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        """
        SELECT fiscal_year, scope1_emissions_tco2e, scope2_emissions_tco2e,
               board_independence_pct, workforce_injury_rate, controversy_flag_count,
               source_document_id
        FROM esg_disclosures
        WHERE subject_id = $1 AND fiscal_year = $2
        """,
        params.subject_id,
        params.fiscal_year,
    )
    return dict(row) if row else None


# query_id -> (async handler, param schema). Same pattern as
# tools/formulas.FORMULA_REGISTRY: one allowlist, enumerable and testable.
QUERY_REGISTRY: dict[str, tuple[Callable, type[BaseModel]]] = {
    "get_financial_statements": (_get_financial_statements, FinancialStatementsQueryParams),
    "get_esg_disclosures": (_get_esg_disclosures, EsgDisclosureQueryParams),
}


async def run_query(conn: asyncpg.Connection, query_id: str, raw_params: dict[str, Any]) -> Any:
    if query_id not in QUERY_REGISTRY:
        raise ValueError(f"Unknown query_id '{query_id}'. Allowed: {sorted(QUERY_REGISTRY)}")
    handler, schema = QUERY_REGISTRY[query_id]
    validated = schema.model_validate(raw_params)
    return await handler(conn, validated)
