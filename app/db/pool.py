"""
Owns the database connection surfaces this service uses:

  1. `app_pool`       -- asyncpg pool for the app's own tables (assessments,
                          audit_events, conflicts, decisions, run_queue).
                          Connects as `engine_app` (see roles.sql), which
                          can INSERT/SELECT the ledger but not UPDATE/DELETE
                          it, and cannot read the source-of-truth tables at
                          all.
  2. `dw_pool`        -- a SEPARATE pool for the Data Engineer agent's SQL
                          tool (tools/data_warehouse.py), connecting as
                          `engine_readonly`. This is a real, distinct
                          connection using `settings.data_warehouse_url` --
                          not just the same pool reused under a different
                          name -- so the privilege boundary between
                          "orchestration service" and "read a financial
                          record" is enforced by Postgres itself, not by
                          which Python function happens to call it.
  3. `checkpointer`   -- LangGraph's PostgresSaver, used purely for durable
                          *execution* state (so an in-flight graph survives
                          a process restart and human-in-the-loop
                          interrupts can be resumed hours or days later).
                          This is separate from the audit_events ledger:
                          the checkpointer is mutable working memory for
                          the orchestrator, while audit_events is the
                          immutable, regulator-facing record.
"""
import json
from contextlib import asynccontextmanager

import asyncpg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import settings

app_pool: asyncpg.Pool | None = None
dw_pool: asyncpg.Pool | None = None
_checkpointer_cm = None
checkpointer: AsyncPostgresSaver | None = None


def _jsonb_encode(value: object) -> str:
    # db/ledger.py already pre-serializes some jsonb params (so it can pass
    # `default=str` for Decimal/UUID/datetime values nested in a dict); other
    # callers may pass a plain dict/list. Accept either without double-
    # encoding a string that's already valid JSON text.
    return value if isinstance(value, str) else json.dumps(value, default=str)


async def _init_connection(conn: asyncpg.Connection) -> None:
    # asyncpg returns json/jsonb columns as raw text by default; without
    # this, every jsonb column we read back (input_summary, output_summary,
    # computed_by, ...) would be a JSON string instead of a dict, and every
    # Pydantic response model built from a fetched row would fail to
    # validate. Register the codec once per connection instead.
    await conn.set_type_codec(
        "jsonb", encoder=_jsonb_encode, decoder=json.loads, schema="pg_catalog", format="text",
    )
    await conn.set_type_codec(
        "json", encoder=_jsonb_encode, decoder=json.loads, schema="pg_catalog", format="text",
    )


async def init_pools() -> None:
    global app_pool, dw_pool, _checkpointer_cm, checkpointer

    app_pool = await asyncpg.create_pool(
        settings.database_url, min_size=2, max_size=10, init=_init_connection,
    )
    dw_pool = await asyncpg.create_pool(
        settings.data_warehouse_url, min_size=1, max_size=5, init=_init_connection,
    )

    # AsyncPostgresSaver manages its own connection for ongoing checkpoint
    # reads/writes. Deliberately does NOT call `.setup()` here: that method
    # issues CREATE TABLE for LangGraph's own checkpoint tables, which the
    # `engine_app` role this connects as does not have CREATE privilege to
    # do (correctly -- a locked-down runtime role shouldn't be able to run
    # DDL at all). Run `python -m app.db.migrate` once, with admin
    # credentials, before the app's first boot -- see that module for what
    # it does and why it's a separate, one-time step rather than part of
    # every process startup.
    _checkpointer_cm = AsyncPostgresSaver.from_conn_string(settings.database_url)
    checkpointer = await _checkpointer_cm.__aenter__()


async def close_pools() -> None:
    global app_pool, dw_pool, _checkpointer_cm
    if app_pool is not None:
        await app_pool.close()
    if dw_pool is not None:
        await dw_pool.close()
    if _checkpointer_cm is not None:
        await _checkpointer_cm.__aexit__(None, None, None)


@asynccontextmanager
async def acquire():
    """Connection from the engine_app pool -- ledger and case-file tables."""
    assert app_pool is not None, "init_pools() must run before acquire()"
    async with app_pool.acquire() as conn:
        yield conn


@asynccontextmanager
async def acquire_dw():
    """Connection from the engine_readonly pool -- source-of-truth tables
    ONLY. Never acquire ledger connections from here; the role's grants
    would refuse the write anyway, but the point is that this function is
    the one place in the codebase that's allowed to touch financial data,
    so a reviewer can audit data access by grepping for callers of this
    function alone."""
    assert dw_pool is not None, "init_pools() must run before acquire_dw()"
    async with dw_pool.acquire() as conn:
        yield conn
