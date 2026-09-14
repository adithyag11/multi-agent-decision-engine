"""
One-time, admin-run migration step: creates LangGraph's own checkpoint
tables (checkpoints, checkpoint_blobs, checkpoint_writes,
checkpoint_migrations) and grants the runtime `engine_app` role ordinary
DML (SELECT/INSERT/UPDATE/DELETE) on them -- nothing more.

Why this is separate from app.db.pool.init_pools(), which every process
boot calls: `AsyncPostgresSaver.setup()` issues CREATE TABLE, and the whole
point of `engine_app` (see roles.sql) is that it is NOT a role that can run
DDL. Baking `.setup()` into ordinary startup would mean either running the
app as an elevated role forever (defeating the purpose of roles.sql) or
watching it fail at boot the way it did the first time this was tried
against real restricted credentials -- which is exactly how this file came
to exist. Run this once per environment, with admin credentials, as part
of the same deploy step that applies schema.sql and roles.sql:

    psql "$ADMIN_DATABASE_URL" -f app/db/schema.sql
    psql "$ADMIN_DATABASE_URL" -v engine_app_pw=... -v engine_readonly_pw=... -f app/db/roles.sql
    ADMIN_DATABASE_URL=... python -m app.db.migrate

Idempotent: `.setup()` is safe to re-run (LangGraph tracks its own applied
migrations in checkpoint_migrations), and the GRANT statements below are
safe to re-run regardless of whether the tables already had them.
"""
import asyncio
import os

import asyncpg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

CHECKPOINT_TABLES = ["checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations"]


async def run(admin_database_url: str) -> None:
    async with AsyncPostgresSaver.from_conn_string(admin_database_url) as saver:
        await saver.setup()
    print(f"checkpoint tables ready: {', '.join(CHECKPOINT_TABLES)}")

    conn = await asyncpg.connect(admin_database_url)
    try:
        for table in CHECKPOINT_TABLES:
            await conn.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO engine_app")
        print(f"granted engine_app DML on: {', '.join(CHECKPOINT_TABLES)}")
    finally:
        await conn.close()


def main() -> None:
    admin_database_url = os.environ.get("ADMIN_DATABASE_URL")
    if not admin_database_url:
        raise SystemExit("ADMIN_DATABASE_URL must be set to a role with CREATE privilege on the target database")
    asyncio.run(run(admin_database_url))


if __name__ == "__main__":
    main()
