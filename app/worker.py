"""
Standalone worker process that consumes app.db.run_queue and drives the
LangGraph orchestration graph. Run as `python -m app.worker`, separately
from the FastAPI process (`uvicorn app.main:app`) and independently
scalable -- any number of worker replicas can run concurrently against the
same queue safely, because claiming a row uses `FOR UPDATE SKIP LOCKED`
(two workers racing for the same row: one gets it, the other's subquery
simply skips it and moves on -- no double-processing, no deadlock).

This is what makes assessment processing durable. The API layer (see
api/routes.py) only ever INSERTs a row into run_queue and returns
immediately -- it never calls the graph itself. If THIS process crashes
mid-run, the LangGraph checkpoint already has the last-completed-node's
state persisted (that's what app.db.pool.checkpointer is for); this
module's `reclaim_stale_run_queue_jobs` call resets any job that's been
'claimed' for too long back to 'queued', and the next worker to poll it
picks the row back up and resumes the graph from its last checkpoint --
so a crash loses at most the work since the last completed node, never
the whole run, and nothing about recovery depends on the crashed
process ever coming back.
"""
import asyncio
import logging
import socket
import time
import uuid
from typing import Any

from langgraph.types import Command

from app.agents.graph import get_compiled_graph
from app.config import settings
from app.db import ledger
from app.db.pool import acquire, close_pools, init_pools

logger = logging.getLogger("decision_engine.worker")

WORKER_ID = f"{socket.gethostname()}-{uuid.uuid4().hex[:8]}"

POLL_INTERVAL_S = 0.5
RECLAIM_INTERVAL_S = 30.0


def _graph_config(assessment_id: str) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": assessment_id},
        "recursion_limit": (settings.max_conflict_retries + 1) * 3 + 6,
    }


async def _claim_one_job(conn) -> dict[str, Any] | None:
    """Atomically claims the oldest queued job, or returns None if the
    queue is empty. The UPDATE...WHERE id = (SELECT ... FOR UPDATE SKIP
    LOCKED) shape is the standard Postgres pattern for a safe multi-
    consumer queue in a single round trip: the subquery locks and picks one
    row, skipping any row a concurrent worker already has locked, and the
    outer UPDATE claims exactly that row."""
    row = await conn.fetchrow(
        """
        UPDATE run_queue
        SET status = 'claimed', claimed_by = $1, claimed_at = now(),
            attempts = attempts + 1, updated_at = now()
        WHERE id = (
            SELECT id FROM run_queue
            WHERE status = 'queued'
            ORDER BY created_at
            FOR UPDATE SKIP LOCKED
            LIMIT 1
        )
        RETURNING id, assessment_id, kind, payload
        """,
        WORKER_ID,
    )
    return dict(row) if row else None


async def _process_job(job: dict[str, Any]) -> None:
    graph = get_compiled_graph()
    assessment_id = str(job["assessment_id"])
    config = _graph_config(assessment_id)

    if job["kind"] == "start":
        await graph.ainvoke(job["payload"], config=config)
    elif job["kind"] == "resume":
        await graph.ainvoke(Command(resume=job["payload"]), config=config)
    else:
        raise ValueError(f"unknown run_queue job kind: {job['kind']!r}")


async def run_once() -> bool:
    """Claims and processes at most one job. Returns True if a job was
    claimed (whether it then succeeded or failed), False if the queue was
    empty -- callers use this to decide whether to poll again immediately
    or back off."""
    async with acquire() as conn:
        job = await _claim_one_job(conn)
    if job is None:
        return False

    try:
        await _process_job(job)
        async with acquire() as conn:
            await conn.execute(
                "UPDATE run_queue SET status = 'done', updated_at = now() WHERE id = $1", job["id"],
            )
    except Exception as exc:
        # A failure here means an unhandled exception inside a node (e.g. a
        # provider outage) -- not a Risk/Critic rejection, which is a
        # normal `reject` verdict handled entirely inside the graph. Mark
        # both the job and the case failed rather than leaving either
        # silently stuck.
        logger.exception("run_queue job %s (assessment %s) failed", job["id"], job["assessment_id"])
        async with acquire() as conn:
            await conn.execute(
                "UPDATE run_queue SET status = 'failed', last_error = $2, updated_at = now() WHERE id = $1",
                job["id"], str(exc),
            )
            await ledger.update_assessment_status(conn, job["assessment_id"], "failed")
    return True


async def run_worker_loop(stop_event: asyncio.Event | None = None) -> None:
    """The main loop. `stop_event` lets an embedder (tests, or a single-
    process dev setup) shut it down cleanly; left as None, it runs forever,
    which is how the standalone `python -m app.worker` entrypoint uses it."""
    last_reclaim = 0.0
    while stop_event is None or not stop_event.is_set():
        now = time.monotonic()
        if now - last_reclaim > RECLAIM_INTERVAL_S:
            async with acquire() as conn:
                reclaimed = await conn.fetchval(
                    "SELECT reclaim_stale_run_queue_jobs(make_interval(secs => $1))",
                    settings.run_queue_stale_after_seconds,
                )
            if reclaimed:
                logger.warning("reclaimed %d stale run_queue job(s)", reclaimed)
            last_reclaim = now

        claimed = await run_once()
        if not claimed:
            await asyncio.sleep(POLL_INTERVAL_S)


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    await init_pools()
    logger.info("worker %s starting", WORKER_ID)
    try:
        await run_worker_loop()
    finally:
        await close_pools()


if __name__ == "__main__":
    asyncio.run(main())
