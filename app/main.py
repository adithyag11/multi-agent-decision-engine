"""
FastAPI application entrypoint. `uvicorn app.main:app --reload` for local
development.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router as assessments_router
from app.db.pool import close_pools, init_pools


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pools()
    yield
    await close_pools()


app = FastAPI(
    title="Multi-Agent Business Decision Engine",
    description=(
        "Headless, API-first orchestration layer for automated ESG scoring "
        "and financial fraud-risk assessment, built on a Supervisor / Data "
        "Engineer / Financial Analyst / Risk-Critic multi-agent pipeline "
        "with a hash-chained, immutable audit ledger."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(assessments_router)


@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
