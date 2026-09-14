"""
Centralized, typed configuration. Every other module imports `settings`
from here instead of reading os.environ directly -- this is what lets a
model swap, retry-policy change, or DB migration stay a one-line config
edit instead of a multi-file find-and-replace.
"""
from functools import lru_cache

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiKeyIdentity(BaseModel):
    """One entry in the API key registry: who a key belongs to, and what
    it's allowed to do. `scopes` gates which endpoints a key can call (see
    app/api/auth.py) -- 'caller' can submit and read assessments, 'reviewer'
    can additionally resolve a human escalation. A key's `name` is what
    gets written into the audit ledger's requested_by / human_reviewer
    fields -- never a client-supplied string, so that field can't be
    spoofed by whoever holds a different, lower-privileged key."""
    name: str
    scopes: list[str] = ["caller"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    data_warehouse_url: str
    anthropic_api_key: str

    agent_model: str = "claude-sonnet-5"
    max_conflict_retries: int = 2

    # Hard ceiling on how many supervisor loop iterations a single
    # assessment can take, independent of the conflict-retry count. This
    # is the backstop against an unforeseen routing bug spinning the graph
    # forever and burning LLM spend -- defense in depth, not a substitute
    # for correct routing logic.
    max_supervisor_steps: int = 12

    # API key -> identity. Populated from a JSON object in the
    # API_KEY_REGISTRY env var; pydantic-settings json-decodes env values
    # for non-str field types automatically. In a real deployment this is
    # backed by an identity provider or a database table with rotation, not
    # a static env var -- this is the minimum viable version of "requests
    # are attributable to a specific caller," not a full IAM system.
    api_key_registry: dict[str, ApiKeyIdentity] = {}

    # How long a run_queue job can sit 'claimed' before a worker is
    # considered dead and another worker is allowed to reclaim it.
    run_queue_stale_after_seconds: int = 300


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
