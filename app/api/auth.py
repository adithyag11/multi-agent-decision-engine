"""
API-key authentication and scope-based authorization.

This is deliberately minimal -- a static, env-configured key registry, not
a full IAM integration -- but it closes a real gap: before this existed,
ANY caller who could reach the service could submit assessments or, more
seriously, resolve a human escalation by POSTing to /resume with an
arbitrary `reviewer` name. Now every request is tied to a specific,
pre-registered identity, and that identity -- not a client-supplied string
in the request body -- is what gets written into the audit ledger's
`requested_by` and `human_reviewer` fields. A caller can no longer claim to
be someone else in a record meant to be legally defensible.

Two scopes:
  'caller'    -- can submit assessments and read status/trace. This is the
                 automated system (e.g. a loan-origination platform)
                 integrating with the engine.
  'reviewer'  -- can additionally resolve a human escalation. Deliberately
                 separate from 'caller': the system that originates a
                 request should not, by default, also be trusted to
                 override the Risk/Critic agent's judgment on it.
"""
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import ApiKeyIdentity, settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_identity(api_key: str | None = Security(_api_key_header)) -> ApiKeyIdentity:
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    identity = settings.api_key_registry.get(api_key)
    if identity is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return identity


def require_scope(scope: str):
    """Returns a FastAPI dependency that authenticates the caller AND
    checks it holds `scope`. Two-step (auth, then authorize) so a request
    with a valid-but-underprivileged key gets a 403 (you are who you say,
    but you can't do this), not a 401 (I don't know who you are) --
    different failure modes a real client needs to be able to tell apart."""

    async def _check(identity: ApiKeyIdentity = Security(get_identity)) -> ApiKeyIdentity:
        if scope not in identity.scopes:
            raise HTTPException(status_code=403, detail=f"API key lacks required scope '{scope}'")
        return identity

    return _check
