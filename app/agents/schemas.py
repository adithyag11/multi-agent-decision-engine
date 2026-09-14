"""
Structured-output schemas for the LLM calls inside each agent node.

Every LLM call in this system is bound with `.with_structured_output(...)`
against one of these Pydantic models -- the model is never allowed to
return free-form prose that the graph then has to parse or regex out of a
string. This matters for two reasons: (1) routing decisions branch on
validated fields (e.g. `RiskVerdict.verdict`, a constrained Literal), never
on substring-matching an LLM's sentence, and (2) every field written to the
audit ledger has a known shape, which is what makes the ledger queryable
for compliance reporting rather than just a pile of chat transcripts.
"""
from typing import Literal

from pydantic import BaseModel, Field


class DataFetchPlan(BaseModel):
    """Data Engineer's decision about what to pull for this assessment."""
    query_id: Literal["get_financial_statements", "get_esg_disclosures"]
    fiscal_years: list[int] = Field(description="Fiscal years to request, most recent first")
    justification: str


class RemediationPlan(BaseModel):
    """Data Engineer's interpretation of a Risk/Critic rejection into a
    concrete, re-runnable data request -- the LLM's job here is purely to
    map the critic's structured complaint onto new query parameters, not to
    invent numbers."""
    revised_fiscal_years: list[int]
    reason_for_revision: str


class FinancialNarrative(BaseModel):
    """Financial Analyst's plain-English gloss on the formula outputs. The
    numeric fields (m_score, composite_score, etc.) come only from
    tools/formulas.py -- this schema never carries a number the LLM
    computed itself."""
    summary: str = Field(description="2-4 sentence plain-English explanation of the computed results")
    attention_points: list[str] = Field(
        description="Specific computed ratios/sub-scores worth the Risk agent's scrutiny, with why"
    )


class RiskVerdict(BaseModel):
    """Risk/Critic agent's structured judgment. `verdict` is what the
    supervisor's conditional edge branches on -- everything else is
    supporting evidence for the audit trail."""
    verdict: Literal["approve", "reject", "escalate"]
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    disputed_fields: list[str] = Field(
        default_factory=list,
        description="Which upstream fields (e.g. 'financial_analysis.m_score') this verdict disputes, if any",
    )
