"""
Shared LangGraph state. Every node reads and returns a partial update to
this TypedDict; LangGraph merges partial updates into the running state
(and, via `checkpointer`, persists it after every node so an interrupted
run can resume from exactly where it left off).

`audit_trail` uses the `operator.add` reducer because it is the one field
multiple nodes append to independently -- LangGraph needs to know to
concatenate rather than overwrite when a partial update includes it. Every
other field is last-write-wins, which is correct here because this graph is
a linear/branching state machine, not a fan-out/fan-in graph -- no two
nodes ever write the same non-list field in the same step.

Note `audit_trail` is an in-memory convenience for assembling the API
response without re-querying Postgres; the authoritative, tamper-evident
record is always the `audit_events` table (see db/ledger.py). If they ever
disagree, the database wins.
"""
import operator
from typing import Annotated, Any, NotRequired, TypedDict


class AuditTrailEntry(TypedDict):
    seq: int
    event_type: str
    node_name: str
    agent_name: str
    reasoning: str
    output_summary: dict[str, Any]


class AssessmentState(TypedDict):
    # --- Immutable request context (set once, at graph entry) ---
    assessment_id: str
    subject_id: str
    subject_type: str
    request_type: str          # 'esg_score' | 'fraud_risk' | 'combined'
    fiscal_year: int
    raw_request: dict[str, Any]

    # --- Working data, populated as agents run ---
    financial_data: NotRequired[dict[str, Any]]     # two periods of financial_statements rows
    esg_data: NotRequired[dict[str, Any]]            # one esg_disclosures row
    data_engineer_notes: NotRequired[list[str]]

    financial_analysis: NotRequired[dict[str, Any]]  # formula outputs + narrative
    risk_assessment: NotRequired[dict[str, Any]]      # verdict + reasons + disputed_fields

    # --- Control flow ---
    conflict_round: int                               # 0 = no conflicts yet
    remediation_request: NotRequired[dict[str, Any] | None]  # what Risk asked Data Engineer to redo
    human_decision: NotRequired[dict[str, Any]]        # set only if human_escalation_node ran
    event_seq: int                                     # monotonic per-assessment ledger sequence
    status: str                                         # mirrors assessments.status

    # --- Terminal output ---
    final_decision: NotRequired[dict[str, Any]]

    audit_trail: Annotated[list[AuditTrailEntry], operator.add]
