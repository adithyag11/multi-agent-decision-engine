"""
Wires the nodes in app/agents/nodes.py into a LangGraph StateGraph.

We build this by hand with StateGraph rather than the `langgraph-supervisor`
prebuilt package. That package is built for peer-to-peer agent handoffs
(via Command(goto=..., graph=Command.PARENT)) in swarm-style systems where
you want agents to hand off directly to one another. This system is the
opposite shape: a single authority (the Supervisor) with one real decision
point (post-Risk-Critic routing) and a hard compliance requirement that
every transition be logged before it happens. Hand-rolling the graph keeps
that routing table -- and the audit write at every edge -- fully explicit
and reviewable, which is what "we can show you every path this graph can
take" requires in a model-risk review.
"""
from langgraph.graph import END, START, StateGraph

from app.agents.nodes import (
    data_engineer_node,
    finalize_decision_node,
    financial_analyst_node,
    human_escalation_node,
    risk_critic_node,
    route_after_risk_critic,
    supervisor_node,
)
from app.agents.state import AssessmentState
from app.db import pool as db_pool


def build_graph() -> StateGraph:
    graph = StateGraph(AssessmentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("data_engineer", data_engineer_node)
    graph.add_node("financial_analyst", financial_analyst_node)
    graph.add_node("risk_critic", risk_critic_node)
    graph.add_node("human_escalation", human_escalation_node)
    graph.add_node("finalize_decision", finalize_decision_node)

    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "data_engineer")
    graph.add_edge("data_engineer", "financial_analyst")
    graph.add_edge("financial_analyst", "risk_critic")

    # The one branch point in the graph: Risk/Critic's verdict decides
    # whether we publish, loop back for a data redo, or escalate to a human.
    graph.add_conditional_edges(
        "risk_critic",
        route_after_risk_critic,
        {
            "finalize_decision": "finalize_decision",
            "data_engineer": "data_engineer",
            "human_escalation": "human_escalation",
        },
    )

    graph.add_edge("human_escalation", "finalize_decision")
    graph.add_edge("finalize_decision", END)

    return graph


_compiled_graph = None


def get_compiled_graph():
    """Lazily compiled, memoized. Must be called only after app.db.pool
    .init_pools() has run (FastAPI lifespan), since it binds the
    PostgresSaver checkpointer created there -- that's what makes an
    in-flight assessment (including one paused on a human-escalation
    interrupt) durable across process restarts.

    Reads `db_pool.checkpointer` (module-qualified) rather than importing
    the name directly: `from app.db.pool import checkpointer` would bind
    this module's own copy to whatever the value was at import time (None,
    since imports happen before init_pools() runs) and never see
    init_pools() reassign the pool module's global afterward -- a stale-
    import bug that made this assertion fire on every real request until
    caught by an end-to-end test."""
    global _compiled_graph
    if _compiled_graph is None:
        assert db_pool.checkpointer is not None, "init_pools() must run before get_compiled_graph()"
        _compiled_graph = build_graph().compile(checkpointer=db_pool.checkpointer)
    return _compiled_graph
