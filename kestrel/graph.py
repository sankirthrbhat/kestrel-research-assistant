"""LangGraph StateGraph wiring for the Kestrel multi-agent workflow.

Graph topology
--------------

    [START]
       │
       ▼
   router_node ──(chit_chat/off_topic)──► finaliser_node ──► [END]
       │ (retrieval)
       ▼
   retriever_node
       │
       ▼
   synthesiser_node
       │
       ▼
   verifier_node
       │
       ▼
   finaliser_node
       │
       ▼
    [END]

Follow-up handling
------------------
The graph is stateless between calls.  Conversation history is passed in via
``KestrelState.history`` and the router rewrites the current question into a
standalone question before retrieval.  The caller (CLI or tests) is
responsible for appending the previous turn to ``history`` before the next
call.

Unsupported evidence handling
------------------------------
When the verifier finds ``insufficient_evidence`` the finaliser will say so
plainly rather than guessing.  A validated new query can request one extra retrieval. The finaliser
renders assessed claims directly, or labeled excerpts when verification is unavailable.

LangSmith tracing
-----------------
Tracing is enabled automatically by setting ``LANGCHAIN_TRACING_V2=true`` and
``LANGCHAIN_API_KEY`` in the environment (or in a ``.env`` file).  Each call
to ``run_query`` / ``run_query_traced`` uses a unique ``run_id`` so individual
turns can be located in the LangSmith UI.  Pass ``conversation_id`` to group
all turns of a conversation together in the trace metadata.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .agents import (
    finaliser_node,
    retriever_node,
    route_after_router,
    route_after_verifier,
    router_node,
    synthesiser_node,
    verifier_node,
)
from .state import KestrelState, make_initial_state
from .tracing import build_run_config, get_trace_url, is_tracing_enabled


def build_graph() -> StateGraph:
    """Construct and compile the Kestrel LangGraph StateGraph."""
    builder = StateGraph(KestrelState)

    # Add nodes
    builder.add_node("router", router_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("synthesiser", synthesiser_node)
    builder.add_node("verifier", verifier_node)
    builder.add_node("finaliser", finaliser_node)

    # Entry point
    builder.add_edge(START, "router")

    # Conditional edge: retrieval path vs. short-circuit (chit_chat / off_topic)
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retrieve": "retriever",
            "finalise": "finaliser",
        },
    )

    # Happy path: retrieval → synthesis → verification
    builder.add_edge("retriever", "synthesiser")
    builder.add_edge("synthesiser", "verifier")
    
    # Verification retry loop
    builder.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "retriever": "retriever",
            "finaliser": "finaliser",
        },
    )

    # Terminal
    builder.add_edge("finaliser", END)

    return builder.compile()


# Module-level compiled graph (lazy to avoid import-time overhead in tests).
_GRAPH: object | None = None


def get_graph() -> object:
    """Return the compiled graph, creating it on first call."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


# ---------------------------------------------------------------------------
# Internal implementation — shared by both public entry points
# ---------------------------------------------------------------------------


def _invoke(
    question: str,
    history: list[dict[str, str]] | None,
    *,
    graph: object | None,
    conversation_id: str | None,
) -> tuple[KestrelState, str | None]:
    """Run the graph and return ``(state, run_id)``."""
    _graph = graph or get_graph()
    initial = make_initial_state(question, history or [])

    # Build the LangGraph config.  Always include metadata even when tracing
    # is off; LangGraph ignores unknown metadata gracefully.
    from .llm import LLMConfig
    cfg = LLMConfig.from_env()
    run_cfg = build_run_config(
        conversation_id=conversation_id,
        model_name=cfg.model if cfg.configured else None,
        extra_metadata={"provider": cfg.provider},
    )

    run_cfg["recursion_limit"] = 12
    result: KestrelState = _graph.invoke(initial, config=run_cfg)  # type: ignore[attr-defined]
    return result, run_cfg["run_id"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_query(
    question: str,
    history: list[dict[str, str]] | None = None,
    *,
    graph: object | None = None,
    conversation_id: str | None = None,
) -> KestrelState:
    """Run a single question through the graph and return the final state.

    Parameters
    ----------
    question:
        The user's question for this turn.
    history:
        Previous turns: ``[{"role": "user"|"assistant", "content": "..."}]``.
        Pass the accumulated list to enable follow-up question handling.
    graph:
        Optional pre-built graph; defaults to the module-level singleton.
    conversation_id:
        Optional identifier grouping turns of the same conversation in
        LangSmith traces.  Has no effect when tracing is disabled.
    """
    state, _ = _invoke(question, history, graph=graph, conversation_id=conversation_id)
    return state


def run_query_traced(
    question: str,
    history: list[dict[str, str]] | None = None,
    *,
    graph: object | None = None,
    conversation_id: str | None = None,
) -> tuple[KestrelState, str | None]:
    """Run a question and return ``(state, trace_url)``.

    ``trace_url`` is the verified LangSmith URL for this run, obtained by
    querying the LangSmith API after the graph completes.  It is ``None`` when:

    * LangSmith is not configured
    * The trace has not yet been indexed (freshly submitted)
    * Any network or authentication failure occurs

    The function never fabricates a URL — every non-None value comes from the
    LangSmith REST API response.
    """
    state, run_id = _invoke(question, history, graph=graph, conversation_id=conversation_id)
    trace_url: str | None = None
    if is_tracing_enabled():
        trace_url = get_trace_url(run_id)
    return state, trace_url


def stream_question(
    question: str,
    history: list[dict[str, str]] | None = None,
    *,
    graph: object | None = None,
    conversation_id: str | None = None,
):
    """Yields (agent_name, status, state_update) events as the graph runs."""
    _graph = graph or get_graph()
    initial = make_initial_state(question, history or [])
    
    from .llm import LLMConfig
    cfg = LLMConfig.from_env()
    run_cfg = build_run_config(
        conversation_id=conversation_id,
        model_name=cfg.model if cfg.configured else None,
        extra_metadata={"provider": cfg.provider},
    )

    run_cfg["recursion_limit"] = 12
    for output in _graph.stream(initial, config=run_cfg): # type: ignore
        for node_name, state_update in output.items():
            yield node_name, "completed", state_update

__all__ = [
    "build_graph",
    "get_graph",
    "run_query",
    "run_query_traced",
    "stream_question",
]
