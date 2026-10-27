"""Shared typed state for the Kestrel multi-agent LangGraph workflow.

The ``KestrelState`` TypedDict is passed between every node in the graph.
All fields have sensible defaults so each agent only needs to write the
fields it owns -- it never has to initialise fields belonging to others.

Typing note: Python 3.14 fully supports ``TypedDict`` with ``total=False``
subclasses for optional fields, but the simplest portable pattern is to
use ``Optional`` annotations with ``None`` defaults via ``Annotated``.
We use plain ``TypedDict`` with all keys required (using defaults in
``make_initial_state``) so that LangGraph can inspect and merge cleanly.
"""

from __future__ import annotations

from typing import Any, TypedDict


# --------------------------------------------------------------------------
# Evidence item
# --------------------------------------------------------------------------


class EvidenceItem(TypedDict):
    """One retrieved chunk ready to pass to the Synthesiser."""

    chunk_id: str
    title: str
    text: str
    score: float
    doc_id: str
    metadata: dict[str, Any]


# --------------------------------------------------------------------------
# Claim (from Synthesiser and Verifier)
# --------------------------------------------------------------------------


class Claim(TypedDict):
    """A single factual assertion with its chunk-level citations."""

    claim: str
    chunk_ids: list[str]


class VerifiedClaim(TypedDict):
    """A claim after the Verifier has assessed it."""

    claim: str
    verdict: str  # supported | partially_supported | conflicting_evidence | insufficient_evidence
    chunk_ids: list[str]
    reason: str


# --------------------------------------------------------------------------
# Citation (Finaliser output)
# --------------------------------------------------------------------------


class Citation(TypedDict):
    chunk_id: str
    title: str


# --------------------------------------------------------------------------
# Top-level graph state
# --------------------------------------------------------------------------


class KestrelState(TypedDict):
    """The mutable bag of state flowing through every node.

    Fields are deliberately coarse-grained: each node reads what it needs
    and writes what it produces.  Retry and error fields have explicit ownership/merge rules.
    """

    # --- Input ----------------------------------------------------------------
    question: str
    """The raw user question for this turn."""

    history: list[dict[str, str]]
    """Previous turns: [{"role": "user"|"assistant", "content": "..."}]."""

    # --- Router / Planner output ---------------------------------------------
    standalone_question: str
    """Rewritten question with pronouns and ellipsis resolved."""

    route: str
    """One of: retrieval | chit_chat | off_topic."""

    sub_queries: list[str]
    """One focused search query per distinct thing the question asks."""

    router_reasoning: str
    """Free-text explanation of the routing decision (for debugging)."""

    # --- Retriever output ----------------------------------------------------
    evidence: list[EvidenceItem]
    """Chunks retrieved for this turn, merged across all sub_queries."""

    # --- Synthesiser output --------------------------------------------------
    draft_answer: str
    """Raw drafted answer before verification."""

    claims: list[Claim]
    """Factual assertions extracted by the Synthesiser, each with chunk ids."""

    missing_evidence: str
    """What the Synthesiser found missing (empty string when nothing is missing)."""

    # --- Verifier output -----------------------------------------------------
    verified_claims: list[VerifiedClaim]
    """Claims annotated with verdicts by the Verifier."""

    verification_verdict: str
    """The most severe verdict across all claims."""

    verification_notes: str
    """Verifier's overall notes (conflicts, version mismatches, etc.)."""

    # --- Finaliser output (also the terminal answer shown to the user) -------
    final_answer: str
    """Polished, user-facing answer."""

    citations: list[Citation]
    """Chunk ids and titles cited in the final answer."""

    # --- Retry logic ---------------------------------------------------------
    retries: int
    """Number of times the verifier has routed back to the retriever."""

    pending_query: str
    """Validated refined query awaiting one additional retrieval."""
    retrieval_queries: list[str]
    """Queries already executed, used to reject repeated refinements."""
    retrieval_attempts: int
    """Includes the initial retrieval; maximum two in the compiled workflow."""
    verification_status: str
    """not_requested | skipped | failed | performed; separate from verdict."""
    errors: list[str]
    """Sanitized stage error categories, including recovered provider failures."""

    # --- Error channel -------------------------------------------------------
    error: str
    """Non-empty when an unrecoverable error occurred; nodes should check this."""


def make_initial_state(question: str, history: list[dict[str, str]] | None = None) -> KestrelState:
    """Return a fully-initialised KestrelState for the start of a turn."""
    return KestrelState(
        question=question,
        history=[dict(turn) for turn in (history or [])],
        standalone_question="",
        route="",
        sub_queries=[],
        router_reasoning="",
        evidence=[],
        draft_answer="",
        claims=[],
        missing_evidence="",
        verified_claims=[],
        verification_verdict="",
        verification_notes="",
        final_answer="",
        citations=[],
        retries=0,
        error="",
        pending_query="",
        retrieval_queries=[],
        retrieval_attempts=0,
        verification_status="not_requested",
        errors=[],

    )


__all__ = [
    "EvidenceItem",
    "Claim",
    "VerifiedClaim",
    "Citation",
    "KestrelState",
    "make_initial_state",
]
