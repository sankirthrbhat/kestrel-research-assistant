"""Agent nodes for the Kestrel multi-agent LangGraph workflow.

Each public function in this module is a LangGraph node: it receives a
``KestrelState`` dict and returns a *partial* dict with only the keys it
writes.  LangGraph merges the partial update back into the running state.

Offline fallback (no API key)
------------------------------
Every node has a deterministic offline path that does NOT call an LLM.
When ``LLMConfig.configured`` is False the node uses rule-based heuristics
instead.  This guarantees the full graph can run -- and the CLI/tests can
exercise it -- with zero credentials.

The offline router uses keyword matching; the offline synthesiser builds a
plain-text excerpts from the top retrieved chunks; offline verification is
explicitly skipped. The deterministic finaliser renders assessed claims or
clearly labeled unverified source excerpts.

JSON parsing
------------
Every LLM agent that returns JSON wraps its call in a try/except that falls
back to the offline path if the model returns unparseable output.  A bad
completion degrades quality but never raises an unhandled exception.

LangSmith tracing
-----------------
Retrieval calls are wrapped with ``@traceable(run_type="retriever")`` when
LangSmith is configured.  The decorator is applied lazily so that importing
this module never fails when langsmith is absent.  Every LLM call is a child
run of its parent node span, automatically captured by LangGraph's built-in
LangSmith integration.
"""

from __future__ import annotations

import json
import re
from typing import Any

from . import config
from .llm import LLMClient, LLMNotConfigured
from .prompts import (
    OFFLINE_CHIT_CHAT,
    OFFLINE_NO_EVIDENCE,
    OFFLINE_OFF_TOPIC,
    ROUTER_SYSTEM,
    ROUTER_USER,
    SYNTHESIS_SYSTEM,
    SYNTHESIS_USER,
    VERIFIER_SYSTEM,
    VERIFIER_USER,
    render,
)
from .retrieval import Retriever, SearchResult, get_retriever
from .state import (
    Citation,
    Claim,
    EvidenceItem,
    KestrelState,
    VerifiedClaim,
)
from .tracing import traceable_retrieval


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

_KESTREL_KEYWORDS = re.compile(
    r"\b(kestrel|beacon|funnel|cohort|trail|kql|warehouse|sync|redshift|"
    r"bigquery|snowflake|plan|starter|growth|scale|api|sdk|retention|"
    r"gdpr|onboarding|incident|outage|postmortem|runbook|query|timeout|"
    r"limit|rate|throttl|pricing|feature|analytic|event|session|user)\b",
    re.IGNORECASE,
)

_CHIT_CHAT_KEYWORDS = re.compile(
    r"\b(hello|hi|hey|thanks|thank you|great|bye|goodbye|who are you|"
    r"what can you|help me|how do you)\b",
    re.IGNORECASE,
)


def _format_history(history: list[dict[str, str]]) -> str:
    if not history:
        return "(none)"
    lines: list[str] = []
    for turn in history[-6:]:  # keep last 3 pairs
        role = turn.get("role", "user").capitalize()
        lines.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(lines)


def _format_evidence(evidence: list[EvidenceItem]) -> str:
    if not evidence:
        return "(no evidence retrieved)"
    parts: list[str] = []
    for item in evidence:
        meta = item.get("metadata", {})
        parts.append(
            f"[{item['chunk_id']}] (title: {item['title']}, score: {item['score']:.3f}, "
            f"version: {meta.get('version', '')}, published: {meta.get('published', '')})\n"
            f"{item['text']}"
        )
    return "\n\n".join(parts)


def _format_metadata(evidence: list[EvidenceItem]) -> str:
    if not evidence:
        return "(none)"
    rows: list[str] = []
    for item in evidence:
        m = item.get("metadata", {})
        rows.append(
            f"chunk_id={item['chunk_id']!r} title={item['title']!r} "
            f"version={m.get('version', '')!r} published={m.get('published', '')!r}"
        )
    return "\n".join(rows)


def _format_claims(claims: list[Claim]) -> str:
    if not claims:
        return "(no claims extracted)"
    return json.dumps(claims, indent=2)


def _safe_json(text: str) -> dict[str, Any] | None:
    """Try to parse a JSON object from ``text``; return None on failure."""
    text = text.strip()
    # Strip markdown fences like ```json ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        # Try to extract the first {...} object from the text (models sometimes
        # add a trailing explanation after the JSON).
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                value = json.loads(m.group(0))
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                pass
    return None


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value.strip()


def _strings(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError(f"{field} must be a list of nonempty strings")
    return list(dict.fromkeys(x.strip() for x in value))


def _claims(value: Any) -> list[Claim]:
    if not isinstance(value, list):
        raise ValueError("claims must be a list")
    result = []
    for c in value:
        if not isinstance(c, dict):
            raise ValueError("claim must be an object")
        text = _text(c.get("claim"), "claim")
        if not text:
            raise ValueError("claim must not be empty")
        result.append(Claim(claim=text, chunk_ids=_strings(c.get("chunk_ids"), "chunk_ids")))
    if len({c["claim"] for c in result}) != len(result):
        raise ValueError("duplicate claims")
    return result


def _failure(state: KestrelState, node: str, exc: Exception) -> dict[str, Any]:
    # Record only the category: provider response bodies can contain private data.
    return {"errors": [*state.get("errors", []), f"{node}: {type(exc).__name__}"]}


def _normal_query(query: str) -> str:
    return " ".join(re.findall(r"\w+", query.casefold()))


def _search_results_to_evidence(results: list[SearchResult]) -> list[EvidenceItem]:
    return [
        EvidenceItem(
            chunk_id=r.chunk_id,
            title=r.title,
            text=r.text,
            score=r.score,
            doc_id=r.doc_id,
            metadata=dict(r.metadata),
        )
        for r in results
    ]


def _get_client() -> tuple[LLMClient, bool]:
    """Return (client, is_configured).  Client is always returned even when
    unconfigured so nodes can choose their offline path."""
    client = LLMClient()
    return client, client.config.configured


# --------------------------------------------------------------------------
# Node: Router / Planner
# --------------------------------------------------------------------------


def _offline_router(question: str, history: list[dict[str, str]]) -> dict[str, Any]:
    """Keyword-based heuristic routing when no LLM is configured."""
    if _CHIT_CHAT_KEYWORDS.search(question) and not _KESTREL_KEYWORDS.search(question):
        route = "chit_chat"
        sub_queries: list[str] = []
        reasoning = "Offline: matched chit_chat keywords, no Kestrel keywords."
    elif _KESTREL_KEYWORDS.search(question):
        route = "retrieval"
        sub_queries = [question]
        reasoning = "Offline: matched Kestrel keywords."
    else:
        # Be generous: when in doubt, try retrieval.
        route = "retrieval"
        sub_queries = [question]
        reasoning = "Offline: no strong signal; defaulting to retrieval."
    return {
        "standalone_question": question,  # no rewrite offline
        "route": route,
        "sub_queries": sub_queries,
        "router_reasoning": reasoning,
    }


def router_node(state: KestrelState) -> dict[str, Any]:
    """Rewrite, classify (retrieval / chit_chat / off_topic) and decompose."""
    question = state["question"]
    history = state["history"]

    client, configured = _get_client()

    if not configured:
        return _offline_router(question, history)

    prompt = render(ROUTER_USER, history=_format_history(history), question=question)
    try:
        raw = client.complete(prompt, system=ROUTER_SYSTEM, temperature=0.0)
        parsed = _safe_json(raw)
        if not parsed:
            raise ValueError(f"Unparseable router output: {raw!r}")
        route = _text(parsed.get("route", "retrieval"), "route")
        if route not in ("retrieval", "chit_chat", "off_topic"):
            raise ValueError("invalid route")
        standalone = _text(parsed.get("standalone_question", question), "standalone_question")
        queries = _strings(parsed.get("sub_queries", [standalone]), "sub_queries")
        return {
            "standalone_question": standalone or question,
            "route": route,
            "sub_queries": queries or [standalone or question],
            "router_reasoning": _text(parsed.get("reasoning", ""), "reasoning"),
        }
    except (LLMNotConfigured, RuntimeError, ValueError) as exc:
        return {**_offline_router(question, history), **_failure(state, "router", exc)}


# --------------------------------------------------------------------------
# Node: Retriever
# --------------------------------------------------------------------------


@traceable_retrieval(name="kestrel_retriever")
def _traced_search(
    retriever: Retriever,
    query: str,
    k: int,
    strategy: str,
) -> list[SearchResult]:
    """Single-query corpus search wrapped with a LangSmith retriever span."""
    return retriever.search_corpus(query, k=k, strategy=strategy)


def retriever_node(
    state: KestrelState,
    retriever: Retriever | None = None,
) -> dict[str, Any]:
    """Run search_corpus for every sub-query and merge evidence."""
    sub_queries = state.get("sub_queries") or [state["standalone_question"] or state["question"]]
    pending = state.get("pending_query", "")
    retrying = bool(pending and state.get("retries", 0) < 1)
    if retrying:
        sub_queries = [pending]
    _retriever = retriever or get_retriever()

    seen_ids: set[str] = set()
    merged: list[EvidenceItem] = []

    for q in sub_queries:
        if not q or not q.strip():
            continue
        results = _traced_search(
            _retriever,
            q,
            k=config.DEFAULT_K,
            strategy=config.DEFAULT_STRATEGY,
        )
        for item in _search_results_to_evidence(results):
            if item["chunk_id"] not in seen_ids:
                seen_ids.add(item["chunk_id"])
                merged.append(item)

    # Re-sort by score descending after merge.
    merged.sort(key=lambda x: x["score"], reverse=True)

    return {
        "evidence": merged,
        "retries": state.get("retries", 0) + int(retrying),
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        "retrieval_queries": [*state.get("retrieval_queries", []), *sub_queries],
        "pending_query": "",
    }


# --------------------------------------------------------------------------
# Node: Synthesiser
# --------------------------------------------------------------------------


def _offline_synthesiser(
    question: str, evidence: list[EvidenceItem]
) -> dict[str, Any]:
    """Build a plain-text draft from the top retrieved chunks."""
    if not evidence:
        return {
            "draft_answer": OFFLINE_NO_EVIDENCE,
            "claims": [],
            "missing_evidence": "No evidence was retrieved for this question.",
        }
    top = evidence[:4]
    bullet_parts = [f"• [{item['chunk_id']}] {item['text'][:300].strip()}" for item in top]
    # «[OFFLINE]» prefix distinguishes deterministic answers from real LLM answers.
    draft = (
        "[OFFLINE] Based on the knowledge base (no LLM configured):\n\n"
        + "\n\n".join(bullet_parts)
    )
    claims: list[Claim] = [
        {"claim": item["text"][:120].strip(), "chunk_ids": [item["chunk_id"]]}
        for item in top
    ]
    return {
        "draft_answer": draft,
        "claims": claims,
        "missing_evidence": "",
    }


def synthesiser_node(state: KestrelState) -> dict[str, Any]:
    """Draft an answer from the retrieved evidence."""
    question = state["standalone_question"] or state["question"]
    evidence = state["evidence"]

    client, configured = _get_client()

    if not configured:
        return _offline_synthesiser(question, evidence)

    evidence_text = _format_evidence(evidence)
    prompt = render(SYNTHESIS_USER, question=question, evidence=evidence_text)
    try:
        raw = client.complete(prompt, system=SYNTHESIS_SYSTEM, temperature=0.0)
        parsed = _safe_json(raw)
        if not parsed:
            raise ValueError(f"Unparseable synthesiser output: {raw!r}")
        return {
            "draft_answer": _text(parsed.get("draft_answer", ""), "draft_answer"),
            "claims": _claims(parsed.get("claims", [])),
            "missing_evidence": _text(parsed.get("missing_evidence", ""), "missing_evidence"),
        }
    except (LLMNotConfigured, RuntimeError, ValueError) as exc:
        return {**_offline_synthesiser(question, evidence), **_failure(state, "synthesiser", exc)}


# --------------------------------------------------------------------------
# Node: Verifier / Critic
# --------------------------------------------------------------------------


_VERDICT_ORDER = {
    "supported": 1,
    "partially_supported": 2,
    "conflicting_evidence": 3,
    "insufficient_evidence": 4,
}


def _offline_verifier(
    claims: list[Claim], evidence: list[EvidenceItem], draft: str,
    *, status: str = "skipped", reason: str = "No model verification performed.",
) -> dict[str, Any]:
    """Evidence being retrieved is not proof that a draft claim is supported."""
    return {
        "verified_claims": [
            VerifiedClaim(claim=c["claim"], verdict="insufficient_evidence",
                          chunk_ids=[], reason=reason) for c in claims
        ],
        "verification_verdict": "insufficient_evidence",
        "verification_status": status,
        "verification_notes": reason,
        "pending_query": "",
    }


def verifier_node(state: KestrelState) -> dict[str, Any]:
    """Validate a complete claim assessment; never upgrade missing evidence."""
    question = state["standalone_question"] or state["question"]
    draft, claims, evidence = state["draft_answer"], state["claims"], state["evidence"]
    client, configured = _get_client()
    if not configured or not claims or not evidence:
        reason = ("No model configured; sources are unverified." if not configured
                  else "No claims or evidence available for verification.")
        return _offline_verifier(claims, evidence, draft, reason=reason)

    prompt = render(VERIFIER_USER, question=question, draft=draft,
                    claims=_format_claims(claims), evidence=_format_evidence(evidence))
    try:
        parsed = _safe_json(client.complete(prompt, system=VERIFIER_SYSTEM, temperature=0.0))
        if not parsed:
            raise ValueError("Expected verifier JSON object")
        raw_claims = parsed.get("claims")
        checked = _claims(raw_claims)
        # Exact text matching makes omissions, replacements and additions detectable.
        expected = {c["claim"] for c in claims}
        if {c["claim"] for c in checked} != expected:
            raise ValueError("Verifier must assess each original claim exactly once")
        known_ids = {e["chunk_id"] for e in evidence}
        verified: list[VerifiedClaim] = []
        for c in raw_claims:
            verdict = c.get("verdict")
            if verdict not in _VERDICT_ORDER:
                raise ValueError("Invalid claim verdict")
            ids = _strings(c["chunk_ids"], "chunk_ids")
            reason = _text(c.get("reason", ""), "reason")
            # A model cannot establish support by citing absent or invented sources.
            if not ids or not set(ids) <= known_ids:
                verdict = "insufficient_evidence"
                reason = "Missing or unknown evidence identifiers."
            verified.append(VerifiedClaim(
                claim=c["claim"].strip(), verdict=verdict,
                chunk_ids=[i for i in ids if i in known_ids], reason=reason,
            ))
        worst = max((c["verdict"] for c in verified), key=_VERDICT_ORDER.get)
        top = parsed.get("verdict", worst)
        if top not in _VERDICT_ORDER:
            raise ValueError("Invalid overall verdict")
        verdict = max((worst, top), key=_VERDICT_ORDER.get)
        refined = _text(parsed.get("refined_query", ""), "refined_query")
        previous = [question, *state.get("sub_queries", []), *state.get("retrieval_queries", [])]
        useful = bool(_normal_query(refined)) and _normal_query(refined) not in {
            _normal_query(q) for q in previous
        }
        needs_more = verdict in ("insufficient_evidence", "partially_supported")
        return {
            "verified_claims": verified,
            "verification_verdict": verdict,
            "verification_status": "performed",
            "verification_notes": _text(parsed.get("notes", ""), "notes"),
            "pending_query": refined if needs_more and useful and state.get("retries", 0) < 1 else "",
        }
    except (LLMNotConfigured, RuntimeError, ValueError, TypeError) as exc:
        return {
            **_offline_verifier(claims, evidence, draft, status="failed",
                                reason="Verification failed; sources remain unverified."),
            **_failure(state, "verifier", exc),
        }


# --------------------------------------------------------------------------
# Finaliser: deterministic formatting prevents new post-verification claims.
# --------------------------------------------------------------------------


def _source_excerpts(evidence: list[EvidenceItem]) -> dict[str, Any]:
    top = evidence[:4]
    return {
        "final_answer": "Unverified source excerpts — these may not answer your question.\n\n"
                        + "\n\n".join(f"[{e['chunk_id']}] {e['text'][:300].strip()}" for e in top),
        "citations": [Citation(chunk_id=e["chunk_id"], title=e["title"]) for e in top],
    }


def _offline_finaliser(route: str, draft: str, evidence: list[EvidenceItem],
                       verdict: str) -> dict[str, Any]:
    if route == "chit_chat":
        return {"final_answer": OFFLINE_CHIT_CHAT, "citations": []}
    if route == "off_topic":
        return {"final_answer": OFFLINE_OFF_TOPIC, "citations": []}
    if not evidence:
        return {"final_answer": OFFLINE_NO_EVIDENCE, "citations": []}
    return _source_excerpts(evidence)


def finaliser_node(state: KestrelState) -> dict[str, Any]:
    """Render only assessed claims; never polish the unchecked draft with an LLM."""
    route, evidence = state["route"], state["evidence"]
    verdict = state.get("verification_verdict") or "insufficient_evidence"
    if route in ("chit_chat", "off_topic") or not evidence:
        return _offline_finaliser(route, "", evidence, verdict)
    if state.get("verification_status") != "performed":
        return _source_excerpts(evidence)

    known = {e["chunk_id"]: e for e in evidence}
    supported, cited = [], []
    for claim in state.get("verified_claims", []):
        ids = claim["chunk_ids"]
        if (claim["verdict"] == "supported" and ids and set(ids) <= known.keys()
                and all(i in known for i in re.findall(r"\[([^\[\]]+)\]", claim["claim"]))):
            supported.append(claim["claim"] + " " + " ".join(f"[{i}]" for i in ids))
            cited.extend(ids)
    # Conflicts and an overall insufficient verdict must not become a confident
    # answer merely because one individual claim was marked supported.
    if verdict in ("insufficient_evidence", "conflicting_evidence") or not supported:
        message = ("The retrieved sources conflict; I cannot give a settled answer."
                   if verdict == "conflicting_evidence" else OFFLINE_NO_EVIDENCE)
        excerpts = _source_excerpts(evidence)
        return {"final_answer": message + "\n\n" + excerpts["final_answer"],
                "citations": excerpts["citations"]}
    prefix = ("Only part of the question is supported; other claims remain unverified.\n\n"
              if verdict == "partially_supported" else "")
    return {
        "final_answer": prefix + "\n\n".join(supported),
        "citations": [Citation(chunk_id=i, title=known[i]["title"]) for i in dict.fromkeys(cited)],
    }


# --------------------------------------------------------------------------
# Routing logic (used by the graph conditional edge)
# --------------------------------------------------------------------------


def route_after_router(state: KestrelState) -> str:
    """Return the edge label for the conditional branch after router_node."""
    route = state.get("route", "retrieval")
    if route in ("chit_chat", "off_topic"):
        return "finalise"  # skip retrieval + synthesis + verification
    return "retrieve"


def route_after_verifier(state: KestrelState) -> str:
    """Return the edge label for the conditional branch after verifier_node."""
    if (state.get("verification_status") == "performed"
            and state.get("pending_query")
            and state.get("retries", 0) < 1):
        return "retriever"
    return "finaliser"


__all__ = [
    "router_node",
    "retriever_node",
    "synthesiser_node",
    "verifier_node",
    "finaliser_node",
    "route_after_router",
    "route_after_verifier",
]
