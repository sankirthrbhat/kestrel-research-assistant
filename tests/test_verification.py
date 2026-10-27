"""Configured-provider regressions: no keys, network, embeddings or index needed."""
import json
from unittest.mock import Mock

import pytest

from kestrel import agents
from kestrel.state import make_initial_state


@pytest.fixture
def evidence_state():
    state = make_initial_state("What is the KQL timeout?")
    state.update(
        route="retrieval", standalone_question=state["question"],
        sub_queries=[state["question"]], draft_answer="The timeout is 60 seconds [spec:1].",
        claims=[{"claim": "The timeout is 60 seconds", "chunk_ids": ["spec:1"]}],
        evidence=[{"chunk_id": "spec:1", "title": "Query limits", "doc_id": "spec",
                   "text": "The KQL timeout is 60 seconds.", "score": 1.0, "metadata": {}}],
    )
    return state


def response(verdict="supported", refined=""):
    return {"claims": [{"claim": "The timeout is 60 seconds", "chunk_ids": ["spec:1"],
                        "verdict": verdict, "reason": "Checked the source."}],
            "verdict": verdict, "notes": "", "refined_query": refined}


def configure(monkeypatch, payload):
    client = Mock()
    client.complete.return_value = json.dumps(payload)
    monkeypatch.setattr(agents, "_get_client", lambda: (client, True))
    return client


def test_valid_configured_verifier(evidence_state, monkeypatch):
    configure(monkeypatch, response())
    result = agents.verifier_node(evidence_state)
    assert result["verification_verdict"] == "supported"


def test_refined_query_routes_to_one_more_retrieval(evidence_state, monkeypatch):
    configure(monkeypatch, response("insufficient_evidence", "KQL execution timeout limits"))
    evidence_state.update(agents.verifier_node(evidence_state))
    assert agents.route_after_verifier(evidence_state) == "retriever"


@pytest.mark.parametrize("payload", [
    [], {"claims": None}, {"claims": []}, {"claims": ["bad"]},
    {"claims": [{"claim": "Changed claim", "chunk_ids": ["spec:1"], "verdict": "supported"}]},
    {"claims": [{"claim": "The timeout is 60 seconds", "chunk_ids": "spec:1", "verdict": "supported"}]},
    {"claims": [{"claim": "The timeout is 60 seconds", "chunk_ids": ["spec:1"], "verdict": "certain"}]},
    {**response(), "verdict": []}, {**response(), "refined_query": None},
    {**response(), "claims": response()["claims"] * 2},
])
def test_invalid_verification_fails_closed(payload, evidence_state, monkeypatch):
    configure(monkeypatch, payload)
    result = agents.verifier_node(evidence_state)
    assert result["verification_status"] == "failed"
    assert result["verification_verdict"] == "insufficient_evidence"
    assert result["pending_query"] == ""
    assert result["errors"]


def test_missing_top_verdict_uses_claim_verdict(evidence_state, monkeypatch):
    payload = response("partially_supported")
    del payload["verdict"]
    configure(monkeypatch, payload)
    assert agents.verifier_node(evidence_state)["verification_verdict"] == "partially_supported"


def test_overall_verdict_cannot_upgrade_claim(evidence_state, monkeypatch):
    payload = response("insufficient_evidence")
    payload["verdict"] = "supported"
    configure(monkeypatch, payload)
    assert agents.verifier_node(evidence_state)["verification_verdict"] == "insufficient_evidence"


@pytest.mark.parametrize("ids", [[], ["invented:4"], ["spec:1", "invented:4"]])
def test_unknown_or_missing_sources_never_supported(ids, evidence_state, monkeypatch):
    payload = response()
    payload["claims"][0]["chunk_ids"] = ids
    configure(monkeypatch, payload)
    result = agents.verifier_node(evidence_state)
    assert result["verification_verdict"] == "insufficient_evidence"
    assert all(i == "spec:1" for c in result["verified_claims"] for i in c["chunk_ids"])


@pytest.mark.parametrize("failure", [RuntimeError("provider failed"), ValueError("bad JSON")])
def test_provider_failure_remains_unverified(failure, evidence_state, monkeypatch):
    client = configure(monkeypatch, {})
    client.complete.side_effect = failure
    result = agents.verifier_node(evidence_state)
    assert result["verification_status"] == "failed"
    assert result["verification_verdict"] != "supported"
    assert result["pending_query"] == ""


def test_malformed_json_is_not_support(evidence_state, monkeypatch):
    client = configure(monkeypatch, {})
    client.complete.return_value = "{not JSON"
    assert agents.verifier_node(evidence_state)["verification_status"] == "failed"


@pytest.mark.parametrize("refined", ["", "  ", "WHAT is THE KQL timeout!!!", "..."])
def test_no_useful_refinement_finishes(refined, evidence_state, monkeypatch):
    configure(monkeypatch, response("insufficient_evidence", refined))
    evidence_state.update(agents.verifier_node(evidence_state))
    assert agents.route_after_verifier(evidence_state) == "finaliser"


@pytest.mark.parametrize("empty_field", ["claims", "evidence"])
def test_nothing_to_verify_skips(empty_field, evidence_state, monkeypatch):
    client = configure(monkeypatch, response())
    evidence_state[empty_field] = []
    result = agents.verifier_node(evidence_state)
    assert result["verification_status"] == "skipped"
    assert result["verification_verdict"] != "supported"
    client.complete.assert_not_called()


def test_offline_does_not_republish_unchecked_draft(evidence_state, monkeypatch):
    monkeypatch.setattr(agents, "_get_client", lambda: (Mock(), False))
    evidence_state["draft_answer"] = "Invented fact [not-real:1]"
    evidence_state.update(agents.verifier_node(evidence_state))
    result = agents.finaliser_node(evidence_state)
    assert evidence_state["verification_status"] == "skipped"
    assert "Unverified" in result["final_answer"]
    assert "Invented fact" not in result["final_answer"]
    assert result["citations"] == [{"chunk_id": "spec:1", "title": "Query limits"}]


def test_finalizer_uses_assessed_claims_and_canonical_titles(evidence_state, monkeypatch):
    configure(monkeypatch, response())
    evidence_state.update(agents.verifier_node(evidence_state))
    evidence_state["draft_answer"] += " Extra invented fact."
    result = agents.finaliser_node(evidence_state)
    assert "Extra invented" not in result["final_answer"]
    assert "[spec:1]" in result["final_answer"]
    assert result["citations"] == [{"chunk_id": "spec:1", "title": "Query limits"}]


@pytest.mark.parametrize("verdict", ["insufficient_evidence", "conflicting_evidence", "partially_supported"])
def test_uncertainty_survives_finalization(verdict, evidence_state, monkeypatch):
    payload = response()
    payload["verdict"] = verdict
    configure(monkeypatch, payload)
    evidence_state.update(agents.verifier_node(evidence_state))
    answer = agents.finaliser_node(evidence_state)["final_answer"]
    assert any(word in answer for word in ("cannot", "conflict", "Only part"))


def test_finalizer_rejects_invented_inline_citation(evidence_state, monkeypatch):
    configure(monkeypatch, response())
    evidence_state.update(agents.verifier_node(evidence_state))
    evidence_state["verified_claims"][0]["claim"] += " [invented:99]"
    assert "invented:99" not in agents.finaliser_node(evidence_state)["final_answer"]


def test_graph_executes_exactly_one_retry_and_resets(monkeypatch, evidence_state):
    from kestrel import graph
    from kestrel.retrieval import SearchResult
    retriever = Mock()
    retriever.search_corpus.return_value = [SearchResult(
        chunk_id="spec:1", text="The KQL timeout is 60 seconds.",
        score=1, title="Query limits", doc_id="spec",
    )]
    monkeypatch.setattr(agents, "get_retriever", lambda: retriever)
    calls = []
    def complete(prompt, *, system, **kwargs):
        if system == agents.ROUTER_SYSTEM:
            return json.dumps({"route": "retrieval", "sub_queries": [evidence_state["question"]]})
        if system == agents.SYNTHESIS_SYSTEM:
            return json.dumps({"draft_answer": evidence_state["draft_answer"], "claims": evidence_state["claims"]})
        calls.append("verify")
        return json.dumps(response("insufficient_evidence", "KQL execution timeout limits"))
    client = Mock()
    client.complete.side_effect = complete
    monkeypatch.setattr(agents, "_get_client", lambda: (client, True))
    compiled = graph.build_graph()
    for _ in range(2):
        result = graph.run_query(evidence_state["question"], graph=compiled)
        assert result["retrieval_attempts"] == 2
        assert result["retries"] == 1
        assert result["pending_query"] == ""
        assert len(result["retrieval_queries"]) == 2
        assert "cannot give a factual answer" in result["final_answer"]
    assert retriever.search_corpus.call_count == 4
    assert len(calls) == 4


@pytest.mark.parametrize("verdict,refined,expected_attempts", [
    ("supported", "", 1), ("insufficient_evidence", "", 1),
    ("partially_supported", "KQL timeout documentation", 2),
    ("conflicting_evidence", "", 1),
])
def test_full_graph_paths(verdict, refined, expected_attempts, monkeypatch, evidence_state):
    from kestrel import graph
    from kestrel.retrieval import SearchResult
    retriever = Mock()
    retriever.search_corpus.return_value = [SearchResult("spec:1", "Timeout is 60 seconds.", 1, "Query limits")]
    monkeypatch.setattr(agents, "get_retriever", lambda: retriever)
    client = Mock()
    def complete(prompt, *, system, **kwargs):
        if system == agents.ROUTER_SYSTEM:
            return json.dumps({"route": "retrieval", "sub_queries": [evidence_state["question"]]})
        if system == agents.SYNTHESIS_SYSTEM:
            return json.dumps({"draft_answer": evidence_state["draft_answer"], "claims": evidence_state["claims"]})
        return json.dumps(response(verdict, refined))
    client.complete.side_effect = complete
    monkeypatch.setattr(agents, "_get_client", lambda: (client, True))
    result = graph.run_query(evidence_state["question"], graph=graph.build_graph())
    assert result["retrieval_attempts"] == expected_attempts
    assert result["final_answer"]
    assert set(c["chunk_id"] for c in result["citations"]) <= {"spec:1"}


@pytest.mark.parametrize("node", [agents.router_node, agents.synthesiser_node])
@pytest.mark.parametrize("payload", [[], {"sub_queries": "bad", "claims": ["bad"]}])
def test_other_agents_validate_model_shapes(node, payload, evidence_state, monkeypatch):
    configure(monkeypatch, payload)
    result = node(evidence_state)
    assert result["errors"]


def test_full_graph_provider_failure_exits_with_unverified_sources(monkeypatch):
    from kestrel import graph
    from kestrel.retrieval import SearchResult
    retriever = Mock()
    retriever.search_corpus.return_value = [SearchResult("spec:1", "Timeout: 60 seconds", 1, "Limits")]
    monkeypatch.setattr(agents, "get_retriever", lambda: retriever)
    client = Mock()
    client.complete.side_effect = RuntimeError("provider unavailable")
    monkeypatch.setattr(agents, "_get_client", lambda: (client, True))
    result = graph.run_query("What is the KQL timeout?", graph=graph.build_graph())
    assert result["retrieval_attempts"] == 1
    assert result["verification_status"] == "failed"
    assert result["retries"] == 0
    assert len(result["errors"]) == 3
    assert "Unverified" in result["final_answer"]


def test_verifier_receives_version_and_publication_metadata(evidence_state, monkeypatch):
    evidence_state["evidence"][0]["metadata"] = {"version": "4.1", "published": "20250901"}
    client = configure(monkeypatch, response())
    agents.verifier_node(evidence_state)
    prompt = client.complete.call_args.args[0]
    assert "version: 4.1" in prompt
    assert "published: 20250901" in prompt


def test_explicit_offline_overrides_ambient_key(monkeypatch):
    from kestrel.llm import LLMConfig
    monkeypatch.setenv("KESTREL_OFFLINE", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert not LLMConfig.from_env().configured
