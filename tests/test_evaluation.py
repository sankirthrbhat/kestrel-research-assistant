import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from eval.run_eval import (evaluate_cases, judge_answer, ordered_rows, retrieval_scores,
                           state_output, summarize)


def row(qid="q1", **extra):
    return {"question_id": qid, "question": qid, "type": "single_hop",
            "expected_answer": "60 seconds", "expected_chunk_ids": ["spec:1"], **extra}


def state():
    return {"final_answer": "60 seconds [spec:1]", "evidence": [{"chunk_id": "spec:1", "text": "60 seconds"}],
            "verification_verdict": "supported", "verification_status": "performed",
            "citations": [{"chunk_id": "spec:1", "title": "Limits"}], "retries": 1,
            "retrieval_attempts": 2}


def test_known_retrieval_metrics():
    scores = retrieval_scores(["a", "b", "c"], ["b", "d"], 2)
    assert scores == {"retrieval_recall@k": .5, "retrieval_mrr@k": .5}
    assert retrieval_scores(["a"], ["b"], 5)["retrieval_recall@k"] == 0
    assert retrieval_scores(["a"], [], 5)["retrieval_mrr@k"] is None


def test_eval_reads_real_verdict_key():
    result = state_output(state())
    assert result["verification_verdict"] == "supported"
    assert result["verification_status"] == "performed"
    assert result["retrieved_chunk_ids"] == ["spec:1"]
    assert result["token_usage"] is None


def test_failures_excluded_but_counted_and_judging_not_implied():
    def runner(q, history):
        if q == "q2":
            raise RuntimeError("failure")
        return state()
    results = evaluate_cases([row(), row("q2")], runner)
    summary = summarize(results)
    assert summary["failure_rate"] == .5
    assert summary["metrics"]["retrieval_recall@k"] == {"mean": 1, "scored_cases": 1}
    assert "faithfulness" not in summary["metrics"]
    assert summary["quality_status_counts"] == {"skipped": 2}


def test_recovered_provider_errors_count_as_failed():
    results = evaluate_cases([row()], lambda q, h: {**state(), "errors": ["verifier: RuntimeError"]})
    assert results[0]["status"] == "failed"
    assert summarize(results)["metrics"] == {}


def test_conversations_sorted_and_history_preserved():
    rows = [row("q2", conversation_id="c", turn=2), row("q1", conversation_id="c", turn=1)]
    calls = []
    def runner(q, history):
        calls.append((q, history))
        return state()
    evaluate_cases(rows, runner)
    assert calls[0] == ("q1", [])
    assert calls[1][0] == "q2"
    assert calls[1][1][0] == {"role": "user", "content": "q1"}


def test_failed_turn_skips_follow_up():
    def runner(q, history):
        raise RuntimeError("failed")
    results = evaluate_cases([row("q1", conversation_id="c", turn=1),
                              row("q2", conversation_id="c", turn=2)], runner)
    assert [r["status"] for r in results] == ["failed", "skipped"]


@pytest.mark.parametrize("rows", [
    [row(), row()], [row("q2", conversation_id="c", turn=2)],
    [row("q1", conversation_id="c", turn=1), row("q2", conversation_id="c", turn=1)],
])
def test_invalid_dataset_order_rejected(rows):
    with pytest.raises(ValueError):
        ordered_rows(rows)


@pytest.mark.parametrize("reply", ["oops", '{"correctness":true}', '{"correctness":NaN}',
                                  '{"correctness":2,"relevance":1,"faithfulness":1}'])
def test_invalid_judge_output_unscored(reply):
    client = Mock(config=SimpleNamespace(configured=True))
    client.complete.return_value = reply
    result = judge_answer(row(), state_output(state()), [], client)
    assert result["status"] == "failed"
    assert result["scores"] == {}


def test_unsupported_is_not_automatically_faithful():
    client = Mock(config=SimpleNamespace(configured=True))
    client.complete.return_value = json.dumps({"correctness": 0, "relevance": 0, "faithfulness": 0})
    result = judge_answer(row(type="unsupported", expected_chunk_ids=[]), state_output(state()), [], client)
    assert result["scores"]["faithfulness"] == 0
    assert result["status"] == "scored"
