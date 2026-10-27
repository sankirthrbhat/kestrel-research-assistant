"""Deterministic teaching demo: ALL model/retrieval responses here are mocked."""
import json
from unittest.mock import Mock, patch

from . import agents, graph
from .retrieval import SearchResult

QUESTION = "What is the KQL timeout?"
CLAIM = {"claim": "The KQL timeout is 60 seconds", "chunk_ids": ["demo:1"]}


def demonstrate(case):
    """Exercise the actual compiled graph using explicitly controlled fixtures."""
    retriever = Mock()
    retriever.search_corpus.return_value = [
        SearchResult("demo:1", "The KQL timeout is 60 seconds.", 1.0, "Demonstration fixture")]
    checks = 0
    def complete(prompt, *, system, **kwargs):
        nonlocal checks
        if system == agents.ROUTER_SYSTEM:
            return json.dumps({"route": "retrieval", "sub_queries": [QUESTION]})
        if system == agents.SYNTHESIS_SYSTEM:
            return json.dumps({"draft_answer": CLAIM["claim"], "claims": [CLAIM]})
        checks += 1
        insufficient = case == "evidence_gap" or (case == "one_retry" and checks == 1)
        verdict = "insufficient_evidence" if insufficient else "supported"
        return json.dumps({
            "claims": [{**CLAIM, "verdict": verdict, "reason": "Scripted demonstration assessment"}],
            "verdict": verdict, "refined_query": "KQL execution timeout specification" if case == "one_retry" else "",
        })
    client = Mock()
    client.complete.side_effect = complete
    # build_run_config is isolated too: the teaching demo never uploads traces.
    with patch.object(agents, "_get_client", return_value=(client, True)), \
         patch.object(agents, "get_retriever", return_value=retriever), \
         patch.object(graph, "build_run_config", return_value={"run_id": "00000000-0000-0000-0000-000000000001",
                                                               "callbacks": []}), \
         patch.dict("os.environ", {"LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "false"}):
        result = graph.run_query(QUESTION, graph=graph.build_graph())
    return {"case": case, "mode": "mocked teaching demonstration",
            "retrieval_attempts": result["retrieval_attempts"], "retries": result["retries"],
            "verification_status": result["verification_status"],
            "verification_verdict": result["verification_verdict"],
            "answer": result["final_answer"]}


def main():
    print("CONTROL-FLOW DEMO — mocked responses, not measured answer accuracy.")
    for case in ("supported", "evidence_gap", "one_retry"):
        print(json.dumps(demonstrate(case), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
