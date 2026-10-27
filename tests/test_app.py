from pathlib import Path
from streamlit.testing.v1 import AppTest

from kestrel import graph

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_ui_shows_unverified_status_even_without_citations(monkeypatch):
    def stream(*args, **kwargs):
        yield "verifier", "completed", {
            "verification_status": "failed", "verification_verdict": "insufficient_evidence",
            "errors": ["verifier: RuntimeError"],
        }
        yield "finaliser", "completed", {"final_answer": "Cannot verify an answer.", "citations": []}
    monkeypatch.setattr(graph, "stream_question", stream)
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    app.chat_input[0].set_value("KQL timeout?").run()
    assert not app.exception
    assert any("Verification failed" in w.value for w in app.warning)
    assert app.session_state["messages"][-1]["verification_status"] == "failed"
    # The status survives rerender/history, not just the streaming response.
    app.run()
    assert any("Verification failed" in w.value for w in app.warning)


def test_ui_source_inspection_and_conversation_reset(monkeypatch):
    def stream(*args, **kwargs):
        yield "retriever", "completed", {"evidence": [
            {"chunk_id": "spec:1", "title": "Limits", "text": "Timeout: 60 seconds.",
             "doc_id": "spec", "metadata": {}}], "retrieval_attempts": 1}
        yield "verifier", "completed", {"verification_status": "performed", "verification_verdict": "supported"}
        yield "finaliser", "completed", {
            "final_answer": "60 seconds [spec:1]", "citations": [{"chunk_id": "spec:1", "title": "Limits"}]}
    monkeypatch.setattr(graph, "stream_question", stream)
    app = AppTest.from_file(str(APP)).run()
    app.chat_input[0].set_value("KQL timeout?").run()
    assert not app.exception
    assert any(t.value == "Timeout: 60 seconds." for t in app.text)
    assert any("Model assessment: supported" in c.value for c in app.caption)
    app.button[0].click().run()
    assert app.session_state["messages"] == []
