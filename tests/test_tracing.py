"""Tests for LangSmith tracing utilities (kestrel.tracing).

All tests run without real LangSmith credentials.  The tracing module degrades
gracefully when not configured — no imports fail and no exceptions are raised.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Clear any ambient tracing keys before the module under test is imported.
for _var in (
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_PROJECT",
):
    os.environ.pop(_var, None)

# Clear LLM keys so graph tests use the offline path.
for _var in ("KESTREL_LLM_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY",
             "OPENAI_API_KEY", "DEEPSEEK_API_KEY"):
    os.environ.pop(_var, None)


# ---------------------------------------------------------------------------
# 1. is_tracing_enabled
# ---------------------------------------------------------------------------


def test_tracing_disabled_without_any_vars(monkeypatch):
    """No env vars → tracing is disabled."""
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    from kestrel.tracing import is_tracing_enabled
    assert is_tracing_enabled() is False


def test_tracing_disabled_when_only_key_set(monkeypatch):
    """API key alone (without LANGCHAIN_TRACING_V2=true) → disabled."""
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__fake")
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

    from kestrel.tracing import is_tracing_enabled
    assert is_tracing_enabled() is False


def test_tracing_disabled_when_only_flag_set(monkeypatch):
    """LANGCHAIN_TRACING_V2=true alone (no key) → disabled."""
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    from kestrel.tracing import is_tracing_enabled
    assert is_tracing_enabled() is False


def test_tracing_enabled_with_both_vars(monkeypatch):
    """Both LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY set → enabled."""
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__fake")

    from kestrel.tracing import is_tracing_enabled
    assert is_tracing_enabled() is True


def test_tracing_enabled_accepts_1_as_flag(monkeypatch):
    """LANGCHAIN_TRACING_V2=1 is also accepted."""
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "1")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__fake")

    from kestrel.tracing import is_tracing_enabled
    assert is_tracing_enabled() is True


def test_tracing_enabled_accepts_langsmith_api_key(monkeypatch):
    """LANGSMITH_API_KEY is also accepted (alternative var name)."""
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "ls__fake")
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

    from kestrel.tracing import is_tracing_enabled
    assert is_tracing_enabled() is True


# ---------------------------------------------------------------------------
# 2. build_run_config
# ---------------------------------------------------------------------------


def test_build_run_config_contains_run_id():
    """build_run_config must always return a non-empty run_id."""
    from kestrel.tracing import build_run_config

    cfg = build_run_config()
    assert "run_id" in cfg
    assert cfg["run_id"]
    # Must be a valid UUID4.
    uuid.UUID(cfg["run_id"], version=4)


def test_build_run_config_each_call_unique():
    """Two consecutive calls must produce different run_ids."""
    from kestrel.tracing import build_run_config

    cfg1 = build_run_config()
    cfg2 = build_run_config()
    assert cfg1["run_id"] != cfg2["run_id"]


def test_build_run_config_includes_conversation_id():
    """conversation_id appears in the metadata dict."""
    from kestrel.tracing import build_run_config

    cfg = build_run_config(conversation_id="test-session-42")
    assert cfg["metadata"]["conversation_id"] == "test-session-42"


def test_build_run_config_includes_model_name():
    """model_name appears in the metadata dict when provided."""
    from kestrel.tracing import build_run_config

    cfg = build_run_config(model_name="llama-3.3-70b-versatile")
    assert cfg["metadata"]["model_name"] == "llama-3.3-70b-versatile"


def test_build_run_config_no_conversation_id_omits_field():
    """When no conversation_id is given, the key must not be present."""
    from kestrel.tracing import build_run_config

    cfg = build_run_config()
    assert "conversation_id" not in cfg["metadata"]


def test_build_run_config_includes_system_tag():
    """The system metadata field must identify the Kestrel assistant."""
    from kestrel.tracing import build_run_config

    cfg = build_run_config()
    assert cfg["metadata"]["system"] == "kestrel-research-assistant"


def test_build_run_config_merges_extra_metadata():
    """Extra metadata is merged into the metadata dict."""
    from kestrel.tracing import build_run_config

    cfg = build_run_config(extra_metadata={"env": "test", "version": "2.0"})
    assert cfg["metadata"]["env"] == "test"
    assert cfg["metadata"]["version"] == "2.0"


def test_build_run_config_includes_kestrel_tag():
    """'kestrel' must appear in the tags list."""
    from kestrel.tracing import build_run_config

    cfg = build_run_config()
    assert "kestrel" in cfg.get("tags", [])


# ---------------------------------------------------------------------------
# 3. get_trace_url
# ---------------------------------------------------------------------------


def test_get_trace_url_returns_none_when_tracing_disabled(monkeypatch):
    """Without tracing configured, get_trace_url must return None."""
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

    from kestrel.tracing import get_trace_url
    assert get_trace_url("some-run-id") is None


def test_get_trace_url_returns_none_for_none_run_id(monkeypatch):
    """Passing None run_id always returns None (even if tracing is enabled)."""
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__fake")

    from kestrel.tracing import get_trace_url
    assert get_trace_url(None) is None


def test_get_trace_url_returns_none_when_client_raises(monkeypatch):
    """When the LangSmith client raises any exception, returns None."""
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__fake")

    # Patch the Client to raise an error (simulates bad credentials / network).
    from unittest.mock import MagicMock, patch
    mock_client = MagicMock()
    mock_client.read_run.side_effect = Exception("Connection refused")

    from kestrel.tracing import get_trace_url
    with patch("kestrel.tracing.Client", return_value=mock_client):
        result = get_trace_url("some-uuid")
    assert result is None


def test_get_trace_url_returns_url_from_api(monkeypatch):
    """When the client returns a run with a URL, that URL is returned."""
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", "ls__fake")

    from unittest.mock import MagicMock, patch

    fake_run = MagicMock()
    fake_run.url = "https://smith.langchain.com/runs/abc-123"
    mock_client = MagicMock()
    mock_client.read_run.return_value = fake_run

    from kestrel.tracing import get_trace_url
    with patch("kestrel.tracing.Client", return_value=mock_client):
        url = get_trace_url("abc-123")

    assert url == "https://smith.langchain.com/runs/abc-123"
    mock_client.read_run.assert_called_once_with("abc-123")


# ---------------------------------------------------------------------------
# 4. Integration: run_query accepts conversation_id
# ---------------------------------------------------------------------------


def test_run_query_accepts_conversation_id():
    """run_query must accept conversation_id kwarg without crashing."""
    from kestrel.graph import build_graph, run_query

    graph = build_graph()
    state = run_query(
        "Hello, who are you?",
        graph=graph,
        conversation_id="test-conv-001",
    )
    # Should complete without error; chit_chat route expected.
    assert state["final_answer"]


def test_run_query_traced_returns_tuple():
    """run_query_traced must return a (state, trace_url) tuple."""
    from kestrel.graph import build_graph, run_query_traced

    graph = build_graph()
    result = run_query_traced(
        "Hello, who are you?",
        graph=graph,
        conversation_id="test-conv-002",
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    state, trace_url = result
    assert state["final_answer"]
    # Without tracing configured, trace_url must be None (not a fabricated URL).
    assert trace_url is None


def test_run_query_traced_trace_url_none_without_config():
    """Without LANGCHAIN_API_KEY, trace_url must be None."""
    from kestrel.graph import build_graph, run_query_traced

    graph = build_graph()
    _, trace_url = run_query_traced("Hello", graph=graph)
    assert trace_url is None
