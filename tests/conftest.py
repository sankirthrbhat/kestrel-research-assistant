"""Shared pytest fixtures for the Kestrel test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kestrel import config, ingest, retrieval  # noqa: E402


def _index_available() -> bool:
    if not (config.BM25_PATH.exists() and config.MANIFEST_PATH.exists()):
        return False
    manifest = ingest.read_manifest()
    if not manifest:
        return False
    return manifest.get("corpus_sha256") == ingest.sha256_of_file(config.CORPUS_PATH)


@pytest.fixture(scope="session")
def corpus_chunks():
    """All corpus chunks, loaded fresh from corpus.jsonl."""
    return ingest.load_corpus()


@pytest.fixture(scope="session")
def retriever(request):
    """A Retriever over the persisted index; skips if the index is stale."""
    if not request.config.getoption("--integration"):
        pytest.skip("Use --integration after building the local index")
    if not _index_available():
        pytest.fail("Index missing or stale. Run: python -m kestrel.ingest")
    return retrieval.Retriever()

def pytest_addoption(parser):
    parser.addoption("--integration", action="store_true",
                     help="Run real-model/index tests after local ingestion")


@pytest.fixture(autouse=True)
def isolated_credentials(monkeypatch):
    """Tests never load a developer's .env or make unmocked provider calls."""
    from kestrel import llm
    import urllib.request
    for key in ("KESTREL_LLM_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY",
                "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "KESTREL_OFFLINE",
                "LANGCHAIN_API_KEY", "LANGSMITH_API_KEY",
                "LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(llm, "_load_dotenv_if_available", lambda: None)
    def no_network(*args, **kwargs):
        raise AssertionError("Unexpected provider HTTP call in test")
    monkeypatch.setattr(urllib.request, "urlopen", no_network)
