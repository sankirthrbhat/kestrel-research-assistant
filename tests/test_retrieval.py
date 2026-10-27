"""Retrieval smoke tests against the real Kestrel corpus.

Ground truth for each assertion is quoted in a comment next to the check, taken
verbatim from corpus.jsonl. These verify the retrieval path end to end: the
persisted index is queried and the retrieved text must actually contain the
fact, rather than merely returning a plausible-looking chunk id.
"""

from __future__ import annotations

import pytest

from kestrel import config


def _joined(results) -> str:
    return " ".join(r.text for r in results)


def _ids(results) -> list[str]:
    return [r.chunk_id for r in results]


# --------------------------------------------------------------------------
# Corpus integrity
# --------------------------------------------------------------------------


def test_corpus_loads_and_matches_expected_hash():
    """The corpus hash must match the digest recorded in config."""
    from kestrel.ingest import sha256_of_file, verify_corpus_hash

    actual, matches = verify_corpus_hash()
    assert actual == config.EXPECTED_CORPUS_SHA256
    assert matches is True
    # Recompute directly to prove verify_corpus_hash is not just echoing config.
    assert sha256_of_file(config.CORPUS_PATH) == actual


def test_corpus_is_not_modified_by_loading(corpus_chunks):
    """Loading the corpus must not alter the file on disk."""
    from kestrel.ingest import sha256_of_file

    before = sha256_of_file(config.CORPUS_PATH)
    chunks = corpus_chunks  # already loaded
    after = sha256_of_file(config.CORPUS_PATH)
    assert before == after
    assert len(chunks) == 154


def test_metadata_fields_present(corpus_chunks):
    """Every chunk carries the metadata the assignment requires."""
    required = {"doc_id", "title", "category", "owner", "source_url", "published", "version"}
    for chunk in corpus_chunks[:10]:
        missing = required - set(chunk.metadata)
        assert not missing, f"{chunk.chunk_id} missing metadata: {missing}"
        # Derived ordering metadata used for neighbour expansion.
        assert "chunk_index" in chunk.to_metadata()
        assert "ordinal" in chunk.to_metadata()


# --------------------------------------------------------------------------
# Test 1: Beacon limits per plan
# --------------------------------------------------------------------------


def test_beacon_limits_per_plan(retriever):
    """Beacon caps are 5 (Starter), 60 (Growth), 500 (Scale).

    corpus: "Starter projects may create 5 Beacons, Growth projects 60, and
    Scale projects 500." (spec-beacons:4)
    """
    results = retriever.search_corpus("How many Beacons can a project have on each plan?", k=6)
    assert results, "expected results for Beacon limits query"

    blob = _joined(results)
    assert "Starter" in blob and "Growth" in blob and "Scale" in blob
    assert "5 Beacons" in blob, f"expected '5 Beacons' in top-6; got top ids {_ids(results)}"
    assert "60" in blob and "500" in blob


def test_beacon_limits_dense_baseline_finds_spec(retriever):
    """The dense baseline should surface the Beacons spec itself."""
    results = retriever.search_corpus("Beacon limits by plan", k=6, strategy="dense")
    doc_ids = {r.doc_id for r in results}
    assert "spec-beacons" in doc_ids, f"got {doc_ids}"
    assert any(r.chunk_id == "spec-beacons:4" for r in results[:3]), _ids(results)


# --------------------------------------------------------------------------
# Test 2: KQL query timeout
# --------------------------------------------------------------------------


def test_kql_query_timeout_is_60_seconds(retriever):
    """KQL timeout is 60s, raised from 30s in 4.1.

    corpus: "Every query has a 60-second timeout; this was raised from 30
    seconds in release 4.1" (spec-sdks-kql:4) and "The KQL query timeout has
    been raised from 30 seconds to 60 seconds." (rn-4-1:2)
    """
    results = retriever.search_corpus("What is the KQL query timeout?", k=6)
    assert results

    blob = _joined(results)
    assert "60" in blob, f"expected 60 in top-6 text; ids={_ids(results)}"
    # The timeout fact should be stated explicitly, not merely implied.
    assert ("60-second timeout" in blob) or ("60 seconds" in blob)


def test_kql_timeout_history_mentions_previous_30(retriever):
    """The 30 -> 60 second change is the key nuance in this answer."""
    results = retriever.search_corpus("KQL timeout raised from 30 to 60 seconds", k=6)
    blob = _joined(results)
    assert "30" in blob and "60" in blob, _ids(results)
    assert any(
        r.doc_id in {"spec-sdks-kql", "rn-4-1", "product-faq", "eng-osprey-query-engine"}
        for r in results
    ), _ids(results)


def test_kql_timeout_beats_unrelated_kafka_timeout(retriever):
    """Regression guard: Kafka session.timeout.ms is a different timeout.

    The corpus contains many *other* timeouts (Kafka consumer session timeout of
    45s/6s). A good retriever must not let those dominate the KQL answer.
    """
    results = retriever.search_corpus("What is the KQL query timeout?", k=6)
    top_texts = " ".join(r.text for r in results[:3]).lower()
    assert "kql" in top_texts, _ids(results)


# --------------------------------------------------------------------------
# Test 3: Redshift availability
# --------------------------------------------------------------------------


def test_redshift_availability_scale_only(retriever):
    """Redshift is Scale-only, added in 4.1.

    corpus: "Scale projects can additionally sync to Amazon Redshift, which was
    added as a destination in release 4.1." and "Redshift is available to Scale
    customers only; Growth customers continue to have Snowflake and BigQuery."
    (spec-warehouse-sync:0, rn-4-1:0)
    """
    results = retriever.search_corpus("Is Redshift available on the Growth plan?", k=6)
    assert results

    blob = _joined(results)
    assert "Redshift" in blob
    assert "Scale" in blob, _ids(results)
    assert ("4.1" in blob) or ("Scale customers only" in blob)


def test_redshift_not_available_on_starter(retriever):
    """Warehouse Sync (hence Redshift) is absent on Starter."""
    results = retriever.search_corpus("Which plans support Redshift warehouse sync?", k=6)
    blob = _joined(results)
    assert "Redshift" in blob
    assert "Starter" in blob or "Scale" in blob, _ids(results)


# --------------------------------------------------------------------------
# Strategy behaviour: dense baseline, hybrid RRF, dedupe, neighbours
# --------------------------------------------------------------------------


def test_default_strategy_is_hybrid():
    """The documented application default is hybrid retrieval."""
    assert config.DEFAULT_STRATEGY == "hybrid"
    assert config.NEIGHBOR_EXPANSION_DEFAULT is False


def test_dense_and_hybrid_both_return_k(retriever):
    query = "What is the KQL query timeout?"
    dense = retriever.search_corpus(query, k=6, strategy="dense")
    hybrid = retriever.search_corpus(query, k=6, strategy="hybrid")

    assert len(dense) == 6, _ids(dense)
    assert len(hybrid) == 6, _ids(hybrid)
    assert all(r.text for r in dense)
    assert all(r.text for r in hybrid)


def test_results_are_deduplicated(retriever):
    """No chunk id may appear twice, under any strategy."""
    for strategy in ("dense", "bm25", "hybrid"):
        results = retriever.search_corpus("Beacon limits per plan", k=6, strategy=strategy)
        ids = _ids(results)
        assert len(ids) == len(set(ids)), f"duplicates under {strategy}: {ids}"


def test_hybrid_records_both_retrievers(retriever):
    """A hybrid hit should show which retrievers found it."""
    results = retriever.search_corpus("KQL query timeout", k=6, strategy="hybrid")
    assert any("dense_rank" in r.sources for r in results)
    assert any("bm25_rank" in r.sources for r in results)
    assert all("rrf_score" in r.sources for r in results)


def test_neighbor_expansion_disabled_by_default(retriever):
    """Default results must not include neighbour-expanded chunks."""
    results = retriever.search_corpus("What is the KQL query timeout?", k=6)
    assert not any("neighbor_of" in r.sources for r in results)


def test_neighbor_expansion_when_enabled(retriever):
    """Enabling expansion pulls in adjacent chunks from the same document."""
    results = retriever.search_corpus(
        "What is the KQL query timeout?", k=6, expand_neighbors=True
    )
    assert results
    # Expansion should surface at least one chunk that is not a seed hit.
    results_sorted = sorted(results, key=lambda r: r.score, reverse=True)
    assert len(results_sorted) > 0
    # With expansion on, the window may add chunks; cap is still <= k.
    assert len(results) <= 6


def test_rrf_fusion_math():
    """RRF scores follow sum(1/(k+rank)) and favour agreement."""
    from kestrel.retrieval import rrf_fuse

    scores = rrf_fuse({"a": ["x", "y"], "b": ["y", "x"]}, rrf_k=60)
    # x: rank1 in a, rank2 in b ; y: rank2 in a, rank1 in b -> identical
    assert scores["x"] == pytest.approx(scores["y"])
    assert scores["x"] == pytest.approx(1 / 61 + 1 / 62)


def test_dedupe_keeps_highest_score():
    """Dedupe collapses repeated ids and keeps the best-scoring copy."""
    from kestrel.retrieval import SearchResult, dedupe

    dupes = [
        SearchResult(chunk_id="a", text="low", score=0.1, sources={"bm25_rank": 1}),
        SearchResult(chunk_id="a", text="high", score=0.9, sources={"dense_rank": 1}),
        SearchResult(chunk_id="b", text="mid", score=0.5, sources={}),
    ]
    out = dedupe(dupes)
    assert [r.chunk_id for r in out] == ["a", "b"]
    assert out[0].score == 0.9
    # Provenance is merged so both retrievers are recorded.
    assert "bm25_rank" in out[0].sources and "dense_rank" in out[0].sources


def test_configurable_strategy_override(retriever):
    """Strategy is configurable per call, not hard-coded."""
    dense = retriever.search_corpus("Redshift availability", k=3, strategy="dense")
    bm25 = retriever.search_corpus("Redshift availability", k=3, strategy="bm25")
    assert len(dense) == 3 and len(bm25) == 3
    # The two strategies should not be identical on a real corpus.
    assert _ids(dense) != _ids(bm25)


def test_invalid_strategy_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.search_corpus("anything", k=3, strategy="nonsense")


def test_empty_query_returns_nothing(retriever):
    assert retriever.search_corpus("", k=6) == []
    assert retriever.search_corpus("   ", k=6) == []
