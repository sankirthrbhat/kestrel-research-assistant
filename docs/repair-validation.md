# Repair validation — 27 October 2026

## Scope

Repairs started from commit ec30538b3cbb0231dc6520e7730db912c030b56a on the public repository. Work was performed in a fresh checkout because the desktop copy had a broken Git HEAD reference and unusable Python environments. Its core Python source matched the fresh checkout; the original folder was preserved.

This report distinguishes actual retrieval/model loading, mocked configured-provider tests, and live external-model testing.

## Defects reproduced before changes

Two new regression tests failed on the original source:
1. A valid configured verifier response raised NameError because worst was undefined.
2. After isolating that crash, insufficient evidence plus a refinement routed directly to the finalizer instead of retrying.

The initial existing-suite run, before building an index, returned 57 passed, 23 skipped and 11 failed. Failures included missing retrieval dependencies/index, the Windows corpus line-ending checksum mismatch, outdated dense-default expectations, and the quota exception mismatch. This was an environment/source baseline, not an answer-accuracy measurement.

## Completed checks

- Full test suite with --integration: **151 passed**, zero failures, zero skips.
- Separate clean environment with only requirements-core.txt: **120 passed, 31 intentionally skipped real-index tests**.
- Real BAAI/bge-small-en-v1.5 weights downloaded and loaded.
- All 154 corpus chunks embedded into 384-dimensional vectors and indexed in Chroma and BM25.
- Corpus bytes match the original committed SHA-256.
- Configured-provider paths tested with mocked valid, malformed and failed responses.
- Actual compiled LangGraph tested for supported responses, evidence gaps, exactly one retry, repeated weak evidence, new-turn reset and provider failures.
- Streamlit AppTest checked status without citations, history rerender, source inspection and conversation reset.
- Three-case mocked teaching demo executed successfully.
- pip check: no broken requirements.
- Python source compilation succeeded.
- One upstream LangGraph/LangChain pending-deprecation warning remains; tests passed.
- Lightweight CI configuration added for Python 3.12 on Windows and Linux. Hosted CI status is separate from local checks.

The application finalizer is deliberately deterministic, so there is no separate untested finalizer model call.

## Fresh retrieval evaluation

Same local corpus, BAAI embedding model, 20 development queries and k=5 for both runs. Seventeen queries have relevance labels; the three unsupported queries have no Recall/MRR score. Raw follow-ups were not rewritten in this retrieval-only comparison.

| Configuration | Recall@5 | MRR@5 | Scored queries | Workflow failures |
| --- | ---: | ---: | ---: | ---: |
| Dense | 0.6186 | 0.6882 | 17 | 0 / 20 |
| Hybrid RRF | 0.8000 | 0.7990 | 17 | 0 / 20 |

Reports:
- [Dense](../results/runs/2026-10-27-dense/metrics_summary.json)
- [Hybrid](../results/runs/2026-10-27-hybrid/metrics_summary.json)
- [Offline workflow](../results/runs/2026-10-27-offline/metrics_summary.json)

The offline workflow completed all 20 cases, with no model judging. Its verification was skipped, not supported. Quality metrics were left unscored. Every report retains the source hashes and settings from its execution; the final metadata-prompt regression and documentation were added afterwards and do not change these offline retrieval calculations.

These measurements show an improvement on this small development set. They do not reproduce the exact denominators of older reports, establish generalization, or measure answer accuracy. No held-out validation or human-rated semantic evaluation was performed.

## What has not been established

- No live external LLM-provider calls were made for this repair.
- No end-to-end live-provider answer accuracy, model-based faithfulness, or token totals are claimed.
- No production reliability, security certification, or real user impact is claimed.
- Hosted CI is not considered passed merely because its configuration exists.
- No live-provider screenshots were fabricated; interface behavior was checked through Streamlit's test harness.

## How to demonstrate it

1. Install the dependencies and build/reuse the index.
2. Run python -m kestrel.demo. Explain that it uses scripted responses to demonstrate control flow.
3. Run python -m streamlit run app.py in offline mode. Ask about Beacon limits and inspect actual retrieved excerpts.
4. Explain why the answer remains unverified without a model.
5. When a provider is configured, manually inspect supported, conflicting and unsupported questions before making claims about semantic quality.

## Accurate resume wording

- Built a hybrid retrieval prototype combining dense embeddings and BM25 with Reciprocal Rank Fusion over a 154-chunk document corpus.
- Added validated claim assessments, one bounded retrieval retry, explicit unverified fallbacks and source inspection; covered workflow, UI and retrieval behavior with automated tests.

Use personal implementation language only to the extent that it reflects your own contribution and understanding. If mentioning measured retrieval performance, state that it was on 17 labeled development queries; avoid turning it into a general accuracy claim.
