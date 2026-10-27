# Understanding the Kestrel repairs

## 1. The crash: a default argument still executes

Before:
```python
final_verdict = str(parsed.get("verdict", worst))
```

Python must evaluate both arguments before calling dict.get. The undefined name worst therefore crashed even when the JSON already contained verdict.

Now, the verifier first validates the response, computes the most cautious claim verdict, and compares it with the overall verdict. An optimistic overall verdict cannot upgrade a claim that lacks evidence. Missing overall verdicts use the calculated value. Invalid verdicts fail explicitly.

Test to explain: a valid response with a supported claim and real source ID used to raise NameError; now it produces a performed assessment.

## 2. The retry: preparing work is different from starting it

Before, the verifier set retries to 1 while preparing the query. The graph then checked retries < 1 and skipped retrieval.

Now:
1. The verifier validates a new, nonempty refinement and stores pending_query.
2. The conditional edge checks that query and the retry budget.
3. The retriever starts the extra attempt, increments retries and clears pending_query.
4. The graph re-synthesizes and re-verifies.
5. No second extra attempt is permitted.

retrieval_attempts counts the initial search stage too. A normal answer has 1 attempt and 0 retries; a refined answer has 2 attempts and 1 retry. Each user turn starts fresh.

A refinement differing only in case, spacing or punctuation is not new. Missing refinements finish cautiously. The graph recursion limit is a secondary guard, not the loop's primary stopping rule.

Test to explain: the real compiled graph receives insufficient evidence twice; it terminates after exactly two retrieval stages.

## 3. Structured output must actually be validated

JSON parsing only proves that text is JSON. It does not prove that claims is a list, that source IDs exist, or that the model assessed every input claim.

The repaired verifier requires:
- An object containing a list of claim objects.
- Each original claim text exactly once, with no additions, omissions or duplicates.
- A verdict from the defined set.
- Source IDs represented as a list of nonempty strings.
- Existing retrieved IDs for any claim to remain supported.

Unknown/absent source IDs downgrade a claim to insufficient evidence. Malformed responses produce a failed verification status. Exact claim matching is intentionally conservative and can reject harmless paraphrases; the prompt tells the model to copy input claim text exactly.

The router and synthesizer also validate their structured outputs. Provider failures record sanitized error categories even when a fallback allows the graph to finish.

## 4. “Found a passage” does not mean “verified an answer”

The old offline verifier automatically labeled claims supported. It performed no check.

The repaired state separates:
- verification_status: not_requested, skipped, failed, performed.
- verification_verdict: supported, partially_supported, conflicting_evidence, insufficient_evidence.

Without a model, the assistant displays labeled unverified excerpts. It does not republish a potentially invented draft. The UI shows status even when no citations exist and preserves it in conversation history. Source text, version, publication metadata and document identity are inspectable.

A performed assessment still means “a model checked this,” not “this is certainly true.”

## 5. The finalizer should not invent new facts after checking

Previously, a final model call rewrote a draft using metadata and could introduce new claims or fabricated citations.

Now finalization is deterministic. It:
- Formats supported assessed claims with real source IDs and canonical titles.
- Excludes unassessed draft additions.
- Preserves partial-support warnings.
- Refuses a settled answer for conflicting/insufficient evidence and shows labeled source excerpts.
- Makes no extra model call.

Tradeoff: output is less polished and more conservative. That is easier to test and explain. The verifier can still make semantic mistakes, and this is not a general prompt-injection defense.

## 6. Failure is not a quality score

The old evaluator read verifier_verdict from state, but the actual field is verification_verdict. It also assigned some judge failures zero and treated unsupported questions as automatically faithful.

The replacement evaluator:
- Reads the real fields and retrieved IDs.
- Runs conversations in explicit turn order.
- Separates completed, failed and skipped cases.
- Records recovered provider failures as failed cases for quality aggregation.
- Skips later turns when their prerequisite turn failed.
- Leaves model-judged quality unscored unless --judge is requested.
- Reports judge failures separately from real zero scores.
- Never interprets the verifier's own label as independent faithfulness.
- Reports unknown token totals as null.
- Saves settings, source/corpus/dataset hashes, versions and scored sample sizes.

Recall@k is the fraction of annotated relevant chunks found in the first k distinct results. MRR@k is 1/rank of the first relevant hit, or zero if no hit exists. Cases without relevance labels have no retrieval score; they are not perfect results.

These retrieval metrics do not establish answer accuracy. The existing 20-case dataset is a development set, with 17 cases carrying relevance labels.

## 7. Repeatable setup and tests

The desktop checkout had a broken Git HEAD reference and Python environments pointing at a missing interpreter. Repairs were made in a fresh checkout instead of rewriting that folder.

A Windows clone also converted corpus line endings from LF to CRLF and broke its byte checksum. The new .gitattributes preserves LF. Corpus content remains identical to the committed source.

Dependencies are documented for Python 3.12. Lightweight tests need no embedding model or credentials; --integration explicitly enables real-index checks. A missing or stale index fails integration tests rather than silently producing a green run with skips. Test fixtures prevent loading personal .env credentials and block unmocked provider HTTP requests.

The original suite contained stale expectations: offline support and a dense default despite a hybrid application default. Those expectations now match the intended behavior. New tests exercise failure modes and actual graph transitions, not just happy-path output.

## Three-minute interview walkthrough

1. **Problem:** I wanted answers traceable to a local document corpus.
2. **Retrieval:** Dense embeddings capture semantic similarity; BM25 captures exact terminology. RRF adds reciprocal ranks from both lists, avoiding direct comparison of incompatible raw scores.
3. **Workflow:** Router → retrieval → draft claims → verifier → finalizer, with one useful refinement allowed.
4. **Bug example:** Explain why dict.get evaluated an undefined default and why incrementing retries too early skipped work.
5. **Honesty:** Offline excerpts are not verified answers. A failed provider is not a wrong-answer score.
6. **Evidence:** Show tests, inspect one real retrieved source, and run the explicitly mocked three-case control-flow demo.
7. **Limit:** I have a small development evaluation; live model quality and production reliability require further validation.

Describe your actual contribution and how you reviewed these assisted repairs. Do not imply that you independently authored work you cannot explain.

## Next useful work after this repair

Run the configured workflow with a provider/model you have access to, inspect supported/unsupported/conflicting cases manually, then create a separate held-out set before making answer-quality claims. More agents, a frontend rewrite, or fine-tuning are not prerequisites for presenting the current retrieval/workflow engineering accurately.
