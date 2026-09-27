# Experiment 08 — Larger K for missing-address queries — DISCARDED

Branch: `experiment/08-missing-address-larger-k`. Reads shared caches and Approach 02 artifacts read-only.

Variant: for queries with a missing address only, approximate Top-80 → exact rescoring → Top-20, unioned with the saved Top-10. All other queries unchanged.

## Smoke (`src/smoke_k80.py`, 34 s CUDA)
1,437 known missing-address Top-10 misses (Experiment 07 sample): 291 true pairs recovered, but 14,549 new candidate pairs were added (14,258 negatives).

## Bounded validation (`src/validate_sample.py`, 67 s)
Fixed sample of 10,000 validation-fold missing-address queries (5,000 US + 5,000 India, 9,760 matched), re-retrieved, re-featurized and re-scored with the Approach 02 validation model. All other validation decisions were reused unchanged.

| | Baseline (reproduced) | Experiment |
|---|---:|---:|
| Macro F0.5 @0.70 | 0.98568978 | 0.98569751 |
| Precision | 0.99360891 | 0.99361321 |
| Recall | 0.96684866 | 0.96686186 |
| False singletons | 280 | 281 |
| Sample candidate recall | 78.41% | 82.62% |
| Sample correct Top-1 | 6,266 | 6,297 |
| Sample accepted true | 5,371 | 5,383 |

Delta F0.5 = +0.0000077. 411 true candidates were recovered, but only 31 net became the correct Top-1 and 12 net became accepted links.

**Decision: DISCARD.** Do not roll out. `src/analyze_transitions.py` (tiny decision-transition audit) was never run because of an interrupted session; it is not needed for the decision.

Audit note: for missing-address queries the query address vector is zero, so retrieval is already effectively name-only; a larger K mostly adds same-name decoys.
