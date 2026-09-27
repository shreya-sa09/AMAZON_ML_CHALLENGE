# Experiment 11 — Cross-fitted address token odds — DISCARDED

Branch: `experiment/11-stage2-address-token-odds`. Harness copied from Experiment 10 (`src/common.py`, recipe identical to
Approach 02 `stage2.fit`; only change: optional `ER_LGB_SEED` env override of the LightGBM seed, default 42).
Baseline = Experiment 10 R1 (Approach 02 + house-number features, reused read-only from `10/cache/feat_hn_*`).

## Hypothesis
Address analogue of Approach 02's name token odds: smoothed P(true | non-numeric address token appears only in the
query / only in S1), cross-fitted across the hash halves exactly like stage2 (`src/ate.py`); 6 features
`ate_q_min, ate_q_max, ate_q_unseen, ate_s_min, ate_s_max, ate_s_unseen`. Fold-A evidence: frequent one-sided tokens
such as `hn`/`plot`/`block` (y≈0.7) vs `rd`/`st`/single letters (y≈0.1–0.25), base 0.37.

## Results (validation fold B, threshold 0.70)
| | R1 seed 42 | **R2 seed 42** | R1 seed 7 | **R2 seed 7** |
|---|---:|---:|---:|---:|
| Macro F0.5 | 0.986268 | 0.986354 | 0.986308 | 0.986313 |
| Precision | 0.99396 | 0.99405 | 0.99400 | 0.99401 |
| Recall | 0.96785 | 0.96791 | 0.96793 | 0.96788 |
| False singletons | 286 | 273 | 276 | 274 |
| Band AUC | 0.99402 | 0.99415 | 0.99407 | 0.99417 |
| Addr-present rejected correct | 11,014 | 10,959 | 10,958 | 10,974 |
| Addr-present accepted wrong | 3,526 | 3,364 | 3,502 | 3,395 |

Gain R2 − R1: +0.000086 (seed 42), +0.000005 (seed 7); mean +0.000046 ≈ seed noise (R1 seed 42 vs 7: 0.000040).
Predicted 0.98630–0.98667 (midpoint 0.98648); actual 0.986354 / 0.986313.
Runtime: estimated 11–13 min for R2 (ATE build estimated 3–5 min, actual 19 s); actual R2 305 s; seed check 2×~290 s.

**Decision: DISCARD.** AUC and accepted-wrong improve consistently, but the F0.5 gain is not reproducible across seeds.
Not carried forward. (May be reconsidered only inside a later measured combination.)
