# Experiment 13 — Label-free sibling evidence Stage-2 features — KEEP

Branch: `experiment/13-stage2-sibling-evidence`. Harness copied from Exp 11 (recipe identical to Approach 02 `stage2.fit`).
Baseline = Exp 10 R1 (Approach 02 + house-number features, `10/cache/feat_hn_*` reused read-only).

## Hypothesis
Pair-level features cannot separate the remaining look-alike errors (Exp 12 residual analysis). Siblings = other
S2/S3 records whose Stage-1 top-1 (`cache/train_best` / `test_best`) is the same S1. Pre-check on fold-A training rows
(`src/diag_siblings.py`, `results/diag_siblings.txt`): when a sibling has an identical normalized address, y-rate is
0.844 (offset 1–2 group) / 0.853 (offset >2) vs 0.11–0.26 otherwise.

Features (`src/sib.py`, 8, label-free, never uses sibling model scores): `sib_n, sib_n_other_src, sib_max_ad,
sib_n_ad_exact, sib_max_nm, sib_n_nm_exact, sib_n_both_exact, sib_ad_adv` (similarities capped at 50 siblings; -1 = n/a).

## Results (validation fold B, threshold 0.70)
| | R1 s42 | **R3 s42** | R1 s7 | **R3 s7** |
|---|---:|---:|---:|---:|
| Macro F0.5 | 0.986268 | **0.987043** | 0.986308 | **0.986976** |
| Precision | 0.99396 | 0.99432 | 0.99400 | 0.99428 |
| Recall | 0.96785 | 0.96931 | 0.96793 | 0.96917 |
| False singletons | 286 | 305 | 276 | 311 |
| Band AUC | 0.99402 | 0.99569 | 0.99407 | 0.99567 |
| Addr-present rejected correct | 11,014 | 8,283 | 10,958 | 8,439 |
| Addr-present accepted wrong | 3,526 | 2,764 | 3,502 | 2,722 |
| F0.5 US / India | — | 0.987505 / 0.986354 | — | 0.987453 / 0.986263 |

Gain +0.000775 (s42) / +0.000668 (s7), ~17× seed noise. Predicted 0.98657–0.98777 (midpoint 0.98717).
Runtime: estimated ~12 min for both seeds; actual 307 s + 288 s ≈ 10 min.

**Decision: KEEP.** Combined with Exp 10: 0.985690 (A02 saved) → 0.987043 / 0.986976.
