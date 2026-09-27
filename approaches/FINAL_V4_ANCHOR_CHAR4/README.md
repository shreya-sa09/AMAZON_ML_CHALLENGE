# FINAL_V4_ANCHOR_CHAR4 — submitted, best confirmed leaderboard result

Branch: `final/v4` (submitted run, commit d32e861); score record on `final/v4-score-0.982899`.
Rollbacks: `FINAL_HN_ONLY` (leaderboard 0.981172), `FINAL_V3_CHAR_EF05`.

## ACTUAL LEADERBOARD RESULT
| | |
|---|---|
| **Score (macro F0.5, Amazon leaderboard)** | **0.982899** |
| Previous best (FINAL_HN_ONLY) | 0.981172 |
| Improvement | **+0.001727** |
| Validation F0.5 (seed 42 / seed 7) | 0.988200 / 0.988251 |
| Leaderboard proxy (seed 42 / seed 7) | 0.985049 / 0.985097 (overstated the gain ~2×: +0.0036 predicted vs +0.0017 actual) |
| Test matched records | 5,830,482 |
| Official validator | PASS |
| Validator `--check-ids` | PASS |
| Submission status | submitted and scored |

This is the best confirmed leaderboard result in the project and the real baseline for future experiments.
The submitted file is `output/matching_results.tsv` from the run documented below (code/config unchanged).

## Pipeline
Unchanged retrieval + Stage 1 → Stage 2 LightGBM (Approach 02 recipe), 122 features = Approach 02's 103
+ house-number 6 (Exp 10) + address token odds 6 (Exp 11) + 4-epoch cross-fitted char pair model score 1 (Exp 17/21)
+ confident-anchor sibling features 6 (Exp 19: other records with Stage-1 p1 ≥ 0.9 on the same S1)
→ expected-F0.5 subset selection per S1 on density-calibrated probabilities (odds × 0.5; Exp 18).

## Validation (fold B) and leaderboard proxy (Exp 15: LB ≈ spread-decoy k=2.4 proxy − 0.0019; estimate)
| | Plain F0.5 (s42 / s7) | LB proxy (s42 / s7) | Real LB |
|---|---:|---:|---:|
| FINAL_HN_ONLY | 0.986268 / 0.986308 | 0.981442 | 0.981172 |
| FINAL_V3 | 0.986630 / 0.986648 | 0.983103 / 0.983126 | not submitted |
| **FINAL_V4** | **0.988200 / 0.988251** | **0.985049 / 0.985097** | not submitted |
`--verify` reproduces char scores and anchor features exactly (max abs diff 0.0) and plain 0.988200 / proxy 0.985049.

## Full test run (`results/final_summary.json`, `results/full_run_log.txt`)
Runtime 1,645 s (char models 2×~7 min GPU, Stage-2 fit ~5 min, best_iteration 2117).
Test-shift gate: V4 model @ uniform 0.70 +0.29% vs FINAL_HN_ONLY; EF0.5 accepts 5,830,482 (V3 5,801,265: +29k, as
predicted by the Exp 19 shift check). Char-score vs stage-1 correlation: US 0.84, India 0.83, France 0.69.
Matched records 5,830,482 (India 2,722,388 · US 2,243,117 · France 864,977); S1 with match 1,633,677; empty 98,867.
Official validator PASS (default and `--check-ids`); independent checks: headers, 1,732,544 unique S1 = test set,
no S2/S3 ID matched twice, 0 foreign IDs.

Submission file (local, gitignored): `C:\Users\shrey\amazon-ml-challenge-2026\approaches\FINAL_V4_ANCHOR_CHAR4\output\matching_results.tsv`
