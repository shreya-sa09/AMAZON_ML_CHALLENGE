# FINAL_V3_CHAR_EF05 — candidate submission (not yet submitted)

Branch: `final/v3`. Rollback point: `approaches/FINAL_HN_ONLY` (leaderboard 0.981172). Previous candidate: `FINAL_V2_HN_ATE_POLICY`.

## Pipeline
Unchanged retrieval (country TF-IDF, CountSketch, top-40 → exact top-10) and Stage 1 (shared cache) →
Stage 2 LightGBM (Approach 02 recipe), 116 features = Approach 02's 103 + house-number offset/parity 6 (Exp 10)
+ cross-fitted address token odds 6 (Exp 11) + **cross-fitted char-level pair model score 1 (Exp 17)** →
**expected-F0.5 subset selection per S1** on density-calibrated probabilities (odds × 0.5; Exp 18).

Char model (`src/charmodel.py`): MatchPyramid-style CNN over name / address character-similarity matrices, trained
from scratch (2 epochs, ~4-5 min per model on RTX 4050, 3 GB VRAM). Model A trained on hash half h0 scores h1 + test,
model B trained on h1 scores h0 + test; test score = mean(A, B).

## Validation (fold B) and leaderboard proxy (Exp 15: LB ≈ spread-decoy k=2.4 proxy − 0.0019)
| | Plain F0.5 | LB proxy | Real leaderboard |
|---|---:|---:|---:|
| FINAL_HN_ONLY @0.70 | 0.986268 | 0.981442 | 0.981172 |
| FINAL_V2 (ATE + group rule) | 0.986073 | 0.982344 | not submitted |
| **FINAL_V3** seed 42 / seed 7 | **0.986630 / 0.986648** | **0.983103 / 0.983126** | not submitted |
`--verify` reproduces the char scores exactly (max abs diff 0.0) and plain 0.986630 / proxy 0.983103.
Expected leaderboard ≈ 0.9828-0.9831 (estimate).

## Full test run (`results/final_summary.json`, `results/full_run_log.txt`)
Runtime 1,421 s (GPU char models + CPU LightGBM). Stage-2 best_iteration 3609.
Test-shift gate: V3 model @ uniform 0.70 accepts 5,843,351 vs FINAL_HN_ONLY 5,850,201 (−0.12%); with EF0.5 5,801,265 (−0.84%).
Char score vs stage-1 p1 correlation on test band: US 0.83, India 0.81, France 0.71 (works on unseen France).
Matched records 5,801,265 (India 2,710,608 · US 2,231,258 · France 859,399); S1 with match 1,633,316; empty 99,228.
Official validator PASS (default and `--check-ids`); independent checks: headers, 1,732,544 unique S1 = test set,
no S2/S3 ID matched twice, 0 foreign IDs.

## Reproduce
From this folder with `PYTHONDONTWRITEBYTECODE=1`: `python src/final_run.py --verify`, then `python src/final_run.py`.
Submission file (local, gitignored): `C:\Users\shrey\amazon-ml-challenge-2026\approaches\FINAL_V3_CHAR_EF05\output\matching_results.tsv`
