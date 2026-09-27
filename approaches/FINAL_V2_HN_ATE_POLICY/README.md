# FINAL_V2_HN_ATE_POLICY — candidate submission (not yet submitted)

Branch: `final/v2`. Rollback point: `approaches/FINAL_HN_ONLY` (leaderboard 0.981172).

## What changed vs FINAL_HN_ONLY
1. **Address token odds** (Exp 11): 6 cross-fitted features, 115 features total. Seed-unstable on plain validation,
   but consistently positive at both seeds under the leaderboard proxy (+0.00009 to +0.00012).
2. **Group-aware decision rule** (Exp 15f): threshold **0.85** when another record already points to the same S1 with
   p_final ≥ 0.90, otherwise **0.70**. Macro F0.5 is per S1; test decoys (~2.4× train density) pile onto S1s that
   already have matches, while lone true matches decide an S1's whole score.

## Leaderboard proxy (Exp 15)
Fold-B validation with every accepted decoy link counted k times on random fold-B S1s. Spread k≈2.4 reproduces the real
leaderboard difference between Approach 02 @0.50 (0.977984) and FINAL_HN_ONLY @0.70 (0.981172) and both absolute scores
with a constant offset of ~0.0019: **LB ≈ proxy − 0.0019**.

| | Plain validation F0.5 | LB proxy | Real leaderboard |
|---|---:|---:|---:|
| Approach 02 @0.50 | 0.985908 (0.984778 consistent) | ≈0.9779 | 0.977984 |
| FINAL_HN_ONLY @0.70 | 0.986268 | 0.981442 | 0.981172 |
| **FINAL_V2** (seed 42 / seed 7) | 0.986073 / 0.986042 | **0.982344 / 0.982299** | not submitted |

Expected leaderboard ≈ 0.9821 (+0.0009 vs 0.981172). Estimate only.

## Verification and test-shift gates
- `--verify`: validation features rebuilt through this code are identical to the Exp 10/11 caches; reproduces plain 0.986354 (uniform 0.70) and proxy 0.982344 exactly.
- Test-shift gate: V2 model at uniform 0.70 accepts 5,841,745 test records vs 5,850,201 for FINAL_HN_ONLY (−0.14%);
  with the rule 5,795,889 (−0.93%; validation −0.55%). No sibling-style blow-up.

## Full run (`results/final_summary.json`, `results/full_run_log.txt`)
Runtime 738 s (CPU, 16 threads). best_iteration 4579. Matched records 5,795,889
(India 2,707,377 · US 2,229,068 · France 859,444). S1 with matches 1,632,638; empty 99,906.
Official validator: PASS (default and `--check-ids`). Independent checks: headers, 1,732,544 unique S1 = test set,
no S2/S3 ID matched twice, 0 foreign IDs.

Submission file (local, gitignored): `C:\Users\shrey\amazon-ml-challenge-2026\approaches\FINAL_V2_HN_ATE_POLICY\output\matching_results.tsv`
