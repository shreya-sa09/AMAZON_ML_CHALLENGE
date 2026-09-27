# FINAL_HN_ONLY — production submission (Approach 02 + Exp 10 house-number features)

Branch: `final/hn-only`. **Submission file: `output/matching_results.tsv`** (candidate set: `output/candidate_pairs.tsv`).

## Summary
| | |
|---|---|
| Final approach | Approach 02 + Exp 10 house-number offset/parity features |
| Features | 109 |
| Threshold | 0.70 |
| Validation macro F0.5 | 0.986268 (Approach 02: 0.985690) |
| Test matched records | 5,850,201 |
| Official validator | PASS (default and `--check-ids`) |

Local submission files (gitignored because of size, not in the repository):
- `C:\Users\shrey\amazon-ml-challenge-2026\approaches\FINAL_HN_ONLY\output\matching_results.tsv` — the file uploaded to the portal
- `C:\Users\shrey\amazon-ml-challenge-2026\approaches\FINAL_HN_ONLY\output\candidate_pairs.tsv`

Reproduce: `set PYTHONDONTWRITEBYTECODE=1` then from this folder `python src/final_run.py --verify` (validation), `python src/final_run.py` (full run, ~8.5 min),
then `python utils/validate_submission.py --matching approaches/FINAL_HN_ONLY/output/matching_results.tsv --candidate approaches/FINAL_HN_ONLY/output/candidate_pairs.tsv --test-dir dataset/test` from the repo root.

## Configuration (frozen, `config/final_config.json`)
- Retrieval / blocking / candidates: unchanged shared cache (country TF-IDF, CountSketch 1024, approx top-40 → exact top-10).
- Stage 1: baseline LightGBM (shared cache `test_best`). Stage 2: Approach 02 recipe (edit features + cross-fitted name
  token odds, runner-up off), trained on both training folds, **+ 6 house-number offset/parity features** (Exp 10).
- 109 features; threshold **0.70**; band [0.002, 0.995]; LightGBM params identical to Approach 02 (seed 42).
- Excluded: sibling evidence (Exp 13/14 — test acceptance shift ~25% vs ~1.4% on validation), address token odds
  (Exp 11), raw style (Exp 12), France normalization (Exp 03/05), decoy weighting (Exp 06), larger K (Exp 08).

## Validation (fold B: S1 id % 5 == 0, 441,655 S1; threshold 0.70)
| | Approach 02 | **FINAL_HN_ONLY** |
|---|---:|---:|
| Macro F0.5 | 0.985690 | **0.986268** (seed 7: 0.986308) |
| Precision | 0.99361 | 0.99396 |
| Recall | 0.96685 | 0.96785 |
| False singletons | 280 | 286 |
Frozen code reproduces 0.986268 exactly (`results/verify_summary.json`).

## Full test run (`results/final_summary.json`, `results/full_run_log.txt`)
- Runtime 506 s (CPU, 16 threads; no GPU). Final model best_iteration 4016.
- Matched records 5,850,201 (Approach 02 at 0.70: 5,847,663; +18,717 / −16,179).
- Per country: India 2,726,116 · US 2,242,489 · France 881,596.
- S1 rows 1,732,544; S1 with matches 1,632,996; empty 99,548.
- Official validator: PASS (default and `--check-ids`). Independent checks: correct headers/columns, 1,732,544 unique
  S1 rows = test S1 set, no S2/S3 ID matched twice, 0 IDs outside the test set.

## Lineage
00 baseline → 01 edit features → 02 cross-fitted token odds (0.985690) → 10 house-number features (0.986268, KEEP).
Diagnostics that led here: 07 recall decomposition, 09 exact-rank / oracle ceilings, 10 signal pre-check.
