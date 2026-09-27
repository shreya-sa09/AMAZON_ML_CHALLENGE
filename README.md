# Amazon ML Challenge 2026 — Business Entity Resolution — FINAL_V5_CHAR8

| | |
|---|---|
| **Leaderboard (macro F0.5)** | **0.983271** (best submission; previous: FINAL_V4 0.982899, FINAL_HN_ONLY 0.981172) |
| Submitted (frozen) code | `e933bdf` (tag `submitted-v5-0.983271`); this branch is based on `620fb40` = `e933bdf` + the V5 run's own README/results |
| Solution folder | `approaches/FINAL_V5_CHAR8/` |
| Validation F0.5 (fold B, S1 id % 5 == 0) | 0.988391 (seed 42) / 0.988417 (seed 7) |
| Features | 122 (identical to V4) — the only change from V4 is the char pair model trained 8 epochs instead of 4 |
| Decision rule | expected-F0.5 subset selection per S1 on density-calibrated probabilities (odds × 0.5) |
| Test matched records | 5,826,691 |
| Official validator | PASS (default and `--check-ids`) |

## Components
- `business_entity_resolution/src/` — baseline pipeline (normalization, TF-IDF retrieval top-10, 64 pair features, Stage-1 LightGBM).
- `approaches/02_STAGE2_CROSSFIT_RUNNERUP/` — Stage-2 recipe (edit features, cross-fitted name-token odds).
- Lineage experiments: `10` house-number · `11` address-token odds · `15` density-matched evaluation · `17` char pair model · `18` expected-F0.5 rule
  · `19` confident-anchor features · `21` 4-epoch char model · `24` 8-epoch char model (models/scores used by `--verify`).
- `approaches/FINAL_HN_ONLY/`, `approaches/FINAL_V3_CHAR_EF05/` — earlier finals whose test scores are read by the test-shift gate;
  `approaches/FINAL_V4_ANCHOR_CHAR4/` — direct predecessor (V5 = V4 + 8 char-model epochs).
- `approaches/FINAL_V5_CHAR8/` — frozen final pipeline, config, run summary, run log, verify log.
- `approaches/LEADERBOARD_HISTORY.md` — all real leaderboard submissions.

## Reproduce (from the repository root, challenge data in `dataset/`)
```
python business_entity_resolution/src/run_all.py
approaches/02_STAGE2_CROSSFIT_RUNNERUP/run_experiment.bat all
# FINAL_HN_ONLY and FINAL_V3_CHAR_EF05 full runs (their cache/test_final_scored.parquet is read by the gate)
# for --verify: run lineage experiments 10, 11, 17, 19, 21, 24 (src/run*.py) to rebuild their caches/models
cd approaches/FINAL_V5_CHAR8 && set PYTHONDONTWRITEBYTECODE=1
python src/final_run.py --verify        # reproduces validation 0.988391
python src/final_run.py                 # full run -> output/matching_results.tsv, candidate_pairs.tsv (~37 min, RTX 4050)
```
Validator: `python utils/validate_submission.py --matching approaches/FINAL_V5_CHAR8/output/matching_results.tsv --candidate approaches/FINAL_V5_CHAR8/output/candidate_pairs.tsv --test-dir dataset/test --check-ids`

## Limitations
No France labels; no-address exact name-twin ties are not separable; validation ~0.005 above leaderboard (denser decoys on test).
Caches, models and TSVs are gitignored and must be regenerated. `approaches/15_DENSITY_MATCHED_EVAL/src/proxy_compare.py` references a
removed diagnostic cache (not on any production or `--verify` path). `make_submission.py` is legacy (packages only the baseline code).
