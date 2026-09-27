# Amazon ML Challenge 2026 — Business Entity Resolution — FINAL_V4_ANCHOR_CHAR4

| | |
|---|---|
| **Leaderboard (macro F0.5)** | **0.982899** (previous: FINAL_HN_ONLY 0.981172) |
| Submitted commit | `6f66b7e` (tag `submitted-v4-0.982899`; code frozen at `d32e861`) |
| Solution folder | `approaches/FINAL_V4_ANCHOR_CHAR4/` |
| Validation F0.5 (fold B, S1 id % 5 == 0) | 0.988200 (seed 42) / 0.988251 (seed 7) |
| Features | 122 = Approach 02's 103 + house-number 6 + address-token odds 6 + char pair model score 1 + confident-anchor features 6 |
| Decision rule | expected-F0.5 subset selection per S1 on density-calibrated probabilities (odds × 0.5) |
| Test matched records | 5,830,482 |
| Official validator | PASS (default and `--check-ids`) |

## Components
- `business_entity_resolution/src/` — baseline pipeline (normalization, TF-IDF retrieval top-10, 64 pair features, Stage-1 LightGBM).
- `approaches/02_STAGE2_CROSSFIT_RUNNERUP/` — Stage-2 recipe (edit features, cross-fitted name-token odds); imported by all finals.
- Lineage experiments (code + write-ups): `10` house-number features · `11` address-token odds · `15` density-matched evaluation (used by `--verify`)
  · `17` char pair model · `18` expected-F0.5 rule · `19` confident-anchor features · `21` 4-epoch char model (models used by `--verify`).
- `approaches/FINAL_HN_ONLY/`, `approaches/FINAL_V3_CHAR_EF05/` — earlier finals; their test scores are read by V4's test-shift gate.
- `approaches/FINAL_V4_ANCHOR_CHAR4/` — frozen final pipeline, config, results and `results/leaderboard_result.json`.
- `approaches/LEADERBOARD_HISTORY.md` — all real leaderboard submissions.

## Reproduce (from the repository root, challenge data in `dataset/`)
```
python business_entity_resolution/src/run_all.py
approaches/02_STAGE2_CROSSFIT_RUNNERUP/run_experiment.bat all
# FINAL_HN_ONLY and FINAL_V3_CHAR_EF05 full runs (their cache/test_final_scored.parquet is read by the V4 gate)
# for --verify: run the lineage experiments 10, 11, 17, 19, 21 (src/run*.py) to rebuild their caches/models
cd approaches/FINAL_V4_ANCHOR_CHAR4 && set PYTHONDONTWRITEBYTECODE=1
python src/final_run.py --verify        # reproduces validation 0.988200
python src/final_run.py                 # full run -> output/matching_results.tsv, candidate_pairs.tsv (~27 min, RTX 4050)
```
Validator: `python utils/validate_submission.py --matching approaches/FINAL_V4_ANCHOR_CHAR4/output/matching_results.tsv --candidate approaches/FINAL_V4_ANCHOR_CHAR4/output/candidate_pairs.tsv --test-dir dataset/test --check-ids`

## Limitations
No France labels; no-address exact name-twin ties are not separable; validation ~0.005 above leaderboard. Caches, models and TSVs are
gitignored and must be regenerated. `approaches/15_DENSITY_MATCHED_EVAL/src/proxy_compare.py` references a removed diagnostic cache
(not used by any production or `--verify` path). `make_submission.py` is legacy (packages only the baseline code).
