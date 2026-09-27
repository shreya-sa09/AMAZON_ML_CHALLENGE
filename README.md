# Amazon ML Challenge 2026 — Business Entity Resolution — FINAL_HN_ONLY

| | |
|---|---|
| **Leaderboard (macro F0.5)** | **0.981172** |
| Submitted commit | `98226dc` (tag `submitted-hn-only-0.981172`) |
| Solution folder | `approaches/FINAL_HN_ONLY/` |
| Validation F0.5 (fold B, S1 id % 5 == 0) | 0.986268 (seed 42) / 0.986308 (seed 7) |
| Features | 109 (Approach 02's 103 + 6 house-number offset/parity features) |
| Decision rule | global threshold 0.70 |
| Test matched records | 5,850,201 |
| Official validator | PASS (default and `--check-ids`) |

## Components
1. `business_entity_resolution/src/` — baseline pipeline: normalization (incl. Indic transliteration dictionary), country-partitioned
   TF-IDF retrieval (name char 3-grams + address words, CountSketch approx top-40 → exact top-10), 64 pair features, Stage-1 LightGBM.
2. `approaches/02_STAGE2_CROSSFIT_RUNNERUP/` — Stage-2 LightGBM on the Stage-1 top-1 (edit-type features + cross-fitted name-token odds).
3. `approaches/10_STAGE2_ADDRESS_SIGNALS/` — house-number offset/parity features (source of the validation evidence and `--verify` cache/model).
4. `approaches/FINAL_HN_ONLY/` — frozen final pipeline (`src/final_run.py`, `config/final_config.json`, results).

## Reproduce (from the repository root, challenge data in `dataset/`)
```
python business_entity_resolution/src/run_all.py                      # shared cache/ (normalization, candidates, features, Stage 1)
approaches/02_STAGE2_CROSSFIT_RUNNERUP/run_experiment.bat all         # Approach 02 validation + final row tables
python approaches/10_STAGE2_ADDRESS_SIGNALS/src/run.py R1_hn hn       # only needed for --verify
cd approaches/FINAL_HN_ONLY && set PYTHONDONTWRITEBYTECODE=1
python src/final_run.py --verify                                      # reproduces validation 0.986268
python src/final_run.py                                               # full run -> output/matching_results.tsv, candidate_pairs.tsv
```
Validator: `python utils/validate_submission.py --matching approaches/FINAL_HN_ONLY/output/matching_results.tsv --candidate approaches/FINAL_HN_ONLY/output/candidate_pairs.tsv --test-dir dataset/test --check-ids`

## Limitations
No France labels (France is unseen in training); records without an address whose names tie exactly with other S1 records are not
separable; validation runs ~0.005 above the leaderboard (denser decoys on test). Large generated files (caches, models, TSVs) are
gitignored and must be regenerated. `make_submission.py` is legacy (packages only the baseline code).
