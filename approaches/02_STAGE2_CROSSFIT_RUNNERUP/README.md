# 02_STAGE2_CROSSFIT_RUNNERUP — stage 2 with cross-fitted token odds (current best)

> The folder name comes from the two ideas that were tested. **Only cross-fitting is kept:** the runner-up
> context was measured, lost, and is **off** in `config/config.json` (`"runner_up": false`).

## 1. Approach
Same design as `01_STAGE2_EDIT_FEATURES`: stage-1 baseline scores, then a stage-2 LightGBM re-scores
each record's top-1 pair for p1 ∈ [0.002, 0.995], and the record is accepted if p ≥ 0.7. Two changes:

1. **Cross-fitted token odds.** In 01 the learned token odds (`e_te_*`) used one half of the held-out
   records, and stage 2 trained on the other half only. Here the odds are learned on half 1 to featurize
   half 0 and vice versa, so **stage 2 trains on both halves** (1.92M rows instead of 0.96M).
   Validation and test rows use odds learned from all training rows.
2. **Converged round budget.** With 2× the rows the model hit the 3,000-round cap. The cap is now 8,000;
   early stopping picks 4,460.

## 2. Hypothesis
01's stage 2 was data-limited: half of the eligible records were spent on the token odds only.

## 3. What changed vs 01
* `src/stage2.py`: `build_training()` (cross-fit), `second_best()` / `pair_block()` (runner-up context,
  optional), `--no-crossfit`, `--runner-up`, `--no-runner-up`, `--max-rounds=N` ablation switches.
* Artifact names carry the variant: `cf1_ru0` = cross-fit on, runner-up off; `_r8000` = round cap.
* `config/config.json`: `cross_fit_te: true`, `runner_up: false`, `max_rounds: 8000`.
* `src/edit_features.py`, `src/paths.py`: identical copies of 01.

## 4. Reused (shared cache, read-only)
`C:\Users\shrey\amazon-ml-challenge-2026\cache`:
* `train_best.parquet`, `test_best.parquet`
* `train_feats\`, `test_feats\`
* `train_scored.parquet`, `test_scored.parquet`
* normalized tables, ids and labels

Nothing from 01's folder is read.

## 5. Newly computed (approach cache: `cache\`)
Feature tables:
* `validate_train_rows_h0/h1_cf1_ru0`, `validate_val_rows_cf1_ru0`
* `final_train_rows_h0/h1_cf1_ru0`, `final_test_rows_cf1_ru0`

Models:
* `stage2_validation_model_cf1_ru0_r8000.txt`, `stage2_final_model_cf1_ru0_r8000.txt`

Scored test:
* `test_final_scored_cf1_ru0_r8000.parquet`

Ablation artifacts (`*_cf1_ru1*`, `*_cf1_ru0.txt`) are kept for reproducibility. Every step resumes from its cached file.

## 6. Files modified
None outside this folder.

## 7. Cache
* Shared (read-only): `C:\Users\shrey\amazon-ml-challenge-2026\cache`
* Approach: `C:\Users\shrey\amazon-ml-challenge-2026\approaches\02_STAGE2_CROSSFIT_RUNNERUP\cache`

## 8. Output
`C:\Users\shrey\amazon-ml-challenge-2026\approaches\02_STAGE2_CROSSFIT_RUNNERUP\output\`
(`matching_results.tsv`, `candidate_pairs.tsv`). Metrics go to `results\`.

## 9. Run command (VS Code PowerShell terminal)
```powershell
cd C:\Users\shrey\amazon-ml-challenge-2026\approaches\02_STAGE2_CROSSFIT_RUNNERUP
$env:ER_CACHE_DIR="C:\Users\shrey\amazon-ml-challenge-2026\cache"
$env:ER_OUTPUT_DIR="C:\Users\shrey\amazon-ml-challenge-2026\approaches\02_STAGE2_CROSSFIT_RUNNERUP\output"
C:\Users\shrey\.venv\Scripts\python.exe -u src\stage2.py all 2>&1 | Tee-Object results\run_all_log.txt
```

## 10. Expected runtime
* **validate:** about 1 min (its features and model are already cached; it only rescores and reports).
* **final:** about 20–30 min:
  * token odds ×3
  * features for about 2.4M training rows and about 2.6M test band rows (about 3–4 min)
  * model fit up to 8,000 rounds (about 6–8 min)
  * test scoring
  * writing the 100M-pair `candidate_pairs.tsv` (about 5–10 min)
* CPU only (the GPU is not used), peak RAM about 7–9 GB.

## 11. Evaluation command
```powershell
cd C:\Users\shrey\amazon-ml-challenge-2026
C:\Users\shrey\.venv\Scripts\python.exe utils\validate_submission.py --matching approaches\02_STAGE2_CROSSFIT_RUNNERUP\output\matching_results.tsv --candidate approaches\02_STAGE2_CROSSFIT_RUNNERUP\output\candidate_pairs.tsv --test-dir dataset\test
```
Validation metrics: `results\validation_cf1_ru0_r8000.json`, `results\validation_sweep_cf1_ru0_r8000.csv`.

## 12–14. Scores (validation fold: S1 id % 5 == 0, 441,655 S1 entities, measured, threshold 0.7)

| Run | Macro F0.5 | Precision | Recall | Singleton acc | FP singletons | India | US | vs 01 |
|---|---|---|---|---|---|---|---|---|
| 00_BASELINE | 0.978244 | 0.98860 | 0.95496 | 0.98053 | 478 | 0.97552 | 0.98007 | |
| 01_STAGE2_EDIT_FEATURES | 0.985140 | 0.99320 | 0.96609 | 0.98786 | 298 | 0.98497 | 0.98525 | |
| 02 cf1_ru0 (cross-fit, 3,000-round cap) | 0.985615 | 0.99360 | 0.96660 | | | | | +0.000475 |
| 02 cf1_ru1 (+ runner-up context) | 0.985499 | 0.99350 | 0.96650 | | | | | +0.000359 (**worse than cf1_ru0; off**) |
| **02 cf1_ru0_r8000 (kept)** | **0.985690** | **0.99361** | **0.96685** | **0.98859** | **280** | **0.98564** | **0.98573** | **+0.000550** |

* **vs baseline:** +0.007446.
* **Hard-band AUC:** 0.9925.
* **Threshold sweep (kept model):**

  | Threshold | 0.5 | 0.6 | 0.65 | 0.7 | 0.75 | 0.8 |
  |---|---|---|---|---|---|---|
  | F0.5 | 0.985908 | 0.985891 | 0.985850 | **0.985690** | 0.985446 | 0.985050 |

  0.5 would add +0.0002, but the threshold is kept at 0.7 because test has more decoys (about 40% vs 26%) and precision is the safer side.

## 15. Known limitations
* The same as 01:
  * top-1 only
  * fold-A records keep their stage-1 decision in validation
  * France is unvalidated
  * no leaderboard number yet
* Gains over 01 are small (+0.00055). The remaining gap to 0.99 is not addressed by stage-2 capacity.

## 16. Next possible experiments
* **Edit features inside the stage-1 model** (all 10 candidates, not only the top-1). This can also fix wrong
  top-1 rankings. It needs new feature tables for about 200M pairs (multiprocess, about 30–60 min) and a stage-1 retrain.
* **Records with no address:** 46% of lost links. They need name-only evidence, e.g. a name-rarity feature
  within the country.
