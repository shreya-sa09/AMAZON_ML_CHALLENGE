# 06_STAGE2_DENSITY_AWARE — Experiment 3: density-aware stage-2 training

**One change vs Approach 02.** In stage-2 training (and early stopping), every **decoy row** (a record with no true
S1) gets sample weight **2.0**, to mimic test's roughly 2× decoys per S1 (estimated in Experiment 1). Everything else is
02's:
* the cached validation feature tables, read-only from `02_STAGE2_CROSSFIT_RUNNERUP/cache/`
* the same rows in the same shuffle order
* the same parameters, round cap and early stopping
* threshold 0.70

No France fix and no other change.

**Evaluation.** Fold B, the same assembly as Experiment 1:
* `consistent`: fold-A records use stage-1 p1 at the same threshold (primary)
* `excluded`: fold-A records dropped (optimistic)
* `×2`: the simulated test decoy density

## Controlled smoke validation (measured)
Same 200,000-row subset (180k training / 20k early stopping), 300 rounds, same order. Only the weight differs.
Stage-2 training rows (full set): 1,921,682 = 750,038 positives, **1,065,854 decoy rows**, 105,790 wrong-S1 negatives.

| Model @0.70 | consistent ×1 | consistent ×2 | excluded ×1 | Hard-band AUC | Band log-loss | Decoy rows accepted | True rows accepted |
|---|---|---|---|---|---|---|---|
| weight 1 (02 recipe) | **0.983816** | **0.982429** | **0.984707** | 0.988191 | 0.13298 | 4,713 | 185,552 |
| weight 2 (density-aware) | 0.983316 | 0.982241 | 0.984206 | 0.988109 | 0.13646 | 3,550 | 181,881 |
| delta | −0.000500 | −0.000188 | −0.000501 | −0.00008 | +0.0035 | −1,163 | −3,671 |

The AUC is unchanged, so **ranking does not improve**. The weight only shifts probabilities down: for every decoy
removed it also removes about 3 true matches. Weight 2 changes the odds by ×2, so weight 2 at threshold t
behaves like weight 1 at a higher threshold. Comparing at odds-equivalent thresholds:

| | consistent ×1 | consistent ×2 | excluded ×1 |
|---|---|---|---|
| weight 1 @0.538 | 0.983191 | 0.981226 | 0.984882 |
| weight 2 @0.538 (≈ weight 1 @0.70 in odds) | 0.983273 | 0.981714 | **0.984969** |
| weight 1 @0.70 | **0.983816** | **0.982429** | 0.984707 |
| weight 2 @0.70 (≈ weight 1 @0.824) | 0.983316 | 0.982241 | 0.984206 |
| weight 1 @0.824 | 0.982864 | 0.981933 | 0.983367 |
| weight 2 @0.824 | 0.981540 | 0.980855 | 0.982043 |

At matched odds the two models are within about ±0.0007 with no consistent winner. The best cell for the primary
metric, both at train density and at ×2, is **weight 1 at 0.70**, which is the existing Approach 02 setup.
Density-aware weighting is therefore **equivalent to moving the threshold**, and Experiment 1 already optimised that.

## Verdict: DISCARD
* No ranking gain (AUC −0.00008), and worse at the fixed 0.70 threshold on every metric (−0.0002 to −0.0005 in smoke).
* Following the protocol, the full validation run was **not** started, because the controlled smoke run was not promising.
* Baseline for reference (02 @0.70, full validation, measured in Experiment 1):
  * F0.5 0.985690, precision 0.993609, recall 0.966849
  * singleton accuracy 0.988595, 280 false singletons, 1,483,478 predicted pairs, 25,508 empty S1
  * India 0.985637, US 0.985725
  * candidate recall 98.69% (unchanged by this experiment)
* Runtime: 34 s (2 × 5 s fits on the smoke subset, plus evaluation).

## Files
* `src/density_train.py` (new), `src/stage2.py` / `src/edit_features.py` / `src/paths.py` (02 copies, unchanged)
* `config/config.json`
* `results/smoke_validation.json` (all numbers above); `results/smoke_validation_grid.csv` is local only (`*.csv` is git-ignored)

## Reproduce
```powershell
cd C:\Users\shrey\amazon-ml-challenge-2026\approaches\06_STAGE2_DENSITY_AWARE
C:\Users\shrey\.venv\Scripts\python.exe -u src\density_train.py --smoke   # about 35 s
C:\Users\shrey\.venv\Scripts\python.exe -u src\density_train.py           # full run (not executed): about 10 min
```
