# 04_THRESHOLD_DENSITY_AUDIT — Experiment 1: threshold and decoy-density audit

**Question.** Approach 02 was submitted at threshold 0.5 (leaderboard 0.977984). Was 0.5 right? Does a
test-like decoy density support a different threshold?

**Method.** No retraining: Approach 02's probabilities are reused, copied to `cache/imported/` (md5 verified).
* **Validation:** fold B (441,655 S1).
  * Fold-B records use 02's stage-2 probability inside the band [0.002, 0.995] and stage-1 p1 outside it, as on test.
  * Fold-A records whose top-1 is a fold-B S1 have no out-of-fold stage-2 score, so there are **three treatments**:
    * `consistent`: stage-1 p1 at the same threshold (pessimistic)
    * `fixed_0.7`: what the earlier 02 sweep did
    * `excluded`: their false links removed (optimistic)
* **Decoy density:** every accepted decoy link is counted k times.
  * Test is estimated at **1.87× (US), 1.94× (India) and 1.67× (France)** train's decoys per S1:
    5.8 records per S1 vs 4.68, and an inverted acceptance rate.
  * Simulation checks all pass: k = 1 reproduces the metric exactly, one row is added per accepted decoy,
    per-entity true positives are identical, and predicted counts never decrease.
  * `x2_spread` puts the copies on random S1 entities instead (a sensitivity check).
* **Uncertainty:** paired bootstrap over S1 entities (300 resamples).

## Results (validation, measured; `results/threshold_grid.csv`, `results/threshold_grid_extra_density.csv`, `results/bootstrap_deltas.csv`)

`consistent` treatment, train density:

| Threshold | F0.5 | Precision | Recall | Singleton acc | False singletons | Predicted pairs | Empty S1 | India | US |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 0.984778 | 0.991426 | 0.969850 | 0.979145 | 512 | 1,492,893 | 25,047 | 0.984677 | 0.984845 |
| 0.60 | 0.985480 | 0.992782 | 0.968467 | 0.985255 | 362 | 1,487,895 | 25,316 | 0.985408 | 0.985529 |
| 0.65 | 0.985655 | 0.993259 | 0.967734 | 0.987169 | 315 | 1,485,735 | 25,403 | 0.985588 | 0.985701 |
| **0.70** | **0.985690** | 0.993609 | 0.966849 | 0.988595 | 280 | 1,483,478 | 25,508 | 0.985637 | 0.985725 |
| 0.75 | 0.985599 | 0.993885 | 0.965754 | 0.990020 | 245 | 1,480,860 | 25,634 | 0.985603 | 0.985596 |
| 0.80 | 0.985363 | 0.994118 | 0.964307 | 0.991568 | 207 | 1,477,734 | 25,786 | 0.985431 | 0.985318 |

F0.5 by treatment and density (best value in each column in bold):

| Threshold | consistent ×1 | fixed_0.7 ×1 | excluded ×1 | consistent ×2 | fixed_0.7 ×2 | excluded ×2 | consistent ×3 | fixed_0.7 ×3 | excluded ×3 |
|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 0.984778 | **0.985908** | **0.986805** | 0.983179 | 0.984340 | 0.985273 | 0.982037 | 0.983218 | 0.984176 |
| 0.60 | 0.985480 | 0.985891 | 0.986791 | 0.984147 | 0.984573 | 0.985510 | 0.983196 | 0.983632 | 0.984594 |
| 0.65 | 0.985655 | 0.985850 | 0.986751 | 0.984447 | **0.984647** | **0.985586** | 0.983583 | 0.983788 | 0.984752 |
| 0.70 | **0.985690** | 0.985690 | 0.986590 | 0.984604 | 0.984604 | 0.985541 | 0.983828 | **0.983828** | **0.984791** |
| 0.75 | 0.985599 | 0.985446 | 0.986347 | **0.984635** | 0.984473 | 0.985411 | **0.983945** | 0.983777 | 0.984741 |
| 0.80 | 0.985363 | 0.985050 | 0.985952 | 0.984531 | 0.984205 | 0.985144 | 0.983936 | 0.983599 | 0.984564 |

Bootstrap, `consistent` treatment: the gain of **0.70 over 0.50** is +0.000912 [+0.00081, +0.00103] at ×1,
+0.001425 [+0.00131, +0.00155] at ×2, and +0.001791 [+0.00167, +0.00192] at ×3.
**0.75 vs 0.70** is −0.000091 at ×1, +0.000031 [−0.000027, +0.000090] at ×2 (not significant), and +0.000118 at ×3.

## Conclusion
* **0.5 was the wrong threshold for test.**
  * At train density it looks fine only under the optimistic fold-A treatments (+0.0002 over 0.7).
  * At the estimated test density (×2) it is the **worst** option under every treatment: −0.0003 to −0.0014.
* **0.75 is not reliably better than 0.70.** It wins only under the pessimistic treatment at ×2/×3, by amounts inside the bootstrap noise.
* **0.70 is the robust choice.** It is within 0.00005 of the best threshold in every one of the 9 settings.
* **Expected leaderboard effect** of 0.70 vs the submitted 0.50: about **+0.0003 to +0.0014**. That is an estimate from a simulation, not a leaderboard measurement.
* **Verdict: KEEP 0.70** as the threshold for the next experiments. A resubmission of `output/matching_results_t0.70.tsv` would confirm it.

## Test files (from 02's probabilities; `candidate_pairs.tsv` is identical to 02's, so it is not rewritten)

| Threshold | Matched records | S1 with a match | Empty S1 | US | India | France |
|---|---|---|---|---|---|---|
| 0.50 (= 02 submission, md5 identical) | 5,901,733 | 1,635,114 | 97,430 | 2,256,097 | 2,748,860 | 896,776 |
| 0.60 | 5,873,333 | 1,633,959 | 98,585 | 2,248,061 | 2,736,484 | 888,788 |
| 0.65 | 5,860,476 | 1,633,408 | 99,136 | 2,244,664 | 2,730,660 | 885,152 |
| **0.70** | 5,847,663 | 1,632,933 | 99,611 | 2,241,073 | 2,724,842 | 881,748 |
| 0.75 | 5,833,442 | 1,632,394 | 100,150 | 2,236,853 | 2,718,634 | 877,955 |
| 0.80 | 5,816,816 | 1,631,735 | 100,809 | 2,231,390 | 2,711,657 | 873,769 |

`matching_results_t0.70.tsv` passes `utils/validate_submission.py` (checked against 02's `candidate_pairs.tsv`).

## Limitations
* Fold-A records have no out-of-fold stage-2 score, so the truth lies between the pessimistic and
  optimistic treatments. Experiment 4 (OOF stage 2) removes this ambiguity.
* The decoy-density factor is estimated, not observed. Test's decoys may also be harder than train's.
* France is not in validation.

## Reproduce
```powershell
cd C:\Users\shrey\amazon-ml-challenge-2026\approaches\04_THRESHOLD_DENSITY_AUDIT
C:\Users\shrey\.venv\Scripts\python.exe -u src\audit.py          # about 1 min (--smoke: 20k-S1 slice)
C:\Users\shrey\.venv\Scripts\python.exe -u src\extra_density.py  # about 30 s
```
Reads the shared cache (read-only) and `cache/imported/`. Writes only to `results/` and `output/`.
