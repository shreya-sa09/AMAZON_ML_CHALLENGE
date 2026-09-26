# 05_FRANCE_NORMALIZATION — Experiment 2: France normalization only

**Change (one variable).** For France S2/S3 records only:
1. address components that name a **département** are rewritten to the **région** that every France S1 record uses
   (Nord and Pas-de-Calais → Hauts-de-France, Gironde → Nouvelle-Aquitaine, Loire-Atlantique → Pays de la Loire);
2. short digit-for-letter tokens are repaired when the result is a legal form or a common France S1 name token
   (`5arl`→`sarl`, `5as`→`sas`, `c1ub`→`club`).

Then, for France only:
* the 64 baseline pair features are recomputed
* stage 1 re-scores with the unchanged baseline model
* stage 2 re-scores with Approach 02's unchanged final model

**Baseline for comparison:** Approach 02 at threshold **0.70** (Experiment 1's `matching_results_t0.70.tsv`).
This experiment uses the same threshold (0.70). No retraining, no candidate regeneration, US/India untouched.

The code is a copy of `03_FRANCE_NORMALIZATION` (built earlier, never run). `03` itself is unchanged. The only
config difference from 03 is `threshold: 0.7`, and `src/style_check.py` is new.

## Validation (fold B, 441,655 S1, measured)
Validation contains **no France records**, and the change touches nothing else, so validation is **identical to the
baseline by construction**. It is also checked on test: every US and India output row is byte-identical to 02 @0.70.

| | F0.5 | Precision | Recall | Singleton acc | False singletons | India | US | France |
|---|---|---|---|---|---|---|---|---|
| 02 @0.70 (baseline) | 0.985690 | 0.993609 | 0.966849 | 0.988595 | 280 | 0.985637 | 0.985725 | not measurable |
| **05 @0.70** | **0.985690** | 0.993609 | 0.966849 | 0.988595 | 280 | 0.985637 | 0.985725 | not measurable |
| difference | 0.000000 | 0 | 0 | 0 | 0 | 0 | 0 | — |

**Candidate recall:** 98.69% on validation, unchanged. France candidates are identical: `candidate_pairs.tsv` is md5-identical to 02's.

## France: label-free evidence (test data)
**Changes applied:** 456,002 of 1,434,993 France addresses (31.8%) and 3,922 names.

**Address-style test.** Writing the département or the région is a formatting choice, so it should not affect the
acceptance rate. The groups are balanced: house-number presence 0.960 vs 0.962, name length 16.5 vs 16.4, S3 share 0.50 vs 0.53.

| France records | Count | Accepted before | Accepted after | Mean p before | Mean p after | Stage-1 accept before | Stage-1 accept after |
|---|---|---|---|---|---|---|---|
| wrote département | 456,002 | 0.5943 | **0.6092** | 0.6105 | 0.6283 | 0.6198 | 0.6337 |
| wrote région | 478,124 | 0.6398 | 0.6398 | 0.6563 | 0.6563 | 0.6534 | 0.6534 |
| neither | 500,867 | 0.6086 | 0.6086 | 0.6291 | 0.6290 | 0.6216 | 0.6215 |

The gap (département − région) closes, but not fully:

| | Before | After | Share of the gap closed |
|---|---|---|---|
| Acceptance | −0.0455 | −0.0306 | 33% |
| Mean p | −0.0459 | −0.0280 | 39% |
| Stage-1 acceptance | −0.0336 | −0.0196 | 42% |

**France decisions @0.70:**
* 881,748 → 888,493 accepted records (**+6,745 net**)
* 8,199 newly accepted, 1,454 no longer accepted, 15 moved to a different S1
* Stage 2 now rejects 2.5% of the records stage 1 accepts, down from 3.2%

## Runtime
369 s in total:
* records: 26 s
* re-featurizing 14.35M pairs: 240 s
* stage 1, token odds and stage 2: 20 s
* writing the outputs: 70 s
* style check: about 15 s

## Verdict
**KEEP (low risk), pending leaderboard confirmation.**
* US and India are byte-identical, and validation is unchanged.
* The France changes remove about 35–40% of a measurable, format-only penalty, and they add accepted matches mainly in the affected group.
* The remaining gap (−0.03) means other France-specific differences remain. It is not claimed as a validation gain, because there is none to measure.
* **Estimated leaderboard effect: +0.0003 to +0.001** (15% of test; about +0.9% France records accepted).
* **Clean check:** submit `04_THRESHOLD_DENSITY_AUDIT/output/matching_results_t0.70.tsv` and this folder's `output/matching_results.tsv`.
  They differ only in France.

## Files
* `src/france_fix.py` (from 03), `src/style_check.py` (new), `src/stage2.py`, `src/edit_features.py`, `src/paths.py` (copies of 02)
* `config/config.json`
* `cache/imported/`: 02's final stage-2 model and test scores (md5 verified)
* Results:
  * `results/final_summary.json`, `results/normalization_changes.json`, `results/style_check.json`
  * logs: `results/run_log.txt`, `results/style_check_log.txt`
* Output: `output/matching_results.tsv`, `output/candidate_pairs.tsv` (both pass `utils/validate_submission.py`)

## Reproduce
```powershell
cd C:\Users\shrey\amazon-ml-challenge-2026\approaches\05_FRANCE_NORMALIZATION
$env:ER_CACHE_DIR="C:\Users\shrey\amazon-ml-challenge-2026\cache"
$env:ER_OUTPUT_DIR="C:\Users\shrey\amazon-ml-challenge-2026\approaches\05_FRANCE_NORMALIZATION\output"
C:\Users\shrey\.venv\Scripts\python.exe -u src\france_fix.py     # about 6 min  (--smoke: about 25 s)
C:\Users\shrey\.venv\Scripts\python.exe -u src\style_check.py    # about 15 s
```
