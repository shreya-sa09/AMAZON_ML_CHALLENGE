# Experiment 14 — Density-robust sibling evidence + test-shift attribution — sibling family DISCARDED for submission

Branch: `experiment/14-sib-density-robust`. Read-only reuse of Exp 10/13 feature caches, Approach 02 tables/models.

## Trigger
The option-B full run of FINAL_BEST_APPROACH (Exp 10 + Exp 13) accepted 6,058,285 test records vs 5,847,663 for
Approach 02 at the same threshold (+3.6%; 230,440 new / 19,818 dropped); on validation the shift was ~+0.2%.
Test has more records per S1 (5.75 vs 4.68) and sibling counts on band rows are higher (US 5.23 vs 4.11 mean).

## R4 = Exp 10 R1 + density-robust sibling features (`src/run14.py`)
Counts removed; exact-match counts turned into 0/1 flags: `sib_max_ad, sib_max_nm, sib_ad_adv, has_ad_exact, has_nm_exact, has_both_exact`.
Validation fold B @0.70: **F0.5 0.986889** (P 0.99429, R 0.96886, FS 326, AUC 0.99544) — R1 0.986268, R3 0.987043.
Predicted 0.98657–0.98697 (in range). Fit 463 s (slower than earlier ~270 s; no other process running).

## Test-shift check (fold-A models, band acceptances at 0.70)
| Model | Validation vs A02 | Test vs A02 |
|---|---:|---:|
| R1 house-number only (`src/shift_check2.py`) | ~+1% | **+0.38%** |
| R4 robust sibling | +1.41% | **+25.2%** |
| R3 full sibling | +1.72% | **+26.6%** |

The sibling family (with or without counts) shifts test acceptances ~18× more than validation; the house-number
features do not. 20 random US test rows newly accepted by R1 all look like genuine noisy copies
(`results/shift_hn_and_examples.txt`). Volume sanity: with ~3.46 true links per S1 (train), test would have ~6.0M
true links; Approach 02 recall/precision implies ~5.86M acceptances; the sibling FINAL accepted 6.06M (> estimated
true links), implying a large false-positive excess.

**Decision:** sibling evidence (Exp 13 and Exp 14) is DISCARDED for the submission: validated on fold B but not
transferable to the test distribution. Final candidate = Approach 02 + Exp 10 house-number features.
