# Experiment 10 — Stage-2 house-number offset / parity features — KEEP

Branch: `experiment/10-stage2-address-signals`. Reads Approach 02 validation tables/config/code read-only; writes only this folder.

## Hypothesis (from Exp 09 + `results/diag_signals.txt`, fold-A training rows only)
In-band address-present Top-1 rows whose first house number differs: y-rate is ~0.7 when the offset is 5–50 and has
the **same parity** (same side of street) but ~0.03 when parity differs (US; India 0.54 vs 0.07). Existing features
(`e_hn_edit`, `e_hn_rel`, `f_num_rel_diff`) are relative / edit-type encodings and cannot express parity.

New features (`src/newfeats.py`, 6 columns, country-agnostic, from normalized `addr_nums`):
`hn_absd1, hn_par1` (first numbers), `hn_absd2, hn_par2` (second numbers), `hn_bestd, hn_best_par` (query's first number vs closest S1 number). -1 = unavailable.

## Protocol (`src/common.py`, `src/run.py`)
Same Approach 02 rows (validate h0/h1 train, fold-B band val), same shuffle (seed 42), same LightGBM params, 8000 rounds,
early stopping 100, same decision assembly (fold-B band rows use Stage-2, all else unchanged), threshold 0.70.
R0 = identical refit without new features (controlled baseline). R1 = R0 + 6 features.

## Results (validation fold B, threshold 0.70)
| | A02 saved | R0 refit | **R1 +hn** |
|---|---:|---:|---:|
| Macro F0.5 | 0.985690 | 0.985647 | **0.986268** |
| Precision | 0.99361 | 0.99359 | 0.99396 |
| Recall | 0.96685 | 0.96677 | 0.96785 |
| False singletons | 280 | 278 | 286 |
| Band AUC | 0.99247 | 0.99245 | 0.99402 |
| Addr-present rejected correct | 12,729 | 12,657 | 11,014 |
| Addr-present accepted wrong | ~3,806 | 3,797 | 3,526 |
| best_iteration | 4460 | 4093 | 3668 |

Refit noise (R0 vs A02 saved): −0.000043. **Gain R1 − R0 = +0.000621** (~15× noise).
Predicted range 0.98579–0.98649 (midpoint 0.98610); actual 0.986268 (+0.00017 vs midpoint).
Runtime: estimated ~12 min for R0+R1; actual 382 s + 376 s = 12.6 min (CPU, 16 threads).
`hn_bestd` and `hn_absd1` rank #4 and #5 by gain.

**Decision: KEEP.** R1 becomes the baseline for the next experiment.
