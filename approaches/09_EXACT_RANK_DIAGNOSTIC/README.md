# Experiment 09 — Exact-rank and loss-ceiling diagnostics (read-only)

Branch: `experiment/09-exact-rank-diagnostic`. Reads shared caches, Approach 02 validation tables and Experiment 07 samples; writes only `results/`. No model, candidate or feature is changed.

## 09a `src/exact_rank.py` — exact rank of the true S1 for the 2,000 Exp 07 Top-10 misses (57 s, CPU)
Exact (unsketched) production score `0.5*cos_name + 0.5*cos_addr` against every S1 of the country.

| Category | Addr missing (1,437) | Addr present (563) |
|---|---:|---:|
| A: exact Top-10 even with all ties → lost only to CountSketch | 33 | **153** |
| B: exact Top-10 only if ties break favourably | **840** | 15 |
| C/D: ≥10 S1 strictly beat the true S1 | 564 | 369 |
| True S1 tied at exact rank 1 | **774** | 10 |
| Median true cos_name / cos_addr | 0.72 / 0 | **0.11** / 0.41 |

Missing-address misses are mostly exact ties among name twins → not identifiable from the query. Address-present misses have heavily changed names; ~27% are sketch-only losses.

## 09b `src/loss_ceiling.py` — oracle F0.5 ceilings on validation fold B (22 s)
Base reconstruction asserted: 0.98568978 @0.70 (280 false singletons, 7,363,903 accepted pairs). Each row replaces ONE bucket with the oracle decision.

| Oracle bucket | All | Addr present | Addr missing |
|---|---:|---:|---:|
| Stage-2 in-band decisions perfect | +0.00671 | **+0.00512** | +0.00163 |
| Accept all correct Top-1 | +0.00484 | +0.00347 | +0.00141 |
| Reject all wrong Top-1 | +0.00215 | +0.00192 | +0.00023 |
| Perfect Top-10 re-ranking | +0.00278 | +0.00052 | +0.00227 |
| Perfect retrieval of Top-10 misses | +0.00419 | +0.00123 | +0.00299 |

Fold-B counts (addr present): 12,729 rejected correct Top-1, 4,231 accepted wrong (3,808 of them unmatched queries), 2,204 wrong Top-1 with truth in Top-10, 5,642 retrieval misses.
Feature means of address-present rejected-correct vs rejected-wrong in-band rows are nearly identical (`stage2_band_profile.csv`): the model lacks signal, not calibration.

## 09c `src/show_examples.py` — raw examples (`results/stage2_error_examples.txt`)
Rejected correct (US): house numbers off by a small offset (1209/1202, 5127/5125, 9699/9700), formatting noise (`0026220`, `##43621`), city aliases (`STT LOUIS`, `... CDP`, `... Township`); several Indic-script names (India). Accepted wrong: near-identical name + nearby number but a changed city/locality or unit.

## Conclusion
The largest *identifiable* opportunity is Stage-2 discrimination on address-present in-band rows (ceiling +0.0051), not retrieval (address-present retrieval ceiling +0.0012, of which only ~27% is sketch loss; missing-address retrieval is ambiguity-bound).
