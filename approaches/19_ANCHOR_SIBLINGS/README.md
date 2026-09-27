# Experiment 19 — Confident-anchor sibling features — KEEP (strongest single gain)

Branch: `experiment/19-anchor-siblings`. Anchors of a (query q, S1 x) row = OTHER records whose Stage-1 top-1 is x with
p1 >= 0.9 (`src/asib.py`). Decoys are rarely confident, so unlike Exp 13 (all siblings, discarded for a +25% test shift)
the features should not scale with decoy density. Features: asib_n, asib_max_ad, asib_has_ad_exact, asib_max_nm,
asib_has_nm_exact, asib_hn_eq (similarity capped at 30 anchors). Validation anchors of fold-B S1 are fold-B records
scored by a stage-1 model never trained on fold B; test stage-1 scores are all out-of-sample.

## Results (R6 = V3/R5 116 features + 6 = 122; identical recipe)
| | R5 s42 | **R6 s42** | R5 s7 | **R6 s7** |
|---|---:|---:|---:|---:|
| plain @0.70 | 0.986726 | 0.987993 | 0.986766 | 0.987987 |
| plain EF0.5 | 0.986630 | **0.987935** | 0.986648 | **0.987955** |
| LB proxy EF0.5 k=2.4 | 0.983103 | **0.984746** | 0.983126 | **0.984741** |
| LB proxy EF0.5 k=1.9 | 0.983665 | 0.985196 | 0.983684 | 0.985211 |
best_iteration 1853 / 2152 (R5: 2823 / 2801). Predicted +0.0003..0.0009 proxy; actual +0.0016.

## Test-shift gate (fold-A models, band acceptances @0.70)
Validation R6 vs R5 +1.58% (India +1.07%, US +2.1%); test +3.81% (India +2.1%, US +5.3%, France +7.4%) → 2.4× (gate 3×;
Exp 13 was 18×). Anchor coverage val 0.936 / test 0.950; exact-address-anchor rate val 11.1% / test 8.3%.
**Decision: KEEP** (UNCERTAIN on France, unmeasurable).
