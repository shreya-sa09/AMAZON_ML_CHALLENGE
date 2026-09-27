# Experiment 22 — Anchor consistency fix (out-of-sample anchors only) — DISCARDED
Also: 22a proxy recalibration on 3 real leaderboard points and a seed-ensemble check (`src/recal_ensemble.py`, `results/recal_ensemble.txt`).

## 22a
Offsets (sim − LB) for A02@0.50 / FINAL_HN_ONLY / FINAL_V4: plain 0.00679/0.00510/0.00530; spread k=2.4 (old proxy)
0.00218/0.00217/0.00405 (spread 0.0019 — worst); same k=2.4 0.00474/0.00381/0.00480 (spread 0.0010 — best).
No setting fits all three better than ±0.0005. Plain validation predicted V4's LB gain well (+0.0019 → +0.0017).
Seed ensemble of R7 s42+s7: plain 0.988231 (= seed mean) → no gain, DISCARD.

## 22 (R8): anchor pool restricted to hash(q_idx)%1000 >= 100 in every split
| | R7 s42 | R8 s42 | R7 s7 | R8 s7 |
|---|---:|---:|---:|---:|
| plain EF0.5 | 0.988200 | 0.988004 | 0.988251 | 0.987990 |
| proxy same k2.4 | 0.987696 | 0.987485 | 0.987750 | 0.987466 |
| proxy spread k1.5 | 0.987753 | 0.987545 | 0.987803 | 0.987531 |
Band acceptances vs R7: val −0.29%, test −0.81%. Predicted −0.0002..+0.0003; actual −0.00023. **DISCARD.**
