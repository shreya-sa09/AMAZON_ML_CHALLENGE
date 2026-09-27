# Experiment 21 — Char pair model trained 4 epochs (vs 2) — KEEP

Branch: `experiment/21-char-longer`. Same architecture/cross-fit as Exp 17, 4 epochs. Replaces char_p in the R6 feature set (R7, 122 features).
Char-only val AUC: A 0.97565, B 0.97532, mean **0.97758** (2-epoch 0.96307; stage-1 p1 0.9706).

| | R6 s42 | **R7 s42** | R6 s7 | **R7 s7** |
|---|---:|---:|---:|---:|
| plain EF0.5 | 0.987935 | **0.988200** | 0.987955 | **0.988251** |
| LB proxy EF0.5 k=2.4 | 0.984746 | **0.985049** | 0.984741 | **0.985097** |
| LB proxy EF0.5 k=1.9 | 0.985196 | 0.985484 | 0.985211 | 0.985531 |
Gain +0.00030 / +0.00036 proxy. Predicted +0.0001..0.0004. Test-shift (fold-A, band acceptances @0.70): val +0.19%, test −0.57%.
Runtime ~45 min (an ~8-minute stall during model A epoch 4 under memory pressure: 1.1 GB free RAM, 5.9/6.1 GB VRAM).
**Decision: KEEP.**
