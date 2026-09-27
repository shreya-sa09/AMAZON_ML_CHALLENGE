# Experiment 17 — Learned character-level pair model as a Stage-2 feature — KEEP

Branch: `experiment/17-char-pair-model`. MatchPyramid-style char CNN trained from scratch in PyTorch (`src/charmodel.py`):
name and address char-similarity matrices (learned + exact-match channels) -> 2D CNN -> logit. Country-agnostic,
normalized strings. Cross-fitted: model A trained on hash half h0 scores h1 + val; model B trained on h1 scores h0 + val;
val = mean(A, B). 2 epochs each on ~960k rows, batch 2048, fp16; ~3.5-5 min per model on the RTX 4050, 3.0 GB VRAM.

Char-only AUC on fold-B band rows: A 0.9621, B 0.9575, mean 0.9631 (train OOF 0.9600; stage-1 p1 0.9706).

## Stage-2 with the extra feature (116 features, identical recipe) vs R2 (115 features)
| | R2 s42 | **R5 s42** | R2 s7 | **R5 s7** |
|---|---:|---:|---:|---:|
| plain @0.70 | 0.986354 | **0.986726** | 0.986313 | **0.986766** |
| plain EF0.5 (c=0.5) | 0.986239 | 0.986630 | 0.986233 | 0.986648 |
| LB proxy EF0.5 (k=2.4) | 0.982578 | **0.983103** | 0.982553 | **0.983126** |
| LB proxy EF0.5 (k=1.9) | 0.983195 | 0.983665 | 0.983181 | 0.983684 |
Gain +0.00053 / +0.00057 proxy, +0.00037 / +0.00045 plain. Predicted +0.0002-0.0010. **KEEP.**
Model was still improving at epoch 2 (under-trained).
