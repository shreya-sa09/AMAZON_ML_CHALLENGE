# Experiment 12 — Raw formatting signals — DISCARDED at the diagnostic stage (no model run)

Branch: `experiment/12-stage2-raw-style`. Read-only diagnostics.

## Hypothesis
Normalization discards raw formatting (case, component order, commas, unit markers, long vs abbreviated suffix,
legal suffix). If true copies and decoys are generated through different formatting paths, these could separate the
ambiguous address-present groups (house-number offset 0 and 1–2).

## Pre-check (`src/diag_raw_style.py`, fold-A training rows h0 only, 860,275 address-present band rows, ~3 s)
Y-rate differences within US `d=1-2` (base ≈0.20): uppercase address 0.203/0.208, component order 0.196/0.208,
long suffix 0.203/0.208, comma difference 0.20–0.22 → no separation. Weak: query unit marker 0.215/0.13 (7.8k rows),
legal-suffix agreement 0.137/0.247 (partly represented by `full_ratio` on `name_norm`). Raw name exactly equal:
0.965 but only 988 rows. Full tables: `results/diag_raw_style.txt`.

**Decision: DISCARD** without a model run — expected F0.5 gain below the measured seed noise (~4e-5); saves ~6 min.

## Residual error analysis of the current best model (Exp 10 R1) — `src/show_r1_errors.py`
Fold-B address-present band errors by house-number group (`results/r1_error_examples.txt`):

| | d=0 | d=1–2 | d>2 | no number |
|---|---:|---:|---:|---:|
| Rejected correct | 1,378 | 4,101 | 4,700 | 835 |
| Accepted wrong | 736 | 499 | 2,032 | 259 |

Examples show pair-level look-alikes with opposite labels (gibberish renamed names at identical addresses; offsets 1–2
with identical names). Next hypothesis (Experiment 13): collective evidence from other records sharing the same Top-1 S1.
