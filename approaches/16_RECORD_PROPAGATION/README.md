# Experiment 16 — Record-to-record propagation for no-address records + leaderboard-proxy ceilings — DISCARDED (propagation)

Branch: `experiment/16-record-propagation`. Read-only diagnostics on training ground truth.

## Hypothesis
True copies share their noisy address across sources (Exp 13). If they also shared their noisy *name*, a no-address
record could inherit the S1 of an address-bearing record with the same exact raw name, breaking name-twin ties.

## Result (`src/diag_propagation.py`, `results/diag_propagation.txt`)
| Partner key | Coverage of 337,018 no-address links | Label-free proposal accuracy | Fold-B: fixed / new wrong |
|---|---:|---:|---:|
| exact raw noisy name | 16% | 55% | 995 / 2,810 |
| normalized core name | 68% | 91% (mostly already correct) | 821 / 1,984 |
Copies do not share noisy names. **DISCARD.**

Assignment constraints: per-S1 links per source vary 0–3+; no-address copies per S1 are Poisson(≈0.16)
(0.850 / 0.139 / 0.0107 vs 0.85 / 0.138 / 0.011) → no quota to exploit.

## Leaderboard-proxy ceilings from FINAL_V2 (`src/ceiling_proxy.py`, `results/ceiling_proxy.txt`)
| Oracle | Plain | LB proxy |
|---|---:|---:|
| none (FINAL_V2) | 0.986073 | 0.982344 |
| address-present Stage-2 decisions (top-1) | 0.991129 | 0.989064 |
| + address-present top-10 ranking | 0.991596 | 0.989532 |
| + address-present retrieval | 0.992730 | 0.990666 |
| all reachable (top-10) records | 0.995485 | 0.993532 |
| everything | 0.999397 | 0.997445 |

Largest identifiable loss: address-present accept/reject quality (+0.0067 proxy). 0.99 LB requires near-perfect
address-present decisions and retrieval; the remainder is no-address name-twin ties, shown unresolvable
(exact ties Exp 09a, no name propagation, no per-source quota).
