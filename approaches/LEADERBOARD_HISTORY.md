# Leaderboard history (actual Amazon ML Challenge 2026 submissions)

Only scores actually returned by the leaderboard are listed. Validation = fold B (S1 id % 5 == 0).
Proxy = Exp 15 density-matched estimate (spread decoys k=2.4, minus 0.0019) — an estimate, not a score.

| Submission | Approach folder | Validation F0.5 | Proxy | **Leaderboard F0.5** | Δ vs previous best |
|---|---|---:|---:|---:|---:|
| Approach 02 @0.50 | `02_STAGE2_CROSSFIT_RUNNERUP/output` | 0.985908 | ≈0.9779 | 0.977984 | — |
| FINAL_HN_ONLY @0.70 | `FINAL_HN_ONLY/output` | 0.986268 | 0.981442 | 0.981172 | +0.003188 |
| **FINAL_V4_ANCHOR_CHAR4** | `FINAL_V4_ANCHOR_CHAR4/output` | **0.988200** / 0.988251 | 0.985049 / 0.985097 | **0.982899** | **+0.001727** |

**Current real baseline: 0.982899 (FINAL_V4_ANCHOR_CHAR4).**

Notes
- The proxy matched the first two submissions within 0.0003 but overstated FINAL_V4's gain ~2×; the anchor-sibling
  features (Exp 19, test shift 2.4× validation, France +7.4%) are the prime suspect. Recalibrate before trusting
  proxy estimates of features that depend on other records.
- Not submitted: FINAL_V2_HN_ATE_POLICY (proxy 0.982344), FINAL_V3_CHAR_EF05 (proxy 0.983103; would isolate the
  anchor-feature effect), FINAL_BEST_APPROACH / FINAL_CANDIDATE_VERIFY (sibling features, not recommended).
