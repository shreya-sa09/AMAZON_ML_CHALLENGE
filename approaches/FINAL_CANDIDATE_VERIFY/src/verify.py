"""FINAL CANDIDATE — Exp10 House-Number + Exp13 Sibling Evidence: validation verification (read-only).

1. Recompute the 14 new validation features through the FROZEN FINAL code path (final_run.add_new_features) and check
   they are identical to the Exp 10 / Exp 13 caches the validation results were measured on.
2. Check the saved Exp 13 seed-42 / seed-7 models use exactly FINAL's 117-feature vector.
3. Score validation fold B with both models on the recomputed features; rebuild macro F0.5 at threshold 0.70 with the
   same decision assembly as all experiments.
Writes only approaches/FINAL_CANDIDATE_VERIFY/results/verify_summary.json. Run with PYTHONDONTWRITEBYTECODE=1.
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "approaches/FINAL_BEST_APPROACH/src"))
import final_run  # noqa: E402  (frozen FINAL code, imported read-only; main() is NOT called)
from common import Evaluator, A02  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"
E10 = ROOT / "approaches/10_STAGE2_ADDRESS_SIGNALS/cache"
E13 = ROOT / "approaches/13_STAGE2_SIBLING_EVIDENCE/cache"
KNOWN = {"approach02": 0.9856897798505745, "exp10": 0.9862679, "exp13_s42": 0.9870430, "exp13_s7": 0.9869760}
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)


def main():
    V = pl.read_parquet(A02 / "cache/validate_val_rows_cf1_ru0.parquet")
    Vf = final_run.add_new_features(V, "train")          # FINAL code path, validation rows live in the train split
    new_cols = final_run.HN_COLS + final_run.sib.SIB_COLS
    cached = V.select("q_idx").join(pl.read_parquet(E10 / "feat_hn_val.parquet"), on="q_idx", how="left", maintain_order="left") \
              .join(pl.read_parquet(E13 / "feat_sib_val.parquet"), on="q_idx", how="left", maintain_order="left")
    diffs = {c: float(np.nanmax(np.abs(Vf[c].to_numpy() - cached[c].to_numpy()))) for c in new_cols}
    identical = all(d == 0.0 for d in diffs.values())
    log(f"feature identity FINAL-code vs experiment caches: {'IDENTICAL' if identical else 'DIFFERENT'} {diffs if not identical else ''}")
    X = Vf.select(final_run.FEATS).to_numpy().astype(np.float32)
    ev = Evaluator()
    res = {}
    for tag, path in (("seed42", E13 / "model_R3_hn_sib.txt"), ("seed7", E13 / "model_R3_hn_sib_s7.txt")):
        m = lgb.Booster(model_file=str(path))
        assert m.feature_name() == final_run.FEATS, f"{tag}: model feature vector differs from FINAL"
        p = m.predict(X, num_threads=16)
        res[tag] = ev.evaluate(f"verify_{tag}", Vf, p)
    s42 = res["seed42"]["f05"]
    summary = {"name": "FINAL CANDIDATE — Exp10 House-Number + Exp13 Sibling Evidence", "threshold": final_run.THR,
               "n_features": len(final_run.FEATS), "feature_identity_vs_caches": identical, "feature_max_abs_diff": diffs,
               "results": res, "known": KNOWN,
               "gain_over_approach02": s42 - KNOWN["approach02"], "gain_over_exp10": s42 - KNOWN["exp10"],
               "gain_over_exp13_reported": s42 - KNOWN["exp13_s42"],
               "reproduces_exp13_s42": abs(s42 - KNOWN["exp13_s42"]) < 1e-6,
               "runtime_sec": round(time.time() - T0)}
    (OUT / "verify_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2, default=float))


if __name__ == "__main__":
    main()
