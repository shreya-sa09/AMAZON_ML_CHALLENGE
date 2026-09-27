"""Experiment 14: density-robust sibling evidence.

Exp 13's count features (sib_n, sib_n_other_src) shift with record density: test has ~5.3 siblings per band row vs
~4.2 on validation, and the full FINAL run accepted +3.6% more test records than Approach 02 (validation: ~+0.2%).
R4 = Exp 10 R1 + 6 density-robust sibling features derived from the cached Exp 13 features (no recomputation):
  sib_max_ad, sib_max_nm, sib_ad_adv, has_ad_exact, has_nm_exact, has_both_exact  (exact counts -> 0/1 flags)
Then a TEST-SHIFT CHECK with fold-A models only (no final retrain): acceptance counts at 0.70 on the test band rows and
on the validation band rows for Approach 02, R3 (Exp 13) and R4.
usage: python src/run14.py      (run with PYTHONDONTWRITEBYTECODE=1)
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

import common
from common import A02, BASE_FEATURES, Evaluator, fit, load_rows, log
from newfeats import HN_COLS

APP = Path(__file__).resolve().parents[1]
CACHE, RES = APP / "cache", APP / "results"
E10 = common.ROOT / "approaches/10_STAGE2_ADDRESS_SIGNALS/cache"
E13 = common.ROOT / "approaches/13_STAGE2_SIBLING_EVIDENCE/cache"
ROBUST = ["sib_max_ad", "sib_max_nm", "sib_ad_adv", "has_ad_exact", "has_nm_exact", "has_both_exact"]
FEATS4 = BASE_FEATURES + HN_COLS + ROBUST
THR = 0.70


def robust(df):
    return df.with_columns((pl.col("sib_n_ad_exact") > 0).cast(pl.Float32).alias("has_ad_exact"),
                           (pl.col("sib_n_nm_exact") > 0).cast(pl.Float32).alias("has_nm_exact"),
                           (pl.col("sib_n_both_exact") > 0).cast(pl.Float32).alias("has_both_exact"))


def with_cached(rows, table):
    return robust(rows.join(pl.read_parquet(E10 / f"feat_hn_{table}.parquet"), on="q_idx", how="left", maintain_order="left")
                  .join(pl.read_parquet(E13 / f"feat_sib_{table}.parquet"), on="q_idx", how="left", maintain_order="left"))


def main():
    t0 = time.time()
    T, V = load_rows()
    T, V = with_cached(T, "train"), with_cached(V, "val")
    log(f"R4: {len(FEATS4)} features")
    m4 = fit(T, FEATS4, CACHE / "model_R4_hn_sibrobust.txt")
    del T
    p4v = m4.predict(V.select(FEATS4).to_numpy().astype(np.float32), num_threads=16)
    pl.DataFrame({"q_idx": V["q_idx"], "p_stage2": p4v}).write_parquet(CACHE / "val_scored_R4.parquet")
    res = Evaluator().evaluate("R4_hn_sibrobust", V, p4v)
    res["runtime_fit_eval_sec"] = round(time.time() - t0)

    # ---- test-shift check with fold-A models (A02 validation model, R3, R4) ----
    sys.path.insert(0, str(common.ROOT / "approaches/FINAL_BEST_APPROACH/src"))
    import final_run  # frozen FINAL code, read-only (feature builder only)
    X = final_run.add_new_features(pl.read_parquet(A02 / "cache/final_test_rows_cf1_ru0.parquet"), "test")
    X = robust(X)
    cty = pl.concat([pl.read_parquet(common.ROOT / f"cache/test_{s}.parquet", columns=["country"]) for s in ("source2", "source3")]) \
            .with_row_index("q_idx")
    ctv = pl.concat([pl.read_parquet(common.ROOT / f"cache/train_{s}.parquet", columns=["country"]) for s in ("source2", "source3")]) \
            .with_row_index("q_idx")
    X = X.join(cty, on="q_idx", how="left", maintain_order="left")
    V = V.join(ctv, on="q_idx", how="left", maintain_order="left")
    models = {"A02": (lgb.Booster(model_file=str(A02 / "cache/stage2_validation_model_cf1_ru0_r8000.txt")), BASE_FEATURES),
              "R3": (lgb.Booster(model_file=str(E13 / "model_R3_hn_sib.txt")), final_run.FEATS),
              "R4": (m4, FEATS4)}
    shift = {}
    for name, D in (("val", V), ("test", X)):
        acc = {}
        for k, (m, f) in models.items():
            p = m.predict(D.select(f).to_numpy().astype(np.float32), num_threads=16)
            acc[k] = D.select("country").with_columns(pl.Series("a", p >= THR)).group_by("country").agg(pl.col("a").sum()) \
                      .sort("country").to_dicts()
        shift[name] = acc
        log(f"{name} band acceptances: {json.dumps(acc)}")
    summ = {}
    for name in ("val", "test"):
        tot = {k: sum(r["a"] for r in v) for k, v in shift[name].items()}
        summ[name] = {"totals": tot, "R3_vs_A02_pct": 100 * (tot["R3"] / tot["A02"] - 1), "R4_vs_A02_pct": 100 * (tot["R4"] / tot["A02"] - 1)}
    res["shift_check"] = {"per_country": shift, "summary": summ}
    res["runtime_sec"] = round(time.time() - t0)
    with open(RES / "runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(res) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "shift_check"}, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
