"""Exp 22: anchor consistency fix. Anchor pool restricted to records Stage-1 never trained on
(hash(q_idx, seed 42) % 1000 >= 100) in EVERY split (train, validation, test), so no anchor has an in-sample p1 and
anchor density is identical everywhere. R8 = R7 features with the new anchor features. Both seeds; plain + two proxies;
test-shift check (fold-A R7 vs R8)."""
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

import asib
import ate
import common
from common import A02, BASE_FEATURES, fit, log, stage2
from newfeats import HN_COLS, hn_features

ROOT = common.ROOT
APP = Path(__file__).resolve().parents[1]
CACHE, RES = APP / "cache", APP / "results"
E10, E11, E19, E21 = (ROOT / f"approaches/{d}/cache" for d in ("10_STAGE2_ADDRESS_SIGNALS", "11_STAGE2_ADDRESS_TOKEN_ODDS",
                                                               "19_ANCHOR_SIBLINGS", "21_CHAR_LONGER"))
sys.path.insert(0, str(ROOT / "approaches/15_DENSITY_MATCHED_EVAL/src"))
sys.path.insert(0, str(ROOT / "approaches/18_EXPECTED_F05_DECISION/src"))
F = BASE_FEATURES + HN_COLS + ate.ATE_COLS + ["char_p"] + asib.ASIB_COLS


def norm(split):
    cols = ["name_core", "addr_norm", "addr_nums"]
    q = pl.concat([pl.read_parquet(ROOT / f"cache/{split}_{s}_norm.parquet", columns=cols) for s in ("source2", "source3")]).with_row_index("q_idx")
    s = pl.read_parquet(ROOT / f"cache/{split}_source1_norm.parquet", columns=cols).with_row_index("s1_idx")
    return q, s


def oos_best(split):
    b = pl.read_parquet(ROOT / f"cache/{split}_best.parquet", columns=["q_idx", "s1_idx", "p1"])
    return b.filter((pl.col("q_idx").hash(42) % 1000) >= 100)


def asib_new(name, rows, split):
    path = CACHE / f"feat_asib_oos_{name}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    qn, _ = norm(split)
    f = pl.concat([rows.select("q_idx"), asib.asib_features(rows, oos_best(split), qn)], how="horizontal")
    f.write_parquet(path)
    log(f"oos-anchor features {name}: {f.height:,} rows; coverage {(f['asib_n'] > 0).mean():.3f}")
    return f


def main():
    t0 = time.time()
    T, V = common.load_rows()
    for D in ("T", "V"):
        pass
    T = T.join(pl.read_parquet(E10 / "feat_hn_train.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(pl.read_parquet(E11 / "feat_ate_train.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(pl.read_parquet(E21 / "char4_oof_train.parquet"), on="q_idx", how="left", maintain_order="left")
    V = V.join(pl.read_parquet(E10 / "feat_hn_val.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(pl.read_parquet(E11 / "feat_ate_val.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(pl.read_parquet(E21 / "char4_val.parquet"), on="q_idx", how="left", maintain_order="left")
    T8 = T.join(asib_new("train", T, "train"), on="q_idx", how="left", maintain_order="left")
    V8 = V.join(asib_new("val", V, "train"), on="q_idx", how="left", maintain_order="left")
    V7 = V.join(pl.read_parquet(E19 / "feat_asib_val.parquet"), on="q_idx", how="left", maintain_order="left")
    del T
    models = {}
    for seed in ("42", "7"):
        os.environ["ER_LGB_SEED"] = seed
        path = CACHE / f"model_R8_s{seed}.txt"
        models[seed] = lgb.Booster(model_file=str(path)) if path.exists() else fit(T8, F, path)
        pl.DataFrame({"q_idx": V8["q_idx"], "p_stage2": models[seed].predict(V8.select(F).to_numpy().astype(np.float32), num_threads=16)}) \
          .write_parquet(CACHE / f"val_scored_R8_s{seed}.parquet")
    del T8
    from density import DensityEval
    from ef05 import ef05_accept
    ev = DensityEval()
    res = {}
    for name, path in (("R7_s42", E21 / "val_scored_R7_s42.parquet"), ("R7_s7", E21 / "val_scored_R7_s7.parquet"),
                       ("R8_s42", CACHE / "val_scored_R8_s42.parquet"), ("R8_s7", CACHE / "val_scored_R8_s7.parquet")):
        b = ev.frame(pl.read_parquet(path))
        bb = b.with_columns(ef05_accept(b, 0.5).cast(pl.Float64).alias("p_final"))
        r = {"plain_ef05": ev.evaluate(bb, pl.lit(0.5))["f05"],
             "proxy_same_k2.4": ev.evaluate(bb, pl.lit(0.5), 2.4, "same")["f05"],
             "proxy_spread_k1.5": ev.evaluate(bb, pl.lit(0.5), 1.5, "spread")["f05"]}
        res[name] = r
        log(f"{name}: {json.dumps({k: round(v, 6) for k, v in r.items()})}")
    # test-shift: fold-A R7 (old anchors) vs R8 (oos anchors) acceptances @0.70
    X = pl.read_parquet(A02 / "cache/final_test_rows_cf1_ru0.parquet")
    qt, st = norm("test")
    p = X.select("q_idx", "s1_idx").with_row_index("_r").join(qt, on="q_idx", how="left").join(st, on="s1_idx", how="left", suffix="_s").sort("_r")
    _, _, _, best = stage2.train_rows()
    best = best.with_columns(pl.col("q_is_val").fill_null(False))
    pool = best.filter(~pl.col("q_is_val") & pl.col("oos")).select("q_idx", "s1_idx", "y")
    qn, sn = norm("train")
    pp = pool.select("q_idx", "s1_idx").with_row_index("_r").join(qn, on="q_idx", how="left").join(sn, on="s1_idx", how="left", suffix="_s") \
        .sort("_r").join(pool.with_row_index("_r").select("_r", "y"), on="_r")
    te = ate.learn(pp)
    del pp, pool, qn, sn
    X = pl.concat([X, hn_features(p), ate.apply(p, te)], how="horizontal").join(pl.read_parquet(E21 / "char4_test.parquet"), on="q_idx", how="left", maintain_order="left")
    X7 = X.join(pl.read_parquet(E19 / "feat_asib_test.parquet"), on="q_idx", how="left", maintain_order="left")
    X8 = X.join(asib_new("test", X, "test"), on="q_idx", how="left", maintain_order="left")
    m7 = lgb.Booster(model_file=str(E21 / "model_R7_s42.txt"))
    shift = {}
    for name, D7, D8 in (("val", V7, V8), ("test", X7, X8)):
        a7 = int((m7.predict(D7.select(F).to_numpy().astype(np.float32), num_threads=16) >= 0.7).sum())
        a8 = int((models["42"].predict(D8.select(F).to_numpy().astype(np.float32), num_threads=16) >= 0.7).sum())
        shift[name] = {"R7": a7, "R8": a8, "R8_vs_R7_pct": 100 * (a8 / a7 - 1),
                       "anchor_coverage_new": float((D8["asib_n"] > 0).mean()), "anchor_coverage_old": float((D7["asib_n"] > 0).mean())}
        log(f"shift {name}: {shift[name]}")
    res["shift"] = shift
    res["runtime_sec"] = round(time.time() - t0)
    json.dump(res, open(RES / "run22_summary.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
