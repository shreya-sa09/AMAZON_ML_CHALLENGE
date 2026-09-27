"""Exp 23: learned record-vs-anchor similarity. For each row pick its closest confident anchor (other record with p1>=0.9
on the same S1; max of fuzz name ratio + address ratio), then score (record, anchor) with the cross-fitted 4-epoch char
model (Exp 21 fold-A models A/B: h0 rows <- model B, h1 rows <- model A, val/test <- mean). R9 = R7 + 'anc_char'."""
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import torch
from rapidfuzz import fuzz, process

import asib
import ate
import charmodel as cm
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
F7 = BASE_FEATURES + HN_COLS + ate.ATE_COLS + ["char_p"] + asib.ASIB_COLS
F9 = F7 + ["anc_char"]


def norm(split):
    q = pl.concat([pl.read_parquet(ROOT / f"cache/{split}_{s}_norm.parquet", columns=["name_core", "addr_norm", "addr_nums"]) for s in ("source2", "source3")]).with_row_index("q_idx")
    s = pl.read_parquet(ROOT / f"cache/{split}_source1_norm.parquet", columns=["name_core", "addr_norm", "addr_nums"]).with_row_index("s1_idx")
    return q, s


def best_anchor(rows, split, qn):
    """row-aligned q2 (closest confident anchor) or null."""
    best = pl.read_parquet(ROOT / f"cache/{split}_best.parquet", columns=["q_idx", "s1_idx", "p1"])
    anc = best.filter(pl.col("p1") >= asib.ANCHOR_P).select(pl.col("q_idx").alias("q2"), "s1_idx")
    r = rows.select("q_idx", "s1_idx").with_row_index("_r")
    s = r.join(anc, on="s1_idx").filter(pl.col("q2") != pl.col("q_idx")).sort("_r", "q2")
    s = s.with_columns(pl.int_range(pl.len()).over("_r").alias("_k")).filter(pl.col("_k") < asib.CAP).drop("_k")
    s = s.join(qn.select(pl.col("q_idx").alias("q2"), pl.col("name_core").alias("nm2"), pl.col("addr_norm").alias("ad2")), on="q2") \
         .join(qn.select("q_idx", "name_core", "addr_norm"), on="q_idx")
    s = s.with_columns([pl.col(c).fill_null("") for c in ("nm2", "ad2", "name_core", "addr_norm")])
    sim = process.cpdist(s["name_core"].to_list(), s["nm2"].to_list(), scorer=fuzz.ratio, workers=-1, dtype=np.float32) + \
          process.cpdist(s["addr_norm"].to_list(), s["ad2"].to_list(), scorer=fuzz.ratio, workers=-1, dtype=np.float32)
    s = s.with_columns(pl.Series("sim", sim)).sort(["_r", "sim"], descending=[False, True]).unique("_r", keep="first")
    return r.join(s.select("_r", "q2"), on="_r", how="left").sort("_r")["q2"]


def pair_arrays(q_idx, q2, qn):
    a = pl.DataFrame({"q_idx": q_idx, "q2": q2}).with_row_index("_r")
    a = a.join(qn, on="q_idx", how="left").join(qn.rename({"q_idx": "q2"}), on="q2", how="left", suffix="_s").sort("_r")
    return {"qn": cm.encode(a["name_core"].fill_null("").str.replace_all(" ", "").to_list(), cm.LN),
            "sn": cm.encode(a["name_core_s"].fill_null("").str.replace_all(" ", "").to_list(), cm.LN),
            "qa": cm.encode(a["addr_norm"].fill_null("").to_list(), cm.LA), "sa": cm.encode(a["addr_norm_s"].fill_null("").to_list(), cm.LA)}


def load(tag):
    m = cm.CharPair().to(cm.DEV)
    m.load_state_dict(torch.load(E21 / f"char4_model_{tag}.pt", map_location=cm.DEV))
    return m


def anc_char_table(name, rows, split, half=None):
    path = CACHE / f"feat_anc_char_{name}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    qn, _ = norm(split)
    q2 = best_anchor(rows, split, qn)
    X = pair_arrays(rows["q_idx"], q2, qn)
    mA, mB = load("A"), load("B")
    if half is None:
        sc = (cm.predict(mA, X) + cm.predict(mB, X)) / 2
    else:   # h0 rows (half False) scored by model B (trained on h1), h1 rows by model A (trained on h0)
        pa, pb = cm.predict(mA, X), cm.predict(mB, X)
        sc = np.where(half, pa, pb)
    sc = np.where(q2.is_null().to_numpy(), -1.0, sc).astype(np.float32)
    f = pl.DataFrame({"q_idx": rows["q_idx"], "anc_char": sc})
    f.write_parquet(path)
    log(f"anc_char {name}: {f.height:,} rows; with anchor {(sc >= 0).mean():.3f}; mean {sc[sc >= 0].mean():.3f}")
    del mA, mB
    torch.cuda.empty_cache()
    return f


def main():
    t0 = time.time()
    h1q = pl.read_parquet(A02 / "cache/validate_train_rows_h1_cf1_ru0.parquet", columns=["q_idx"]).with_columns(pl.lit(True).alias("h1"))
    T, V = common.load_rows()
    T = T.join(pl.read_parquet(E10 / "feat_hn_train.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(pl.read_parquet(E11 / "feat_ate_train.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(pl.read_parquet(E21 / "char4_oof_train.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(pl.read_parquet(E19 / "feat_asib_train.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(h1q, on="q_idx", how="left", maintain_order="left").with_columns(pl.col("h1").fill_null(False))
    V = V.join(pl.read_parquet(E10 / "feat_hn_val.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(pl.read_parquet(E11 / "feat_ate_val.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(pl.read_parquet(E21 / "char4_val.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(pl.read_parquet(E19 / "feat_asib_val.parquet"), on="q_idx", how="left", maintain_order="left")
    T = T.join(anc_char_table("train", T, "train", half=T["h1"].to_numpy()), on="q_idx", how="left", maintain_order="left")
    V = V.join(anc_char_table("val", V, "train"), on="q_idx", how="left", maintain_order="left")
    models = {}
    for seed in ("42", "7"):
        os.environ["ER_LGB_SEED"] = seed
        path = CACHE / f"model_R9_s{seed}.txt"
        models[seed] = lgb.Booster(model_file=str(path)) if path.exists() else fit(T, F9, path)
        pl.DataFrame({"q_idx": V["q_idx"], "p_stage2": models[seed].predict(V.select(F9).to_numpy().astype(np.float32), num_threads=16)}) \
          .write_parquet(CACHE / f"val_scored_R9_s{seed}.parquet")
    del T
    from density import DensityEval
    from ef05 import ef05_accept
    ev = DensityEval()
    res = {}
    for name, path in (("R7_s42", E21 / "val_scored_R7_s42.parquet"), ("R7_s7", E21 / "val_scored_R7_s7.parquet"),
                       ("R9_s42", CACHE / "val_scored_R9_s42.parquet"), ("R9_s7", CACHE / "val_scored_R9_s7.parquet")):
        b = ev.frame(pl.read_parquet(path))
        bb = b.with_columns(ef05_accept(b, 0.5).cast(pl.Float64).alias("p_final"))
        r = {"plain_ef05": ev.evaluate(bb, pl.lit(0.5))["f05"], "proxy_same_k2.4": ev.evaluate(bb, pl.lit(0.5), 2.4, "same")["f05"],
             "proxy_spread_k1.5": ev.evaluate(bb, pl.lit(0.5), 1.5, "spread")["f05"]}
        res[name] = r
        log(f"{name}: {json.dumps({k: round(v, 6) for k, v in r.items()})}")
    # test shift: fold-A R7 vs R9 @0.70 on val band and test band
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
    X = pl.concat([X, hn_features(p), ate.apply(p, te)], how="horizontal") \
          .join(pl.read_parquet(E21 / "char4_test.parquet"), on="q_idx", how="left", maintain_order="left") \
          .join(pl.read_parquet(E19 / "feat_asib_test.parquet"), on="q_idx", how="left", maintain_order="left")
    X = X.join(anc_char_table("test", X, "test"), on="q_idx", how="left", maintain_order="left")
    m7 = lgb.Booster(model_file=str(E21 / "model_R7_s42.txt"))
    shift = {}
    for name, D in (("val", V), ("test", X)):
        a7 = int((m7.predict(D.select(F7).to_numpy().astype(np.float32), num_threads=16) >= 0.7).sum())
        a9 = int((models["42"].predict(D.select(F9).to_numpy().astype(np.float32), num_threads=16) >= 0.7).sum())
        shift[name] = {"R7": a7, "R9": a9, "R9_vs_R7_pct": 100 * (a9 / a7 - 1), "anc_char_mean": float(D.filter(pl.col("anc_char") >= 0)["anc_char"].mean())}
        log(f"shift {name}: {shift[name]}")
    res["shift"] = shift
    res["runtime_sec"] = round(time.time() - t0)
    json.dump(res, open(RES / "run23_summary.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
