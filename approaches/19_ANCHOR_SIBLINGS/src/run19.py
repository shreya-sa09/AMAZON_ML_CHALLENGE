"""Exp 19: R6 = V3 features (R5, 116) + 6 confident-anchor sibling features. Both seeds, plain + LB proxy (EF0.5 c=0.5),
and a test-shift check (fold-A R5 vs R6 acceptance change on test band vs on validation band). Read-only on inputs."""
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import torch

import asib
import ate
import charmodel as cm
import common
from common import A02, BASE_FEATURES, fit, log, stage2
from newfeats import HN_COLS, hn_features

ROOT = common.ROOT
APP = Path(__file__).resolve().parents[1]
CACHE, RES = APP / "cache", APP / "results"
E10, E11, E17 = (ROOT / f"approaches/{d}/cache" for d in ("10_STAGE2_ADDRESS_SIGNALS", "11_STAGE2_ADDRESS_TOKEN_ODDS", "17_CHAR_PAIR_MODEL"))
sys.path.insert(0, str(ROOT / "approaches/15_DENSITY_MATCHED_EVAL/src"))
sys.path.insert(0, str(ROOT / "approaches/18_EXPECTED_F05_DECISION/src"))
F5 = BASE_FEATURES + HN_COLS + ate.ATE_COLS + ["char_p"]
F6 = F5 + asib.ASIB_COLS


def norm(split):
    cols = ["name_core", "addr_norm", "addr_nums"]
    q = pl.concat([pl.read_parquet(ROOT / f"cache/{split}_{s}_norm.parquet", columns=cols) for s in ("source2", "source3")]).with_row_index("q_idx")
    s = pl.read_parquet(ROOT / f"cache/{split}_source1_norm.parquet", columns=cols).with_row_index("s1_idx")
    return q, s


def asib_table(name, rows, split):
    path = CACHE / f"feat_asib_{name}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    qn, _ = norm(split)
    best = pl.read_parquet(ROOT / f"cache/{split}_best.parquet", columns=["q_idx", "s1_idx", "p1"])
    f = pl.concat([rows.select("q_idx"), asib.asib_features(rows, best, qn)], how="horizontal")
    f.write_parquet(path)
    log(f"asib {name}: {f.height:,} rows; anchor coverage {(f['asib_n'] > 0).mean():.3f}")
    return f


def main():
    t0 = time.time()
    T, V = common.load_rows()
    for D, tab in ((T, "train"), (V, "val")):
        pass
    T = T.join(pl.read_parquet(E10 / "feat_hn_train.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(pl.read_parquet(E11 / "feat_ate_train.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(pl.read_parquet(E17 / "char_oof_train.parquet"), on="q_idx", how="left", maintain_order="left")
    V = V.join(pl.read_parquet(E10 / "feat_hn_val.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(pl.read_parquet(E11 / "feat_ate_val.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(pl.read_parquet(E17 / "char_val.parquet"), on="q_idx", how="left", maintain_order="left")
    T = T.join(asib_table("train", T, "train"), on="q_idx", how="left", maintain_order="left")
    V = V.join(asib_table("val", V, "train"), on="q_idx", how="left", maintain_order="left")
    models = {}
    for seed in ("42", "7"):
        os.environ["ER_LGB_SEED"] = seed
        path = CACHE / f"model_R6_s{seed}.txt"
        models[seed] = lgb.Booster(model_file=str(path)) if path.exists() else fit(T, F6, path)
        pv = models[seed].predict(V.select(F6).to_numpy().astype(np.float32), num_threads=16)
        pl.DataFrame({"q_idx": V["q_idx"], "p_stage2": pv}).write_parquet(CACHE / f"val_scored_R6_s{seed}.parquet")
    del T
    from density import DensityEval
    from ef05 import ef05_accept
    ev = DensityEval()
    res = {}
    for name, path in (("R5_s42", E17 / "val_scored_R5_char.parquet"), ("R5_s7", E17 / "val_scored_R5_char_s7.parquet"),
                       ("R6_s42", CACHE / "val_scored_R6_s42.parquet"), ("R6_s7", CACHE / "val_scored_R6_s7.parquet")):
        b = ev.frame(pl.read_parquet(path))
        bb = b.with_columns(pl.when(ef05_accept(b, 0.5)).then(pl.lit(1.0)).otherwise(pl.lit(0.0)).alias("p_final"))
        r = {"plain_u0.7": ev.evaluate(b, pl.lit(0.7))["f05"], "plain_ef05": ev.evaluate(bb, pl.lit(0.5))["f05"],
             "proxy_ef05": ev.evaluate(bb, pl.lit(0.5), 2.4, "spread")["f05"] - 0.0019,
             "proxy_ef05_k1.9": ev.evaluate(bb, pl.lit(0.5), 1.9, "spread")["f05"] - 0.0019}
        res[name] = r
        log(f"{name}: {json.dumps({k: round(v, 6) for k, v in r.items()})}")

    # ---- test-shift check: fold-A R5 vs R6 on the test band (features built like validation) ----
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
    Xa = {"qn": cm.encode(p["name_core"].fill_null("").str.replace_all(" ", "").to_list(), cm.LN),
          "sn": cm.encode(p["name_core_s"].fill_null("").str.replace_all(" ", "").to_list(), cm.LN),
          "qa": cm.encode(p["addr_norm"].fill_null("").to_list(), cm.LA), "sa": cm.encode(p["addr_norm_s"].fill_null("").to_list(), cm.LA)}
    cps = []
    for tag in ("A", "B"):
        m = cm.CharPair().to(cm.DEV)
        m.load_state_dict(torch.load(E17 / f"char_model_{tag}.pt", map_location=cm.DEV))
        cps.append(cm.predict(m, Xa))
        del m
    X = pl.concat([X, hn_features(p), ate.apply(p, te), pl.DataFrame({"char_p": ((cps[0] + cps[1]) / 2).astype(np.float32)})], how="horizontal")
    X = X.join(asib_table("test", X, "test"), on="q_idx", how="left", maintain_order="left")
    cty = pl.concat([pl.read_parquet(ROOT / f"cache/test_{s}_norm.parquet", columns=["country"]) for s in ("source2", "source3")]).with_row_index("q_idx")
    ctv = pl.concat([pl.read_parquet(ROOT / f"cache/train_{s}_norm.parquet", columns=["country"]) for s in ("source2", "source3")]).with_row_index("q_idx")
    m5 = lgb.Booster(model_file=str(E17 / "model_R5_char.txt"))
    shift = {}
    for name, D, c in (("val", V, ctv), ("test", X, cty)):
        a5 = m5.predict(D.select(F5).to_numpy().astype(np.float32), num_threads=16) >= 0.7
        a6 = models["42"].predict(D.select(F6).to_numpy().astype(np.float32), num_threads=16) >= 0.7
        g = D.select("q_idx").join(c, on="q_idx", how="left", maintain_order="left").with_columns(pl.Series("a5", a5), pl.Series("a6", a6)) \
             .group_by("country").agg(pl.col("a5").sum(), pl.col("a6").sum()).sort("country")
        shift[name] = {"per_country": g.to_dicts(), "total_R5": int(a5.sum()), "total_R6": int(a6.sum()),
                       "R6_vs_R5_pct": 100 * (a6.sum() / a5.sum() - 1),
                       "anchor_coverage": float((D["asib_n"] > 0).mean()), "has_ad_exact_rate": float((D["asib_has_ad_exact"] > 0).mean())}
        log(f"shift {name}: {json.dumps(shift[name])}")
    res["shift"] = shift
    res["runtime_sec"] = round(time.time() - t0)
    json.dump(res, open(RES / "run19_summary.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
