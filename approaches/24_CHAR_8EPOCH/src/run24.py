"""Exp 24: char model trained 8 epochs (vs 4 in V4/R7). R10 = R7 feature set with char_p from 8-epoch models.
Test arrays are encoded only after training (memory). Both seeds; plain + proxies; test-shift vs R7 (fold-A)."""
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score

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
F = BASE_FEATURES + HN_COLS + ate.ATE_COLS + ["char_p"] + asib.ASIB_COLS
EPOCHS = 8


def norm(split):
    cols = ["name_core", "addr_norm", "addr_nums"]
    q = pl.concat([pl.read_parquet(ROOT / f"cache/{split}_{s}_norm.parquet", columns=cols) for s in ("source2", "source3")]).with_row_index("q_idx")
    s = pl.read_parquet(ROOT / f"cache/{split}_source1_norm.parquet", columns=cols).with_row_index("s1_idx")
    return q, s


def arrays(rows, qn, sn):
    p = rows.select("q_idx", "s1_idx").with_row_index("_r").join(qn, on="q_idx", how="left").join(sn, on="s1_idx", how="left", suffix="_s").sort("_r")
    return p, {"qn": cm.encode(p["name_core"].fill_null("").str.replace_all(" ", "").to_list(), cm.LN),
               "sn": cm.encode(p["name_core_s"].fill_null("").str.replace_all(" ", "").to_list(), cm.LN),
               "qa": cm.encode(p["addr_norm"].fill_null("").to_list(), cm.LA), "sa": cm.encode(p["addr_norm_s"].fill_null("").to_list(), cm.LA)}


def main():
    t0 = time.time()
    h0 = pl.read_parquet(A02 / "cache/validate_train_rows_h0_cf1_ru0.parquet", columns=["q_idx", "s1_idx", "y"])
    h1 = pl.read_parquet(A02 / "cache/validate_train_rows_h1_cf1_ru0.parquet", columns=["q_idx", "s1_idx", "y"])
    V0 = pl.read_parquet(A02 / "cache/validate_val_rows_cf1_ru0.parquet", columns=["q_idx", "s1_idx", "y"])
    if (CACHE / "char8_val.parquet").exists():
        oof, val = pl.read_parquet(CACHE / "char8_oof_train.parquet"), pl.read_parquet(CACHE / "char8_val.parquet")
    else:
        qn, sn = norm("train")
        _, A0 = arrays(h0, qn, sn)
        _, A1 = arrays(h1, qn, sn)
        _, AV = arrays(V0, qn, sn)
        del qn, sn
        out = {}
        for tag, Xtr, ytr, Xoth in (("A", A0, h0["y"].to_numpy(), A1), ("B", A1, h1["y"].to_numpy(), A0)):
            t = time.time()
            m = cm.train_model(Xtr, ytr, epochs=EPOCHS, log=log)
            out[tag] = (cm.predict(m, Xoth), cm.predict(m, AV))
            torch.save(m.state_dict(), CACHE / f"char8_model_{tag}.pt")
            log(f"char8 {tag}: {time.time() - t:.0f}s; val AUC {roc_auc_score(V0['y'], out[tag][1]):.5f}")
            del m
            torch.cuda.empty_cache()
        oof = pl.concat([pl.DataFrame({"q_idx": h1["q_idx"], "char_p": out["A"][0]}), pl.DataFrame({"q_idx": h0["q_idx"], "char_p": out["B"][0]})])
        val = pl.DataFrame({"q_idx": V0["q_idx"], "char_p": (out["A"][1] + out["B"][1]) / 2})
        oof.write_parquet(CACHE / "char8_oof_train.parquet")
        val.write_parquet(CACHE / "char8_val.parquet")
        del A0, A1, AV
    log(f"char8-only AUC val {roc_auc_score(V0['y'], val['char_p']):.5f} (4-epoch 0.97758)")
    T, V = common.load_rows()
    T = T.join(pl.read_parquet(E10 / "feat_hn_train.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(pl.read_parquet(E11 / "feat_ate_train.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(oof, on="q_idx", how="left", maintain_order="left") \
         .join(pl.read_parquet(E19 / "feat_asib_train.parquet"), on="q_idx", how="left", maintain_order="left")
    V = V.join(pl.read_parquet(E10 / "feat_hn_val.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(pl.read_parquet(E11 / "feat_ate_val.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(val, on="q_idx", how="left", maintain_order="left") \
         .join(pl.read_parquet(E19 / "feat_asib_val.parquet"), on="q_idx", how="left", maintain_order="left")
    models = {}
    for seed in ("42", "7"):
        os.environ["ER_LGB_SEED"] = seed
        path = CACHE / f"model_R10_s{seed}.txt"
        models[seed] = lgb.Booster(model_file=str(path)) if path.exists() else fit(T, F, path)
        pl.DataFrame({"q_idx": V["q_idx"], "p_stage2": models[seed].predict(V.select(F).to_numpy().astype(np.float32), num_threads=16)}) \
          .write_parquet(CACHE / f"val_scored_R10_s{seed}.parquet")
    del T
    from density import DensityEval
    from ef05 import ef05_accept
    ev = DensityEval()
    res = {}
    for name, path in (("R7_s42", E21 / "val_scored_R7_s42.parquet"), ("R7_s7", E21 / "val_scored_R7_s7.parquet"),
                       ("R10_s42", CACHE / "val_scored_R10_s42.parquet"), ("R10_s7", CACHE / "val_scored_R10_s7.parquet")):
        b = ev.frame(pl.read_parquet(path))
        bb = b.with_columns(ef05_accept(b, 0.5).cast(pl.Float64).alias("p_final"))
        r = {"plain_ef05": ev.evaluate(bb, pl.lit(0.5))["f05"], "proxy_same_k2.4": ev.evaluate(bb, pl.lit(0.5), 2.4, "same")["f05"],
             "proxy_spread_k1.5": ev.evaluate(bb, pl.lit(0.5), 1.5, "spread")["f05"]}
        res[name] = r
        log(f"{name}: {json.dumps({k: round(v, 6) for k, v in r.items()})}")
    # test shift (fold-A): R7 with 4-epoch test char scores vs R10 with 8-epoch test char scores
    X = pl.read_parquet(A02 / "cache/final_test_rows_cf1_ru0.parquet")
    qt, st = norm("test")
    p, AT = arrays(X, qt, st)
    c8 = []
    for tag in ("A", "B"):
        m = cm.CharPair().to(cm.DEV)
        m.load_state_dict(torch.load(CACHE / f"char8_model_{tag}.pt", map_location=cm.DEV))
        c8.append(cm.predict(m, AT))
        del m
    del AT
    torch.cuda.empty_cache()
    _, _, _, best = stage2.train_rows()
    best = best.with_columns(pl.col("q_is_val").fill_null(False))
    pool = best.filter(~pl.col("q_is_val") & pl.col("oos")).select("q_idx", "s1_idx", "y")
    qn, sn = norm("train")
    pp = pool.select("q_idx", "s1_idx").with_row_index("_r").join(qn, on="q_idx", how="left").join(sn, on="s1_idx", how="left", suffix="_s") \
        .sort("_r").join(pool.with_row_index("_r").select("_r", "y"), on="_r")
    te = ate.learn(pp)
    del pp, pool, qn, sn
    X = pl.concat([X, hn_features(p), ate.apply(p, te)], how="horizontal") \
          .join(pl.read_parquet(E19 / "feat_asib_test.parquet"), on="q_idx", how="left", maintain_order="left")
    X7 = X.join(pl.read_parquet(E21 / "char4_test.parquet"), on="q_idx", how="left", maintain_order="left")
    X10 = X.with_columns(pl.Series("char_p", ((c8[0] + c8[1]) / 2).astype(np.float32)))
    V7 = V.drop("char_p").join(pl.read_parquet(E21 / "char4_val.parquet"), on="q_idx", how="left", maintain_order="left")
    m7 = lgb.Booster(model_file=str(E21 / "model_R7_s42.txt"))
    shift = {}
    for name, D7, D10 in (("val", V7, V), ("test", X7, X10)):
        a7 = int((m7.predict(D7.select(F).to_numpy().astype(np.float32), num_threads=16) >= 0.7).sum())
        a10 = int((models["42"].predict(D10.select(F).to_numpy().astype(np.float32), num_threads=16) >= 0.7).sum())
        shift[name] = {"R7": a7, "R10": a10, "R10_vs_R7_pct": 100 * (a10 / a7 - 1)}
        log(f"shift {name}: {shift[name]}")
    res["shift"] = shift
    res["runtime_sec"] = round(time.time() - t0)
    json.dump(res, open(RES / "run24_summary.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
