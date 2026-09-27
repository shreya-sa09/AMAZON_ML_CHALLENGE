"""Exp 21: char model trained 4 epochs (vs 2) -> replace char_p in the R6 feature set (R7). Both seeds; plain + proxy;
test-shift check (fold-A R6 vs R7 band acceptances on validation vs test)."""
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
E10, E11, E17, E19 = (ROOT / f"approaches/{d}/cache" for d in ("10_STAGE2_ADDRESS_SIGNALS", "11_STAGE2_ADDRESS_TOKEN_ODDS", "17_CHAR_PAIR_MODEL", "19_ANCHOR_SIBLINGS"))
sys.path.insert(0, str(ROOT / "approaches/15_DENSITY_MATCHED_EVAL/src"))
sys.path.insert(0, str(ROOT / "approaches/18_EXPECTED_F05_DECISION/src"))
F = BASE_FEATURES + HN_COLS + ate.ATE_COLS + ["char_p"] + asib.ASIB_COLS
EPOCHS = 4


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
    X0t = pl.read_parquet(A02 / "cache/final_test_rows_cf1_ru0.parquet", columns=["q_idx", "s1_idx"])
    qn, sn = norm("train")
    _, A0 = arrays(h0, qn, sn)
    _, A1 = arrays(h1, qn, sn)
    _, AV = arrays(V0, qn, sn)
    qt, st = norm("test")
    pt_, AT = arrays(X0t, qt, st)
    out = {}
    if (CACHE / "char4_val.parquet").exists():
        oof, val, test = (pl.read_parquet(CACHE / f"char4_{n}.parquet") for n in ("oof_train", "val", "test"))
    else:
        for tag, Xtr, ytr, Xoth in (("A", A0, h0["y"].to_numpy(), A1), ("B", A1, h1["y"].to_numpy(), A0)):
            t = time.time()
            m = cm.train_model(Xtr, ytr, epochs=EPOCHS, log=log)
            out[tag] = (cm.predict(m, Xoth), cm.predict(m, AV), cm.predict(m, AT))
            torch.save(m.state_dict(), CACHE / f"char4_model_{tag}.pt")
            log(f"char4 {tag}: {time.time() - t:.0f}s; val AUC {roc_auc_score(V0['y'], out[tag][1]):.5f}")
            del m
            torch.cuda.empty_cache()
        oof = pl.concat([pl.DataFrame({"q_idx": h1["q_idx"], "char_p": out["A"][0]}), pl.DataFrame({"q_idx": h0["q_idx"], "char_p": out["B"][0]})])
        val = pl.DataFrame({"q_idx": V0["q_idx"], "char_p": (out["A"][1] + out["B"][1]) / 2})
        test = pl.DataFrame({"q_idx": X0t["q_idx"], "char_p": (out["A"][2] + out["B"][2]) / 2})
        for n, d in (("oof_train", oof), ("val", val), ("test", test)):
            d.write_parquet(CACHE / f"char4_{n}.parquet")
    log(f"char4-only AUC val {roc_auc_score(V0['y'], val['char_p']):.5f} (2-epoch: 0.96307)")
    del A0, A1, AV, AT
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
        path = CACHE / f"model_R7_s{seed}.txt"
        models[seed] = lgb.Booster(model_file=str(path)) if path.exists() else fit(T, F, path)
        pl.DataFrame({"q_idx": V["q_idx"], "p_stage2": models[seed].predict(V.select(F).to_numpy().astype(np.float32), num_threads=16)}) \
          .write_parquet(CACHE / f"val_scored_R7_s{seed}.parquet")
    del T
    from density import DensityEval
    from ef05 import ef05_accept
    ev = DensityEval()
    res = {}
    for name, path in (("R6_s42", E19 / "val_scored_R6_s42.parquet"), ("R6_s7", E19 / "val_scored_R6_s7.parquet"),
                       ("R7_s42", CACHE / "val_scored_R7_s42.parquet"), ("R7_s7", CACHE / "val_scored_R7_s7.parquet")):
        b = ev.frame(pl.read_parquet(path))
        bb = b.with_columns(pl.when(ef05_accept(b, 0.5)).then(pl.lit(1.0)).otherwise(pl.lit(0.0)).alias("p_final"))
        r = {"plain_ef05": ev.evaluate(bb, pl.lit(0.5))["f05"], "proxy_ef05": ev.evaluate(bb, pl.lit(0.5), 2.4, "spread")["f05"] - 0.0019,
             "proxy_ef05_k1.9": ev.evaluate(bb, pl.lit(0.5), 1.9, "spread")["f05"] - 0.0019}
        res[name] = r
        log(f"{name}: {json.dumps({k: round(v, 6) for k, v in r.items()})}")
    # test-shift: fold-A R6 vs R7 acceptances @0.70 on val band and test band
    _, _, _, best = stage2.train_rows()
    best = best.with_columns(pl.col("q_is_val").fill_null(False))
    pool = best.filter(~pl.col("q_is_val") & pl.col("oos")).select("q_idx", "s1_idx", "y")
    pp = pool.select("q_idx", "s1_idx").with_row_index("_r").join(qn, on="q_idx", how="left").join(sn, on="s1_idx", how="left", suffix="_s") \
        .sort("_r").join(pool.with_row_index("_r").select("_r", "y"), on="_r")
    te = ate.learn(pp)
    del pp, pool
    X = pl.read_parquet(A02 / "cache/final_test_rows_cf1_ru0.parquet")
    X = pl.concat([X, hn_features(pt_), ate.apply(pt_, te)], how="horizontal").join(pl.read_parquet(E19 / "feat_asib_test.parquet"), on="q_idx", how="left", maintain_order="left")
    X6 = X.join(pl.read_parquet(E17 / "char_val.parquet").head(0), on="q_idx", how="left")  # placeholder schema
    # R6 fold-A needs 2-epoch char scores on test: recompute with Exp 17 models
    Xa = {"qn": cm.encode(pt_["name_core"].fill_null("").str.replace_all(" ", "").to_list(), cm.LN),
          "sn": cm.encode(pt_["name_core_s"].fill_null("").str.replace_all(" ", "").to_list(), cm.LN),
          "qa": cm.encode(pt_["addr_norm"].fill_null("").to_list(), cm.LA), "sa": cm.encode(pt_["addr_norm_s"].fill_null("").to_list(), cm.LA)}
    c2 = []
    for tag in ("A", "B"):
        m = cm.CharPair().to(cm.DEV)
        m.load_state_dict(torch.load(E17 / f"char_model_{tag}.pt", map_location=cm.DEV))
        c2.append(cm.predict(m, Xa))
    X6 = X.with_columns(pl.Series("char_p", ((c2[0] + c2[1]) / 2).astype(np.float32)))
    X7 = X.join(test, on="q_idx", how="left", maintain_order="left")
    V6 = V.drop("char_p").join(pl.read_parquet(E17 / "char_val.parquet"), on="q_idx", how="left", maintain_order="left")
    m6 = lgb.Booster(model_file=str(E19 / "model_R6_s42.txt"))
    shift = {}
    for name, D6, D7 in (("val", V6, V), ("test", X6, X7)):
        a6 = int((m6.predict(D6.select(F).to_numpy().astype(np.float32), num_threads=16) >= 0.7).sum())
        a7 = int((models["42"].predict(D7.select(F).to_numpy().astype(np.float32), num_threads=16) >= 0.7).sum())
        shift[name] = {"R6": a6, "R7": a7, "R7_vs_R6_pct": 100 * (a7 / a6 - 1)}
        log(f"shift {name}: {shift[name]}")
    res["shift"] = shift
    res["runtime_sec"] = round(time.time() - t0)
    json.dump(res, open(RES / "run21_summary.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
