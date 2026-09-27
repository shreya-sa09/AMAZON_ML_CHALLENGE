"""Exp 17 driver: cross-fitted char-level pair model -> one extra Stage-2 feature on top of V2 (A02 + hn + ate).
Model A: train on h0 rows -> score h1 rows + val. Model B: train on h1 -> score h0 + val. Val score = mean(A, B).
Then the Approach 02 Stage-2 recipe with 116 features; evaluated plain + LB proxy with the Exp 18 EF0.5 rule (c=0.5)
and the Exp 15 group rule, against R2 (115 features) under the same policies."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score

import charmodel as cm
import common
from common import A02, BASE_FEATURES, SEED, fit, log
from newfeats import HN_COLS
import ate

ROOT = common.ROOT
APP = Path(__file__).resolve().parents[1]
CACHE, RES = APP / "cache", APP / "results"
sys.path.insert(0, str(ROOT / "approaches/15_DENSITY_MATCHED_EVAL/src"))
sys.path.insert(0, str(ROOT / "approaches/18_EXPECTED_F05_DECISION/src"))
FEATS = BASE_FEATURES + HN_COLS + ate.ATE_COLS + ["char_p"]


def encode_rows(rows, qn, sn):
    p = rows.select("q_idx", "s1_idx").with_row_index("_r").join(qn, on="q_idx", how="left").join(sn, on="s1_idx", how="left", suffix="_s").sort("_r")
    return {"qn": cm.encode(p["name_core"].fill_null("").str.replace_all(" ", "").to_list(), cm.LN),
            "sn": cm.encode(p["name_core_s"].fill_null("").str.replace_all(" ", "").to_list(), cm.LN),
            "qa": cm.encode(p["addr_norm"].fill_null("").to_list(), cm.LA),
            "sa": cm.encode(p["addr_norm_s"].fill_null("").to_list(), cm.LA)}


def main():
    t0 = time.time()
    h0 = pl.read_parquet(A02 / "cache/validate_train_rows_h0_cf1_ru0.parquet", columns=["q_idx", "s1_idx", "y"])
    h1 = pl.read_parquet(A02 / "cache/validate_train_rows_h1_cf1_ru0.parquet", columns=["q_idx", "s1_idx", "y"])
    V0 = pl.read_parquet(A02 / "cache/validate_val_rows_cf1_ru0.parquet", columns=["q_idx", "s1_idx", "y"])
    qn = pl.concat([pl.read_parquet(ROOT / f"cache/train_{s}_norm.parquet", columns=["name_core", "addr_norm"]) for s in ("source2", "source3")]).with_row_index("q_idx")
    sn = pl.read_parquet(ROOT / "cache/train_source1_norm.parquet", columns=["name_core", "addr_norm"]).with_row_index("s1_idx")
    X0, X1, XV = encode_rows(h0, qn, sn), encode_rows(h1, qn, sn), encode_rows(V0, qn, sn)
    del qn, sn
    log(f"encoded h0 {h0.height:,} h1 {h1.height:,} val {V0.height:,}")
    scores = {}
    for tag, Xtr, ytr, Xoth in (("A", X0, h0["y"].to_numpy(), X1), ("B", X1, h1["y"].to_numpy(), X0)):
        t = time.time()
        m = cm.train_model(Xtr, ytr, epochs=2, log=log)
        scores[tag] = (cm.predict(m, Xoth), cm.predict(m, XV))
        torch.save(m.state_dict(), CACHE / f"char_model_{tag}.pt")
        log(f"char model {tag}: trained+scored in {time.time() - t:.0f}s; val AUC {roc_auc_score(V0['y'], scores[tag][1]):.5f}")
        del m
        torch.cuda.empty_cache()
    oof = pl.concat([pl.DataFrame({"q_idx": h1["q_idx"], "char_p": scores["A"][0]}), pl.DataFrame({"q_idx": h0["q_idx"], "char_p": scores["B"][0]})])
    val = pl.DataFrame({"q_idx": V0["q_idx"], "char_p": (scores["A"][1] + scores["B"][1]) / 2})
    oof.write_parquet(CACHE / "char_oof_train.parquet")
    val.write_parquet(CACHE / "char_val.parquet")
    log(f"char-only AUC: val {roc_auc_score(V0['y'], val['char_p']):.5f}; train OOF {roc_auc_score(pl.concat([h1['y'], h0['y']]), oof['char_p']):.5f}")

    # Stage-2 with the extra feature (identical recipe; hn/ate from Exp 10/11 caches)
    T, V = common.load_rows()
    for name, rows in (("train", None), ("val", None)):
        pass
    E10, E11 = ROOT / "approaches/10_STAGE2_ADDRESS_SIGNALS/cache", ROOT / "approaches/11_STAGE2_ADDRESS_TOKEN_ODDS/cache"
    T = T.join(pl.read_parquet(E10 / "feat_hn_train.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(pl.read_parquet(E11 / "feat_ate_train.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(oof, on="q_idx", how="left", maintain_order="left")
    V = V.join(pl.read_parquet(E10 / "feat_hn_val.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(pl.read_parquet(E11 / "feat_ate_val.parquet"), on="q_idx", how="left", maintain_order="left") \
         .join(val, on="q_idx", how="left", maintain_order="left")
    m = fit(T, FEATS, CACHE / "model_R5_char.txt")
    del T
    pv = m.predict(V.select(FEATS).to_numpy().astype(np.float32), num_threads=16)
    pl.DataFrame({"q_idx": V["q_idx"], "p_stage2": pv}).write_parquet(CACHE / "val_scored_R5_char.parquet")

    from density import DensityEval
    from ef05 import ef05_accept
    ev = DensityEval()
    res = {}
    for name, path in (("R2_s42", E11 / "val_scored_R2_hn_ate.parquet"), ("R5_char", CACHE / "val_scored_R5_char.parquet")):
        b = ev.frame(pl.read_parquet(path))
        acc = ef05_accept(b, 0.5)
        bb = b.with_columns(pl.when(acc).then(pl.lit(1.0)).otherwise(pl.lit(0.0)).alias("p_final"))
        n = ((pl.col("p_final") >= 0.9).cast(pl.Int32).sum().over("s1_idx") - (pl.col("p_final") >= 0.9).cast(pl.Int32))
        g = b.with_columns(pl.when(n > 0).then(pl.lit(0.85)).otherwise(pl.lit(0.70)).alias("thr"))
        r = {"plain_u0.7": ev.evaluate(b, pl.lit(0.7))["f05"], "plain_ef05": ev.evaluate(bb, pl.lit(0.5))["f05"],
             "proxy_ef05": ev.evaluate(bb, pl.lit(0.5), 2.4, "spread")["f05"] - 0.0019,
             "proxy_ef05_k1.9": ev.evaluate(bb, pl.lit(0.5), 1.9, "spread")["f05"] - 0.0019,
             "proxy_group": ev.evaluate(g, pl.col("thr"), 2.4, "spread")["f05"] - 0.0019}
        res[name] = r
        log(f"{name}: {json.dumps({k: round(v, 6) for k, v in r.items()})}")
    res["best_iteration_R5"] = m.best_iteration
    res["runtime_sec"] = round(time.time() - t0)
    json.dump(res, open(RES / "run17_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
