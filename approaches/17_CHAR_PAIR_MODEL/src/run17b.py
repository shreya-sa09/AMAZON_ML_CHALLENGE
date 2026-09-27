"""Exp 17b: seed-stability refit of the Stage-2 model with the cached cross-fitted char scores (ER_LGB_SEED=7)."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

import common
from common import BASE_FEATURES, fit, log
from newfeats import HN_COLS
import ate

ROOT = common.ROOT
APP = Path(__file__).resolve().parents[1]
CACHE, RES = APP / "cache", APP / "results"
sys.path.insert(0, str(ROOT / "approaches/15_DENSITY_MATCHED_EVAL/src"))
sys.path.insert(0, str(ROOT / "approaches/18_EXPECTED_F05_DECISION/src"))
FEATS = BASE_FEATURES + HN_COLS + ate.ATE_COLS + ["char_p"]
E10, E11 = ROOT / "approaches/10_STAGE2_ADDRESS_SIGNALS/cache", ROOT / "approaches/11_STAGE2_ADDRESS_TOKEN_ODDS/cache"
seed = os.environ.get("ER_LGB_SEED", "42")
T0 = time.time()
T, V = common.load_rows()
T = T.join(pl.read_parquet(E10 / "feat_hn_train.parquet"), on="q_idx", how="left", maintain_order="left") \
     .join(pl.read_parquet(E11 / "feat_ate_train.parquet"), on="q_idx", how="left", maintain_order="left") \
     .join(pl.read_parquet(CACHE / "char_oof_train.parquet"), on="q_idx", how="left", maintain_order="left")
V = V.join(pl.read_parquet(E10 / "feat_hn_val.parquet"), on="q_idx", how="left", maintain_order="left") \
     .join(pl.read_parquet(E11 / "feat_ate_val.parquet"), on="q_idx", how="left", maintain_order="left") \
     .join(pl.read_parquet(CACHE / "char_val.parquet"), on="q_idx", how="left", maintain_order="left")
m = fit(T, FEATS, CACHE / f"model_R5_char_s{seed}.txt")
del T
pv = m.predict(V.select(FEATS).to_numpy().astype(np.float32), num_threads=16)
pl.DataFrame({"q_idx": V["q_idx"], "p_stage2": pv}).write_parquet(CACHE / f"val_scored_R5_char_s{seed}.parquet")
from density import DensityEval
from ef05 import ef05_accept
ev = DensityEval()
res = {}
for name, path in ((f"R2_s{seed}", E11 / ("val_scored_R2_hn_ate_s7.parquet" if seed == "7" else "val_scored_R2_hn_ate.parquet")),
                   (f"R5_char_s{seed}", CACHE / f"val_scored_R5_char_s{seed}.parquet")):
    b = ev.frame(pl.read_parquet(path))
    bb = b.with_columns(pl.when(ef05_accept(b, 0.5)).then(pl.lit(1.0)).otherwise(pl.lit(0.0)).alias("p_final"))
    r = {"plain_u0.7": ev.evaluate(b, pl.lit(0.7))["f05"], "plain_ef05": ev.evaluate(bb, pl.lit(0.5))["f05"],
         "proxy_ef05": ev.evaluate(bb, pl.lit(0.5), 2.4, "spread")["f05"] - 0.0019,
         "proxy_ef05_k1.9": ev.evaluate(bb, pl.lit(0.5), 1.9, "spread")["f05"] - 0.0019}
    res[name] = r
    log(f"{name}: {json.dumps({k: round(v, 6) for k, v in r.items()})}")
res["runtime_sec"] = round(time.time() - T0)
json.dump(res, open(RES / f"run17b_seed{seed}.json", "w"), indent=2)
