"""22a: proxy recalibration on 3 real LB points + R7 seed-ensemble check. Read-only."""
import sys
from pathlib import Path
import numpy as np
import polars as pl
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "approaches/15_DENSITY_MATCHED_EVAL/src"))
sys.path.insert(0, str(ROOT / "approaches/18_EXPECTED_F05_DECISION/src"))
import common  # noqa
from density import DensityEval
from ef05 import ef05_accept
ev = DensityEval()
E21 = ROOT / "approaches/21_CHAR_LONGER/cache"
A02, E10 = common.A02, ROOT / "approaches/10_STAGE2_ADDRESS_SIGNALS/cache"
LB = {"A02@0.50": 0.977984, "HN@0.70": 0.981172, "V4_ef05": 0.982899}
a02 = ev.frame(pl.read_parquet(A02 / "cache/validate_val_scored_cf1_ru0_r8000.parquet", columns=["q_idx", "p_stage2"]))
hn = ev.frame(pl.read_parquet(E10 / "val_scored_R1_hn.parquet"))
v4 = ev.frame(pl.read_parquet(E21 / "val_scored_R7_s42.parquet"))
v4 = v4.with_columns(ef05_accept(v4, 0.5).cast(pl.Float64).alias("p_final"))
cfg = {"A02@0.50": (a02, 0.5), "HN@0.70": (hn, 0.7), "V4_ef05": (v4, 0.5)}
print("== proxy recalibration (sim - LB should be constant across the 3 submissions)")
for mode in ("same", "spread"):
    for k in (1.0, 1.25, 1.5, 1.75, 1.9, 2.4):
        if mode == "same" and k == 1.0:
            continue
        d = {n: ev.evaluate(b, pl.lit(t), k, mode)["f05"] - LB[n] for n, (b, t) in cfg.items()}
        vals = np.array(list(d.values()))
        print(f"{mode:6s} k={k}: offsets " + " ".join(f"{n}={v:.5f}" for n, v in d.items()) + f"  spread(max-min)={vals.max() - vals.min():.5f}", flush=True)
d = {n: ev.evaluate(b, pl.lit(t))["f05"] - LB[n] for n, (b, t) in cfg.items()}
print("plain k=1   : offsets " + " ".join(f"{n}={v:.5f}" for n, v in d.items()) + f"  spread={max(d.values()) - min(d.values()):.5f}")
print("== R7 seed ensemble (mean of s42 and s7 probabilities), EF0.5 c=0.5")
s42 = pl.read_parquet(E21 / "val_scored_R7_s42.parquet").rename({"p_stage2": "a"})
s7 = pl.read_parquet(E21 / "val_scored_R7_s7.parquet").rename({"p_stage2": "b"})
ens = s42.join(s7, on="q_idx").select("q_idx", ((pl.col("a") + pl.col("b")) / 2).alias("p_stage2"))
for name, sc in (("R7_s42", pl.read_parquet(E21 / "val_scored_R7_s42.parquet")), ("R7_s7", pl.read_parquet(E21 / "val_scored_R7_s7.parquet")), ("R7_ens", ens)):
    b = ev.frame(sc)
    bb = b.with_columns(ef05_accept(b, 0.5).cast(pl.Float64).alias("p_final"))
    print(f"{name}: plain {ev.evaluate(bb, pl.lit(0.5))['f05']:.6f}  k1.5 {ev.evaluate(bb, pl.lit(0.5), 1.5, 'spread')['f05']:.6f}  k2.4 {ev.evaluate(bb, pl.lit(0.5), 2.4, 'spread')['f05']:.6f}", flush=True)
