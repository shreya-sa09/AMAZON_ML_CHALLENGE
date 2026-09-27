"""Exp 15e (reverse direction): lenient for lone matches, strict inside groups. Read-only."""
import polars as pl
import common
from density import DensityEval
ev = DensityEval()
b = ev.frame(pl.read_parquet(common.ROOT / "approaches/11_STAGE2_ADDRESS_TOKEN_ODDS/cache/val_scored_R2_hn_ate.parquet"))
b = b.with_columns(((pl.col("p_final") >= 0.9).cast(pl.Int32).sum().over("s1_idx") - (pl.col("p_final") >= 0.9).cast(pl.Int32)).alias("n"))
for ta, tg in ((0.7, 0.8), (0.75, 0.8), (0.7, 0.85), (0.75, 0.85), (0.8, 0.85), (0.8, 0.9)):
    thr = pl.when(pl.col("n") > 0).then(pl.lit(tg)).otherwise(pl.lit(ta))
    p, q = ev.evaluate(b, thr, 2.4, "spread"), ev.evaluate(b, thr, 1.9, "spread")
    print(f"alone={ta} group={tg}: LB-proxy {p['f05'] - 0.0019:.6f} (k1.9 {q['f05'] - 0.0019:.6f})", flush=True)
