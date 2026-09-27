"""R6: EF0.5 calibration re-check (both seeds) + remaining leaderboard-proxy ceilings. Read-only."""
import sys
from pathlib import Path
import polars as pl
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "approaches/15_DENSITY_MATCHED_EVAL/src"))
sys.path.insert(0, str(ROOT / "approaches/18_EXPECTED_F05_DECISION/src"))
import common  # noqa
from density import DensityEval
from ef05 import ef05_accept
ev = DensityEval()
E19 = ROOT / "approaches/19_ANCHOR_SIBLINGS/cache"
for s in ("42", "7"):
    b = ev.frame(pl.read_parquet(E19 / f"val_scored_R6_s{s}.parquet"))
    for c in (0.35, 0.5, 0.7):
        bb = b.with_columns(pl.when(ef05_accept(b, c)).then(pl.lit(1.0)).otherwise(pl.lit(0.0)).alias("p_final"))
        print(f"R6 s{s} c={c}: plain {ev.evaluate(bb, pl.lit(0.5))['f05']:.6f}  proxy {ev.evaluate(bb, pl.lit(0.5), 2.4, 'spread')['f05'] - 0.0019:.6f}  k1.9 {ev.evaluate(bb, pl.lit(0.5), 1.9, 'spread')['f05'] - 0.0019:.6f}", flush=True)
# remaining ceilings from R6 s42 with EF0.5 c=0.5
b = ev.frame(pl.read_parquet(E19 / "val_scored_R6_s42.parquet"))
b = b.with_columns(pl.when(ef05_accept(b, 0.5)).then(pl.lit(1.0)).otherwise(pl.lit(0.0)).alias("p_final"), pl.col("addr_missing").fill_null(False))
sid = ev.best.select("s1_idx", "s1_id").unique()
V, M, AP = pl.col("q_is_val"), pl.col("true_s1").is_not_null(), ~pl.col("addr_missing")
def run(name, sel):
    x = b.with_columns(pl.when(V & sel & M).then(pl.col("true_s1")).otherwise(pl.col("s1_idx")).alias("s1_new"),
                       pl.when(V & sel).then(pl.when(M).then(pl.lit(1.0)).otherwise(pl.lit(0.0))).otherwise(pl.col("p_final")).alias("p_final"))
    x = x.drop("s1_id").join(sid.rename({"s1_idx": "s1_new"}), on="s1_new", how="left")
    print(f"{name:52s} plain {ev.evaluate(x, pl.lit(0.5))['f05']:.6f}  proxy {ev.evaluate(x, pl.lit(0.5), 2.4, 'spread')['f05'] - 0.0019:.6f}", flush=True)
run("R6 + EF0.5 (no oracle)", pl.lit(False))
run("oracle: address-present top-1 decisions", AP & ((pl.col("s1_idx") == pl.col("true_s1")).fill_null(False) | ~M))
run("oracle: address-present incl. retrieval/ranking", AP)
run("oracle: no-address incl. retrieval/ranking", ~AP)
