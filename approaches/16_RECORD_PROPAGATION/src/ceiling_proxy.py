"""Leaderboard-proxy ceilings (spread k=2.4, LB ~= proxy - 0.0019) starting from FINAL_V2 validation scores (Exp 11 R2 s42 +
group rule). Oracles replace fold-B decisions for one population at a time. Read-only."""
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "approaches/15_DENSITY_MATCHED_EVAL/src"))
import common  # noqa: E402
from density import DensityEval  # noqa: E402

ev = DensityEval()
b = ev.frame(pl.read_parquet(ROOT / "approaches/11_STAGE2_ADDRESS_TOKEN_ODDS/cache/val_scored_R2_hn_ate.parquet"))
n = ((pl.col("p_final") >= 0.9).cast(pl.Int32).sum().over("s1_idx") - (pl.col("p_final") >= 0.9).cast(pl.Int32))
b = b.with_columns(pl.when(n > 0).then(pl.lit(0.85)).otherwise(pl.lit(0.70)).alias("thr"))
cands = pl.concat([pl.read_parquet(f, columns=["q_idx", "s1_idx"]) for f in sorted((ROOT / "cache/train_cands").glob("*.parquet"))])
in10 = cands.join(b.filter(pl.col("q_is_val") & pl.col("true_s1").is_not_null()).select("q_idx", pl.col("true_s1").alias("s1_idx")),
                  on=["q_idx", "s1_idx"], how="semi").select("q_idx").unique().with_columns(pl.lit(True).alias("t10"))
b = b.join(in10, on="q_idx", how="left").with_columns(pl.col("t10").fill_null(False), pl.col("addr_missing").fill_null(False))
sid = ev.best.select("s1_idx", "s1_id").unique()
V, M, AP = pl.col("q_is_val"), pl.col("true_s1").is_not_null(), ~pl.col("addr_missing")


def run(name, sel_fix):
    """sel_fix: fold-B rows whose decision is replaced by the oracle (accept the true S1 iff it is reachable, else reject)."""
    x = b.with_columns(pl.when(V & sel_fix).then(pl.when(M).then(pl.col("true_s1")).otherwise(pl.col("s1_idx"))).otherwise(pl.col("s1_idx")).alias("s1_new"),
                       pl.when(V & sel_fix).then(pl.when(M).then(pl.lit(2.0)).otherwise(pl.lit(-1.0))).otherwise(pl.col("p_final")).alias("p_final"))
    x = x.drop("s1_id").join(sid.rename({"s1_idx": "s1_new"}), on="s1_new", how="left")
    p, q = ev.evaluate(x, pl.col("thr")), ev.evaluate(x, pl.col("thr"), 2.4, "spread")
    print(f"{name:58s} plain {p['f05']:.6f}  LB-proxy {q['f05'] - 0.0019:.6f}", flush=True)


run("FINAL_V2 (no oracle)", pl.lit(False))
run("oracle: address-present, Stage-2 decisions on top-1 only", AP & (pl.col("s1_idx") == pl.col("true_s1")).fill_null(False) | AP & ~M)
run("oracle: address-present, everything within top-10", AP & (pl.col("t10") | ~M))
run("oracle: address-present, everything incl. retrieval", AP)
run("oracle: all decisions on reachable (top-10) records", pl.col("t10") | ~M)
run("oracle: everything (perfect)", pl.lit(True))
