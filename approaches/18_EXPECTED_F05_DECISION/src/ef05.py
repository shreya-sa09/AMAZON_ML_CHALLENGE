"""Exp 18: expected-F0.5 subset selection per S1 (Bayes-optimal decision rule given calibrated probabilities).

For each S1: records whose top-1 is this S1, sorted by calibrated p (desc). Choose k maximizing
    E[F0.5](k) ~= 1.25 * sum(p_1..k) / (0.25 * sum(p_all) + k)
and accept that prefix only if it beats the empty set, P(no true match) = prod(1 - p_i).
Calibration to test density: odds multiplier c (p' = c*odds / (1 + c*odds)).
Scored with the Exp 15 leaderboard proxy (spread decoys k=2.4, LB ~= proxy - 0.0019). Read-only."""
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "approaches/15_DENSITY_MATCHED_EVAL/src"))
import common  # noqa: E402
from density import DensityEval  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
E11 = ROOT / "approaches/11_STAGE2_ADDRESS_TOKEN_ODDS/cache"


def ef05_accept(b, c=1.0, floor=0.0):
    """b: frame with s1_idx, p_final. Returns boolean Series 'accept' aligned with b (via row index)."""
    p = b["p_final"].clip(1e-6, 1 - 1e-6)
    odds = p / (1 - p) * c
    pc = odds / (1 + odds)
    x = b.select("s1_idx").with_row_index("_r").with_columns(pl.Series("pc", pc))
    x = x.filter(pl.col("pc") >= floor) if floor > 0 else x
    x = x.sort(["s1_idx", "pc"], descending=[False, True]).with_columns(
        pl.col("pc").cum_sum().over("s1_idx").alias("cum"),
        pl.col("pc").sum().over("s1_idx").alias("tot"),
        pl.int_range(1, pl.len() + 1).over("s1_idx").alias("k"),
        (1 - pl.col("pc")).log().sum().over("s1_idx").exp().alias("p_none"))
    x = x.with_columns((1.25 * pl.col("cum") / (0.25 * pl.col("tot") + pl.col("k"))).alias("val"))
    x = x.with_columns(pl.col("val").max().over("s1_idx").alias("vmax"))
    kbest = x.filter(pl.col("val") == pl.col("vmax")).group_by("s1_idx").agg(pl.col("k").min().alias("kbest"))
    x = x.join(kbest, on="s1_idx").with_columns(((pl.col("k") <= pl.col("kbest")) & (pl.col("vmax") > pl.col("p_none"))).alias("acc"))
    acc = np.zeros(b.height, dtype=bool)
    acc[x.filter(pl.col("acc"))["_r"].to_numpy()] = True
    return pl.Series("accept", acc)


def main():
    ev = DensityEval()
    rows = []
    for name, f in (("R2_s42", "val_scored_R2_hn_ate.parquet"), ("R2_s7", "val_scored_R2_hn_ate_s7.parquet")):
        b = ev.frame(pl.read_parquet(E11 / f))
        n = ((pl.col("p_final") >= 0.9).cast(pl.Int32).sum().over("s1_idx") - (pl.col("p_final") >= 0.9).cast(pl.Int32))
        g = b.with_columns(pl.when(n > 0).then(pl.lit(0.85)).otherwise(pl.lit(0.70)).alias("thr"))
        pr, p19 = ev.evaluate(g, pl.col("thr"), 2.4, "spread"), ev.evaluate(g, pl.col("thr"), 1.9, "spread")
        common.log(f"{name} group rule (baseline): LB-proxy {pr['f05'] - 0.0019:.6f} (k1.9 {p19['f05'] - 0.0019:.6f})")
        rows.append({"model": name, "policy": "group_rule", "lb_proxy": pr["f05"] - 0.0019, "lb_proxy_k1.9": p19["f05"] - 0.0019})
        for c in (1.0, 0.7, 0.5, 0.35, 0.25):
            acc = ef05_accept(b, c)
            bb = b.with_columns(pl.when(acc).then(pl.lit(1.0)).otherwise(pl.lit(0.0)).alias("p_final"))
            plain = ev.evaluate(bb, pl.lit(0.5))
            p24, p19 = ev.evaluate(bb, pl.lit(0.5), 2.4, "spread"), ev.evaluate(bb, pl.lit(0.5), 1.9, "spread")
            rows.append({"model": name, "policy": f"ef05_c{c}", "plain": plain["f05"], "lb_proxy": p24["f05"] - 0.0019,
                         "lb_proxy_k1.9": p19["f05"] - 0.0019, "P": p24["precision"], "R": p24["recall"], "FS": p24["false_singletons"],
                         "accepted": p24["accepted"]})
            common.log(f"{name} EF0.5 c={c}: plain {plain['f05']:.6f}  LB-proxy {p24['f05'] - 0.0019:.6f} (k1.9 {p19['f05'] - 0.0019:.6f}) "
                       f"acc {p24['accepted']:,} FS {p24['false_singletons']}")
    pl.DataFrame(rows).write_csv(RES / "ef05_sweep.csv")


if __name__ == "__main__":
    main()
