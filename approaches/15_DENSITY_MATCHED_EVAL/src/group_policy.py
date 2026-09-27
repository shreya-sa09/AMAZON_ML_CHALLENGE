"""Exp 15e: metric-aware decision rule. Macro F0.5 is per S1: a false positive that would be an S1's ONLY
accepted record costs far more than one inside a group of confident matches. Threshold depends on whether another
record already points to the same S1 with p >= CONF (model scores only, no labels). Proxy = spread k=2.4 - 0.0019."""
import sys
import time
from pathlib import Path

import polars as pl

import common
from density import DensityEval

RES = Path(__file__).resolve().parents[1] / "results"
OFFSET = 0.0019
E11 = common.ROOT / "approaches/11_STAGE2_ADDRESS_TOKEN_ODDS/cache"
E10 = common.ROOT / "approaches/10_STAGE2_ADDRESS_SIGNALS/cache"
CONF = 0.9


def main():
    ev = DensityEval()
    which = sys.argv[1] if len(sys.argv) > 1 else "R2_s42"
    path = {"R2_s42": E11 / "val_scored_R2_hn_ate.parquet", "R2_s7": E11 / "val_scored_R2_hn_ate_s7.parquet",
            "R1_s42": E10 / "val_scored_R1_hn.parquet"}[which]
    b = ev.frame(pl.read_parquet(path))
    # number of OTHER records pointing to the same S1 with p_final >= CONF
    b = b.with_columns(((pl.col("p_final") >= CONF).cast(pl.Int32).sum().over("s1_idx")
                        - (pl.col("p_final") >= CONF).cast(pl.Int32)).alias("n_other_conf"))
    rows = []
    for ta in (0.80, 0.85, 0.90, 0.95):
        for tg in (0.50, 0.60, 0.70, 0.80):
            if tg > ta:
                continue
            thr = pl.when(pl.col("n_other_conf") > 0).then(pl.lit(tg)).otherwise(pl.lit(ta))
            plain = ev.evaluate(b, thr)
            prox = ev.evaluate(b, thr, 2.4, "spread")
            prox19 = ev.evaluate(b, thr, 1.9, "spread")
            rows.append({"model": which, "t_alone": ta, "t_group": tg, "plain": plain["f05"], "lb_proxy": prox["f05"] - OFFSET,
                         "lb_proxy_k1.9": prox19["f05"] - OFFSET, "proxy_P": prox["precision"], "proxy_R": prox["recall"],
                         "proxy_FS": prox["false_singletons"], "accepted": prox["accepted"]})
            common.log(f"{which} alone={ta} group={tg}: plain {plain['f05']:.6f}  LB-proxy {prox['f05'] - OFFSET:.6f} "
                       f"(k1.9 {prox19['f05'] - OFFSET:.6f})  FS {prox['false_singletons']}  acc {prox['accepted']:,}")
    pl.DataFrame(rows).write_csv(RES / f"group_policy_{which}.csv")


if __name__ == "__main__":
    main()
