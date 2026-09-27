"""Exp 15d: leaderboard-proxy (spread, k=2.4, minus 0.0019) comparison of saved validation scores. Read-only."""
import sys
import time
from pathlib import Path

import polars as pl

import common
from density import DensityEval

RES = Path(__file__).resolve().parents[1] / "results"
OFFSET = 0.0019
E10 = common.ROOT / "approaches/10_STAGE2_ADDRESS_SIGNALS/cache"
E11 = common.ROOT / "approaches/11_STAGE2_ADDRESS_TOKEN_ODDS/cache"
E13 = common.ROOT / "approaches/13_STAGE2_SIBLING_EVIDENCE/cache"
MODELS = {"R1_hn_s42": E10 / "val_scored_R1_hn.parquet", "R1_hn_s7": E11 / "val_scored_R1_hn_s7.parquet",
          "R2_ate_s42": E11 / "val_scored_R2_hn_ate.parquet", "R2_ate_s7": E11 / "val_scored_R2_hn_ate_s7.parquet",
          "R3_sib_s42": E13 / "val_scored_R3_hn_sib.parquet"}


def main():
    ev = DensityEval()
    rows = []
    for name, path in MODELS.items():
        b = ev.frame(pl.read_parquet(path))
        for t in (0.70, 0.80, 0.85):
            plain = ev.evaluate(b, pl.lit(t))
            prox = ev.evaluate(b, pl.lit(t), 2.4, "spread")
            rows.append({"model": name, "t": t, "plain_f05": plain["f05"], "proxy_f05": prox["f05"],
                         "lb_proxy": prox["f05"] - OFFSET, "proxy_P": prox["precision"], "proxy_R": prox["recall"],
                         "proxy_FS": prox["false_singletons"]})
            common.log(f"{name:12s} t={t}: plain {plain['f05']:.6f}  LB-proxy {prox['f05'] - OFFSET:.6f}")
    pl.DataFrame(rows).write_csv(RES / "proxy_compare.csv")


if __name__ == "__main__":
    main()
