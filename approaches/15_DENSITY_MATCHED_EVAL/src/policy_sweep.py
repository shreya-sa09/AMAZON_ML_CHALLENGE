"""Exp 15b: decision policy for FINAL_HN_ONLY scores under the calibrated leaderboard proxy (spread, k~2.4).
Global threshold sweep + separate thresholds for address-missing vs address-present records. Read-only."""
import json
import time
from pathlib import Path

import polars as pl

import common
from density import DensityEval

RES = Path(__file__).resolve().parents[1] / "results"
OFFSET = 0.0019   # calibrated: LB ~= sim(spread, k~2.4) - 0.0019
T0 = time.time()


def main():
    ev = DensityEval()
    r1 = ev.frame(pl.read_parquet(common.ROOT / "approaches/10_STAGE2_ADDRESS_SIGNALS/cache/val_scored_R1_hn.parquet"))
    rows = []

    def run(name, thr, ks=(1.0, 1.9, 2.4, 2.5)):
        r = {"policy": name}
        for k in ks:
            m = ev.evaluate(r1, thr, k, "spread" if k > 1 else "same")
            r[f"f05_k{k}"] = m["f05"]
            if k == 2.4:
                r.update({"P_k2.4": m["precision"], "R_k2.4": m["recall"], "FS_k2.4": m["false_singletons"], "accepted": m["accepted"],
                          "lb_proxy": m["f05"] - OFFSET})
        rows.append(r)
        common.log(f"{name:28s} plain {r['f05_k1.0']:.6f}  k1.9 {r['f05_k1.9']:.6f}  k2.4 {r['f05_k2.4']:.6f}  "
                   f"k2.5 {r['f05_k2.5']:.6f}  LB-proxy {r['lb_proxy']:.6f}")

    for t in (0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95):
        run(f"global t={t}", pl.lit(t))
    am = pl.col("addr_missing").fill_null(False)
    for tp_ in (0.70, 0.80, 0.85, 0.90):
        for tm in (0.80, 0.90, 0.95, 0.98):
            if tm <= tp_:
                continue
            run(f"present={tp_} missing={tm}", pl.when(am).then(pl.lit(tm)).otherwise(pl.lit(tp_)))
    pl.DataFrame(rows).write_csv(RES / "policy_sweep.csv")
    common.log(f"done in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
