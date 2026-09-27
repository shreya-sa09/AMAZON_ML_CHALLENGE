"""Exp 15: calibrate the density simulation against the two real leaderboard points.
LB: Approach 02 @0.50 = 0.977984 ; FINAL_HN_ONLY (Exp 10 R1) @0.70 = 0.981172  (delta +0.003188)."""
import json
import time
from pathlib import Path

import polars as pl

import common
from density import DensityEval

RES = Path(__file__).resolve().parents[1] / "results"
LB = {"A02@0.50": 0.977984, "R1@0.70": 0.981172}
T0 = time.time()


def main():
    ev = DensityEval()
    a02 = ev.frame(pl.read_parquet(common.A02 / "cache/validate_val_scored_cf1_ru0_r8000.parquet", columns=["q_idx", "p_stage2"]))
    r1 = ev.frame(pl.read_parquet(common.ROOT / "approaches/10_STAGE2_ADDRESS_SIGNALS/cache/val_scored_R1_hn.parquet"))
    cfgs = {"A02@0.50": (a02, 0.50), "A02@0.70": (a02, 0.70), "R1@0.70": (r1, 0.70)}
    rows = []
    for mode in ("same", "spread"):
        for k in (1.0, 1.5, 1.9, 2.5):
            r = {"mode": mode, "k": k}
            for name, (b, t) in cfgs.items():
                m = ev.evaluate(b, pl.lit(t), k, mode)
                r[name] = m["f05"]
                r[name + "_fs"] = m["false_singletons"]
            r["sim_delta_R1_vs_A02@0.50"] = r["R1@0.70"] - r["A02@0.50"]
            r["sim_delta_A02_0.70_vs_0.50"] = r["A02@0.70"] - r["A02@0.50"]
            r["lb_delta"] = LB["R1@0.70"] - LB["A02@0.50"]
            r["gap_R1_sim_minus_lb"] = r["R1@0.70"] - LB["R1@0.70"]
            rows.append(r)
            common.log(f"{mode} k={k}: A02@.5 {r['A02@0.50']:.6f} A02@.7 {r['A02@0.70']:.6f} R1@.7 {r['R1@0.70']:.6f} "
                       f"sim_delta {r['sim_delta_R1_vs_A02@0.50']:+.6f} (LB {r['lb_delta']:+.6f})")
    pl.DataFrame(rows).write_csv(RES / "calibration.csv")
    (RES / "calibration.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    common.log(f"done in {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
