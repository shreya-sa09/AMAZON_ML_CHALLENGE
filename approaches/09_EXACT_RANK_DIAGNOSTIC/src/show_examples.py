"""Experiment 09c: fixed-seed raw-text examples of address-present in-band Stage-2 errors (read-only)."""
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "results"
A02 = ROOT / "approaches/02_STAGE2_CROSSFIT_RUNNERUP"
sys.stdout.reconfigure(encoding="utf-8")


def main():
    raw_q = pl.concat([pl.read_parquet(ROOT / f"cache/train_{s}.parquet", columns=["business_name", "business_address"])
                       for s in ("source2", "source3")]).with_row_index("q_idx")
    raw_s1 = pl.read_parquet(ROOT / "cache/train_source1.parquet",
                             columns=["business_name", "business_address", "country"]).with_row_index("s1_idx")
    v = pl.read_parquet(A02 / "cache/validate_val_rows_cf1_ru0.parquet", columns=["q_idx", "s1_idx", "y", "f_q_addr_missing"]) \
        .join(pl.read_parquet(A02 / "cache/validate_val_scored_cf1_ru0_r8000.parquet", columns=["q_idx", "p_stage2"]), on="q_idx") \
        .filter(pl.col("f_q_addr_missing") == 0)
    sets = {
        "rejected_correct_p0.3-0.7": v.filter(pl.col("y") & pl.col("p_stage2").is_between(0.3, 0.7, closed="left")),
        "rejected_correct_p<0.2": v.filter(pl.col("y") & (pl.col("p_stage2") < 0.2)),
        "accepted_wrong": v.filter(~pl.col("y") & (pl.col("p_stage2") >= 0.7)),
    }
    lines = []
    for name, d in sets.items():
        ex = d.sample(n=15, seed=7).join(raw_q, on="q_idx").join(raw_s1, on="s1_idx", suffix="_s1").sort("p_stage2")
        lines.append(f"\n===== {name} (n={d.height:,}) =====")
        for r in ex.iter_rows(named=True):
            lines.append(f"p={r['p_stage2']:.3f} [{r['country']}]\n  Q : {r['business_name']} | {r['business_address']}\n"
                         f"  S1: {r['business_name_s1']} | {r['business_address_s1']}")
    text = "\n".join(lines)
    (OUT / "stage2_error_examples.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
