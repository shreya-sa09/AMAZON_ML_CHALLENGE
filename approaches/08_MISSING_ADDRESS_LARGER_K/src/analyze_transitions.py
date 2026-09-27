"""Compact decision-transition audit for the bounded validation sample."""
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "results"


def main():
    keys = pl.read_csv(OUT / "validation_sample_keys.csv", schema_overrides={"q_idx": pl.UInt32, "true_s1": pl.UInt32})
    old = (pl.read_parquet(ROOT / "cache/train_best.parquet")
           .join(keys.select("q_idx"), on="q_idx", how="semi")
           .join(pl.read_parquet(ROOT / "approaches/02_STAGE2_CROSSFIT_RUNNERUP/cache/validate_val_scored_cf1_ru0_r8000.parquet",
                                 columns=["q_idx", "p_stage2"]), on="q_idx", how="left")
           .with_columns(pl.coalesce("p_stage2", "p1").alias("old_p"))
           .select("q_idx", pl.col("s1_idx").alias("old_s1"), "old_p"))
    new = (pl.read_parquet(OUT / "validation_sample_new_best.parquet")
           .select("q_idx", pl.col("s1_idx").alias("new_s1"), pl.col("p_final").alias("new_p")))
    d = keys.join(old, on="q_idx").join(new, on="q_idx").with_columns(
        (pl.col("old_s1") == pl.col("true_s1")).fill_null(False).alias("old_top1_true"),
        (pl.col("new_s1") == pl.col("true_s1")).fill_null(False).alias("new_top1_true"),
        (pl.col("old_p") >= 0.7).alias("old_accept"),
        (pl.col("new_p") >= 0.7).alias("new_accept"))
    d = d.with_columns((pl.col("old_top1_true") & pl.col("old_accept")).alias("old_tp"),
                       (pl.col("new_top1_true") & pl.col("new_accept")).alias("new_tp"))
    def n(expr):
        return d.filter(expr).height
    smoke = pl.read_csv(OUT / "smoke_k80_pairs.csv")
    result = {"sample_queries": d.height,
              "gained_correct_top1": n(~pl.col("old_top1_true") & pl.col("new_top1_true")),
              "lost_correct_top1": n(pl.col("old_top1_true") & ~pl.col("new_top1_true")),
              "gained_accepted_true": n(~pl.col("old_tp") & pl.col("new_tp")),
              "lost_accepted_true": n(pl.col("old_tp") & ~pl.col("new_tp")),
              "old_accepted_wrong": n(pl.col("old_accept") & ~pl.col("old_top1_true")),
              "new_accepted_wrong": n(pl.col("new_accept") & ~pl.col("new_top1_true")),
              "old_total_accepted_sample": n(pl.col("old_accept")),
              "new_total_accepted_sample": n(pl.col("new_accept")),
              "smoke_recovered_top20_from_old_top40": smoke.filter(pl.col("new_top20") & pl.col("in_old_top40")).height,
              "smoke_recovered_top20_absent_old_top40": smoke.filter(pl.col("new_top20") & ~pl.col("in_old_top40")).height}
    (OUT / "transition_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
