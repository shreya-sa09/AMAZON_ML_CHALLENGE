"""Pre-check for Experiment 13: collective 'sibling' evidence. Siblings of a (query q, S1 x) row = other S2/S3 records
whose Stage-1 top-1 is the same x (cache/train_best). Label-free text similarities only.
Fold-A Stage-2 TRAINING rows (h0), address present; read-only."""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "results"
A02 = ROOT / "approaches/02_STAGE2_CROSSFIT_RUNNERUP"
E10 = ROOT / "approaches/10_STAGE2_ADDRESS_SIGNALS/cache"
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_formatting("ASCII_MARKDOWN"); pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)
T0 = time.time()
CAP = 50


def cp(a, b):
    return process.cpdist(a, b, scorer=fuzz.ratio, workers=-1, dtype=np.float32)


def main():
    t = pl.read_parquet(A02 / "cache/validate_train_rows_h0_cf1_ru0.parquet",
                        columns=["q_idx", "s1_idx", "y", "f_q_addr_missing", "f_ad_ratio", "f_q_is_s3"]) \
        .filter(pl.col("f_q_addr_missing") == 0) \
        .join(pl.read_parquet(E10 / "feat_hn_train.parquet", columns=["q_idx", "hn_absd1"]), on="q_idx")
    best = pl.read_parquet(ROOT / "cache/train_best.parquet", columns=["q_idx", "s1_idx", "p1"])
    qn = pl.concat([pl.read_parquet(ROOT / f"cache/train_{s}_norm.parquet", columns=["name_core", "addr_norm"]).with_columns(
        pl.lit(s == "source3").alias("is_s3")) for s in ("source2", "source3")]).with_row_index("q_idx")
    sn = pl.read_parquet(ROOT / "cache/train_source1_norm.parquet", columns=["addr_norm"]).with_row_index("s1_idx")
    sib = t.select("q_idx", "s1_idx").join(best.rename({"q_idx": "q2", "p1": "p1_2"}), on="s1_idx") \
        .filter(pl.col("q2") != pl.col("q_idx"))
    sib = sib.with_columns(pl.int_range(pl.len()).over("q_idx").alias("_k")).filter(pl.col("_k") < CAP).drop("_k")
    print(f"[{time.time()-T0:.0f}s] sibling pairs {sib.height:,} for {t.height:,} rows", flush=True)
    sib = sib.join(qn.rename({"q_idx": "q2", "name_core": "nm2", "addr_norm": "ad2", "is_s3": "s3_2"}), on="q2") \
        .join(qn.select("q_idx", "name_core", "addr_norm", "is_s3"), on="q_idx") \
        .join(sn.rename({"addr_norm": "ad_s1"}), on="s1_idx")
    sib = sib.with_columns(pl.Series("r_ad", cp(sib["addr_norm"].to_list(), sib["ad2"].to_list())),
                           pl.Series("r_nm", cp(sib["name_core"].to_list(), sib["nm2"].to_list())),
                           pl.Series("r_ad2_s1", cp(sib["ad2"].to_list(), sib["ad_s1"].to_list())))
    print(f"[{time.time()-T0:.0f}s] similarities done", flush=True)
    agg = sib.group_by("q_idx").agg(
        pl.len().alias("sib_n"),
        (pl.col("s3_2") != pl.col("is_s3")).sum().alias("sib_n_other_src"),
        pl.col("r_ad").max().alias("sib_max_ad"), pl.col("r_nm").max().alias("sib_max_nm"),
        pl.col("r_ad2_s1").max().alias("sib_best_ad_to_s1"),
        pl.col("p1_2").max().alias("sib_max_p1"))
    d = t.join(agg, on="q_idx", how="left").with_columns(pl.col("sib_n").fill_null(0),
        (pl.col("sib_best_ad_to_s1") - pl.col("f_ad_ratio")).alias("sib_ad_adv"),
        pl.when(pl.col("hn_absd1") == 0).then(pl.lit("d=0")).when(pl.col("hn_absd1").is_between(1, 2)).then(pl.lit("d=1-2"))
          .when(pl.col("hn_absd1") < 0).then(pl.lit("na")).otherwise(pl.lit("d>2")).alias("hn_grp"))
    lines = [f"rows {d.height:,}; y-rate {d['y'].mean():.3f}; rows with >=1 sibling {d.filter(pl.col('sib_n') > 0).height:,}"]
    specs = [("sib_n", [0, 1, 2, 3, 5, 10]), ("sib_n_other_src", [0, 1, 2, 3]), ("sib_max_ad", [60, 80, 90, 95, 99.9]),
             ("sib_max_nm", [60, 80, 90, 95, 99.9]), ("sib_ad_adv", [-20, -5, 0, 5, 20]), ("sib_max_p1", [0.1, 0.5, 0.9, 0.99])]
    for col, br in specs:
        g = d.with_columns(pl.col(col).cut(br, left_closed=True).alias("bin")) \
             .group_by("hn_grp", "bin").agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("y_rate")) \
             .filter(pl.col("n") >= 300).sort("hn_grp", "bin")
        lines.append(f"\n## {col}\n{g}")
    text = "\n".join(lines)
    (OUT / "diag_siblings.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"[{time.time()-T0:.0f}s] done")


if __name__ == "__main__":
    main()
