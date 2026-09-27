"""Label-free collective 'sibling' features. Siblings of a (query q, S1 x) row = other S2/S3 records whose Stage-1
top-1 (train_best / test_best) is the same x. Uses only normalized text of the records themselves, never labels and
never sibling model scores (training-side stage-1 scores are in-sample for 10% of records)."""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

SIB_COLS = ["sib_n", "sib_n_other_src", "sib_max_ad", "sib_n_ad_exact", "sib_max_nm", "sib_n_nm_exact",
            "sib_n_both_exact", "sib_ad_adv"]
CAP = 50  # similarity computed against at most CAP siblings per row (counts use the full degree)


def _cp(a, b):
    return process.cpdist(a, b, scorer=fuzz.ratio, workers=-1, dtype=np.float32)


def sib_features(rows, best, qn, sn, f_ad_ratio):
    """rows: q_idx, s1_idx (row order kept). best: q_idx, s1_idx for ALL queries of the split.
    qn: q_idx, name_core, addr_norm, is_s3. sn: s1_idx, addr_norm. f_ad_ratio: Series aligned with rows."""
    r = rows.select("q_idx", "s1_idx").with_row_index("_r")
    deg = best.group_by("s1_idx").agg(pl.len().alias("_deg"))
    s = r.join(best.select(pl.col("q_idx").alias("q2"), "s1_idx"), on="s1_idx").filter(pl.col("q2") != pl.col("q_idx")) \
         .sort("_r", "q2")
    s = s.with_columns(pl.int_range(pl.len()).over("_r").alias("_k")).filter(pl.col("_k") < CAP).drop("_k")
    s = s.join(qn.select(pl.col("q_idx").alias("q2"), pl.col("name_core").alias("nm2"), pl.col("addr_norm").alias("ad2"),
                         pl.col("is_s3").alias("s3_2")), on="q2") \
         .join(qn.select("q_idx", "name_core", "addr_norm", "is_s3"), on="q_idx") \
         .join(sn.select("s1_idx", pl.col("addr_norm").alias("ad_s1")), on="s1_idx")
    s = s.with_columns([pl.col(c).fill_null("") for c in ("nm2", "ad2", "name_core", "addr_norm", "ad_s1")])
    both_ad = (s["addr_norm"] != "") & (s["ad2"] != "")
    r_ad = _cp(s["addr_norm"].to_list(), s["ad2"].to_list())
    r_ad2s1 = _cp(s["ad2"].to_list(), s["ad_s1"].to_list())
    s = s.with_columns(pl.Series("r_ad", r_ad), pl.Series("r_nm", _cp(s["name_core"].to_list(), s["nm2"].to_list())),
                       pl.Series("r_ad2s1", r_ad2s1), both_ad.alias("both_ad"))
    s = s.with_columns(pl.when(pl.col("both_ad")).then(pl.col("r_ad")).otherwise(None).alias("r_ad"),
                       pl.when(pl.col("ad2") != "").then(pl.col("r_ad2s1")).otherwise(None).alias("r_ad2s1"))
    a = s.group_by("_r").agg(
        (pl.col("s3_2") != pl.col("is_s3")).sum().cast(pl.Float32).alias("sib_n_other_src"),
        pl.col("r_ad").max().alias("sib_max_ad"),
        (pl.col("r_ad") >= 99.99).sum().cast(pl.Float32).alias("sib_n_ad_exact"),
        pl.col("r_nm").max().alias("sib_max_nm"),
        (pl.col("r_nm") >= 99.99).sum().cast(pl.Float32).alias("sib_n_nm_exact"),
        ((pl.col("r_nm") >= 99.99) & (pl.col("r_ad") >= 99.99)).sum().cast(pl.Float32).alias("sib_n_both_exact"),
        pl.col("r_ad2s1").max().alias("_best_ad_s1"))
    out = r.join(deg, on="s1_idx", how="left").join(a, on="_r", how="left").sort("_r").with_columns(
        (pl.col("_deg").fill_null(1) - 1).cast(pl.Float32).alias("sib_n"),
        pl.Series("_f_ad", np.asarray(f_ad_ratio, dtype=np.float32)))
    out = out.with_columns((pl.col("_best_ad_s1") - pl.col("_f_ad")).alias("sib_ad_adv"))
    return out.select([pl.col(c).cast(pl.Float32).fill_null(-1.0) for c in SIB_COLS])
