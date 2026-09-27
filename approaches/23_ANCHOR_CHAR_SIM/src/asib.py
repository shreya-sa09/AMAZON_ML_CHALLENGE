"""Confident-anchor sibling features. Anchors of a (query q, S1 x) row = OTHER S2/S3 records whose Stage-1 top-1 is x
with p1 >= ANCHOR_P. Decoys are rarely confident, so unlike Exp 13 (all siblings) these should not scale with decoy density."""
import re

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

ASIB_COLS = ["asib_n", "asib_max_ad", "asib_has_ad_exact", "asib_max_nm", "asib_has_nm_exact", "asib_hn_eq"]
ANCHOR_P = 0.9
CAP = 30


def _cp(a, b):
    return process.cpdist(a, b, scorer=fuzz.ratio, workers=-1, dtype=np.float32)


def asib_features(rows, best, qn):
    """rows: q_idx, s1_idx (order kept). best: q_idx, s1_idx, p1 for ALL queries of the split.
    qn: q_idx, name_core, addr_norm, addr_nums."""
    r = rows.select("q_idx", "s1_idx").with_row_index("_r")
    anc = best.filter(pl.col("p1") >= ANCHOR_P).select(pl.col("q_idx").alias("q2"), "s1_idx")
    s = r.join(anc, on="s1_idx").filter(pl.col("q2") != pl.col("q_idx")).sort("_r", "q2")
    s = s.with_columns(pl.int_range(pl.len()).over("_r").alias("_k")).filter(pl.col("_k") < CAP).drop("_k")
    s = s.join(qn.select(pl.col("q_idx").alias("q2"), pl.col("name_core").alias("nm2"), pl.col("addr_norm").alias("ad2"),
                         pl.col("addr_nums").alias("nu2")), on="q2") \
         .join(qn.select("q_idx", "name_core", "addr_norm", "addr_nums"), on="q_idx")
    s = s.with_columns([pl.col(c).fill_null("") for c in ("nm2", "ad2", "nu2", "name_core", "addr_norm", "addr_nums")])
    both = (s["addr_norm"] != "") & (s["ad2"] != "")
    s = s.with_columns(pl.Series("r_ad", _cp(s["addr_norm"].to_list(), s["ad2"].to_list())), both.alias("both"),
                       pl.Series("r_nm", _cp(s["name_core"].to_list(), s["nm2"].to_list())),
                       (pl.col("addr_nums").str.extract(r"^(\d+)") == pl.col("nu2").str.extract(r"^(\d+)")).fill_null(False).alias("hn_eq"))
    s = s.with_columns(pl.when(pl.col("both")).then(pl.col("r_ad")).otherwise(None).alias("r_ad"))
    a = s.group_by("_r").agg(pl.len().cast(pl.Float32).alias("asib_n"), pl.col("r_ad").max().alias("asib_max_ad"),
                             (pl.col("r_ad") >= 99.99).any().cast(pl.Float32).alias("asib_has_ad_exact"),
                             pl.col("r_nm").max().alias("asib_max_nm"),
                             (pl.col("r_nm") >= 99.99).any().cast(pl.Float32).alias("asib_has_nm_exact"),
                             pl.col("hn_eq").any().cast(pl.Float32).alias("asib_hn_eq"))
    out = r.join(a, on="_r", how="left").sort("_r").with_columns(pl.col("asib_n").fill_null(0.0))
    return out.select([pl.col(c).cast(pl.Float32).fill_null(-1.0) for c in ASIB_COLS])
