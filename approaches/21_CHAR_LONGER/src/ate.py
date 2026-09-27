"""Cross-fitted ADDRESS token odds (the address analogue of Approach 02's name token odds).

For a (query, S1) pair, the non-numeric address tokens present on only one side are looked up in a smoothed
P(true match | token appears query-only) table (and S1-only table), learned from labelled top-1 rows.
Cross-fitting mirrors stage2.build_training: rows of hash half 0 get odds learned on half 1 and vice versa;
validation rows get odds learned on the whole fold-A pool. Values: -1 unseen token, -2 no such token.
"""
import polars as pl

ATE_COLS = ["ate_q_min", "ate_q_max", "ate_q_unseen", "ate_s_min", "ate_s_max", "ate_s_unseen"]
SMOOTH = 20.0


def _diff_tokens(pairs):
    """pairs: _r, addr_norm, addr_norm_s -> (_r, qo list, so list) of non-numeric one-sided tokens."""
    nonnum = pl.element().filter(~pl.element().str.contains(r"^\d+$") & (pl.element() != ""))
    return pairs.select(
        "_r",
        pl.col("addr_norm").fill_null("").str.split(" ").list.set_difference(pl.col("addr_norm_s").fill_null("").str.split(" "))
          .list.eval(nonnum).alias("qo"),
        pl.col("addr_norm_s").fill_null("").str.split(" ").list.set_difference(pl.col("addr_norm").fill_null("").str.split(" "))
          .list.eval(nonnum).alias("so"))


def learn(pairs_y):
    """pairs_y: _r, y, addr_norm, addr_norm_s -> {'qo': DataFrame(tok, te), 'so': ...}"""
    d = _diff_tokens(pairs_y).join(pairs_y.select("_r", "y"), on="_r")
    prior = float(pairs_y["y"].mean())
    out = {}
    for side in ("qo", "so"):
        g = d.select("y", pl.col(side).alias("tok")).explode("tok").drop_nulls() \
             .group_by("tok").agg(pl.len().alias("n"), pl.col("y").sum().alias("k"))
        out[side] = g.select("tok", ((pl.col("k") + SMOOTH * prior) / (pl.col("n") + SMOOTH)).cast(pl.Float32).alias("te"))
    return out


def apply(pairs, te):
    """pairs: _r, addr_norm, addr_norm_s (sorted by _r) -> row-aligned ATE_COLS."""
    d = _diff_tokens(pairs)
    res = pairs.select("_r")
    for side, pre in (("qo", "ate_q"), ("so", "ate_s")):
        e = d.select("_r", pl.col(side).alias("tok")).explode("tok").drop_nulls().join(te[side], on="tok", how="left")
        a = e.group_by("_r").agg(pl.col("te").min().alias(f"{pre}_min"), pl.col("te").max().alias(f"{pre}_max"),
                                 pl.col("te").is_null().sum().cast(pl.Float32).alias(f"{pre}_unseen"),
                                 pl.len().alias("_n"))
        a = a.with_columns(pl.col(f"{pre}_min").fill_null(-1.0), pl.col(f"{pre}_max").fill_null(-1.0)).drop("_n")
        res = res.join(a, on="_r", how="left").with_columns(
            pl.col(f"{pre}_min").fill_null(-2.0), pl.col(f"{pre}_max").fill_null(-2.0), pl.col(f"{pre}_unseen").fill_null(0.0))
    return res.sort("_r").select([pl.col(c).cast(pl.Float32) for c in ATE_COLS])
