"""Exp 20 diagnostic: anchor-based candidate recovery for address-present records. Read-only."""
import sys
from pathlib import Path
import polars as pl
from rapidfuzz import fuzz, process
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "business_entity_resolution/src"))
from train import label_table, query_folds  # noqa
sys.stdout.reconfigure(encoding="utf-8")
s1, q, lab, tp = label_table()
qf = query_folds(lab, s1)
nm = pl.concat([pl.read_parquet(ROOT / f"cache/train_{s}_norm.parquet", columns=["name_core", "addr_norm", "addr_missing"]) for s in ("source2", "source3")]).with_row_index("q_idx")
sn = pl.read_parquet(ROOT / "cache/train_source1_norm.parquet", columns=["name_core"]).with_row_index("s1_idx")
best = pl.read_parquet(ROOT / "cache/train_best.parquet", columns=["q_idx", "s1_idx", "p1"])
r6 = pl.read_parquet(ROOT / "approaches/19_ANCHOR_SIBLINGS/cache/val_scored_R6_s42.parquet")
d = best.join(qf, on="q_idx", how="left").join(nm, on="q_idx").join(r6, on="q_idx", how="left").with_columns(pl.coalesce("p_stage2", "p1").alias("pf"))
d = d.with_columns(((pl.col("pf") >= 0.7) & (pl.col("s1_idx") == pl.col("true_s1"))).fill_null(False).alias("ok"))
anc = d.filter((pl.col("p1") >= 0.9) & ~pl.col("addr_missing") & (pl.col("addr_norm") != "")).select(pl.col("q_idx").alias("q2"), pl.col("s1_idx").alias("x"), "addr_norm")
amb = anc.group_by("addr_norm").agg(pl.col("x").n_unique().alias("nx"))
anc = anc.join(amb.filter(pl.col("nx") == 1), on="addr_norm").select("q2", "x", "addr_norm").unique(["q2"])
Q = d.filter(pl.col("q_is_val").fill_null(False) & ~pl.col("addr_missing"))
j = Q.select("q_idx", "s1_idx", "true_s1", "ok", "addr_norm", "name_core").join(anc, on="addr_norm").filter(pl.col("q2") != pl.col("q_idx"))
j = j.group_by("q_idx").agg(pl.first("s1_idx"), pl.first("true_s1"), pl.first("ok"), pl.first("name_core"), pl.col("x").first()).filter(pl.col("x") != pl.col("s1_idx"))
j = j.join(sn.rename({"s1_idx": "x", "name_core": "nm_x"}), on="x")
j = j.with_columns(pl.Series("nsim", process.cpdist(j["name_core"].to_list(), j["nm_x"].to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)),
                   (pl.col("x") == pl.col("true_s1")).alias("prop_ok"))
print(f"fold-B address-present records: {Q.height:,}; currently not ok (matched): {Q.filter(~pl.col('ok') & pl.col('true_s1').is_not_null()).height:,}")
print(f"proposals (x != current top-1): {j.height:,}; accuracy {j['prop_ok'].mean():.3f}")
g = j.with_columns(pl.col("nsim").cut([30, 50, 70, 85, 95], left_closed=True).alias("nsim_bin")).group_by("nsim_bin").agg(
    pl.len().alias("n"), pl.col("prop_ok").mean().round(3).alias("acc"),
    (pl.col("prop_ok") & ~pl.col("ok")).sum().alias("would_fix"), (~pl.col("prop_ok") & pl.col("ok")).sum().alias("would_break_if_forced")).sort("nsim_bin")
print(g)
