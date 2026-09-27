"""Test-shift check for the group-aware rule (conf 0.9, alone 0.70, group 0.85) using Exp 10 R1 scores:
validation (fold-A model R1 on fold B) vs test (FINAL_HN_ONLY final scores). Read-only."""
import polars as pl
import common
from density import DensityEval


def apply(b, conf=0.9, ta=0.70, tg=0.85):
    b = b.with_columns(((pl.col("p_final") >= conf).cast(pl.Int32).sum().over("s1_idx") - (pl.col("p_final") >= conf).cast(pl.Int32)).alias("n"))
    return b.with_columns(pl.when(pl.col("n") > 0).then(pl.lit(tg)).otherwise(pl.lit(ta)).alias("thr"))


def summary(name, b):
    b = apply(b)
    u7, u8 = (b["p_final"] >= 0.7).sum(), (b["p_final"] >= 0.8).sum()
    pol = (b["p_final"] >= b["thr"]).sum()
    mid = b.filter(pl.col("p_final").is_between(0.7, 0.85, closed="left"))
    print(f"{name}: records {b.height:,}; accepted uniform0.7 {u7:,} | uniform0.8 {u8:,} ({100*(u8/u7-1):+.3f}%) | policy {pol:,} ({100*(pol/u7-1):+.3f}% vs 0.7); "
          f"rows with p in [0.70,0.85): {mid.height:,} of which grouped {100*(mid['n']>0).mean():.1f}%", flush=True)


ev = DensityEval()
v = ev.frame(pl.read_parquet(common.ROOT / "approaches/10_STAGE2_ADDRESS_SIGNALS/cache/val_scored_R1_hn.parquet")).filter(pl.col("q_is_val"))
summary("VALIDATION fold-B records (R1 fold-A model)", v.select("s1_idx", "p_final"))
t = pl.read_parquet(common.ROOT / "approaches/FINAL_HN_ONLY/cache/test_final_scored.parquet", columns=["q_idx", "s1_idx", "p_final"])
c = pl.concat([pl.read_parquet(common.ROOT / f"cache/test_{s}_norm.parquet", columns=["country"]) for s in ("source2", "source3")]).with_row_index("q_idx")
summary("TEST all (FINAL_HN_ONLY)", t)
for cty in ("US", "India", "France"):
    summary(f"TEST {cty}", t.join(c, on="q_idx").filter(pl.col("country") == cty))
