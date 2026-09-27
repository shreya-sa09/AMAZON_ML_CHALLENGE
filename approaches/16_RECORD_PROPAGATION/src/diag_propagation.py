"""Exp 16 diagnostic: can a record WITHOUT an address inherit the S1 of another record sharing its exact noisy raw name?
Measured on training ground truth; label-free proposal uses only other records' Stage-1 top-1 and scores. Read-only."""
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(ROOT / "business_entity_resolution/src"))
from train import label_table, query_folds  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
T0 = time.time()
lines = []


def log(m):
    print(f"[{time.time() - T0:5.0f}s] {m}", flush=True)
    lines.append(m)


def main():
    s1, q, lab, tp = label_table()
    qf = query_folds(lab, s1)
    raw = pl.concat([pl.read_parquet(ROOT / f"cache/train_{s}.parquet", columns=["business_name"]).with_columns(pl.lit(s == "source3").alias("is_s3"))
                     for s in ("source2", "source3")]).with_row_index("q_idx")
    nm = pl.concat([pl.read_parquet(ROOT / f"cache/train_{s}_norm.parquet", columns=["addr_missing", "name_core", "country"]) for s in ("source2", "source3")]).with_row_index("q_idx")
    best = pl.read_parquet(ROOT / "cache/train_best.parquet", columns=["q_idx", "s1_idx", "p1"])
    d = raw.join(nm, on="q_idx").join(qf, on="q_idx", how="left").join(best, on="q_idx", how="left")
    d = d.with_columns(pl.col("business_name").fill_null(""))
    # current decisions for fold-B (FINAL_HN_ONLY validation model R1): accepted & correct?
    r1 = pl.read_parquet(ROOT / "approaches/10_STAGE2_ADDRESS_SIGNALS/cache/val_scored_R1_hn.parquet")
    d = d.join(r1, on="q_idx", how="left").with_columns(pl.coalesce("p_stage2", "p1").alias("p_final"))
    d = d.with_columns(((pl.col("p_final") >= 0.7) & (pl.col("s1_idx") == pl.col("true_s1"))).fill_null(False).alias("ok_now"))
    na = d.filter(pl.col("addr_missing") & pl.col("true_s1").is_not_null())
    log(f"no-address matched records: {na.height:,}; currently correct+accepted (all folds, approx): {na['ok_now'].mean():.3f}")

    for key in ("business_name", "name_core"):
        # partners: OTHER records with the same key that HAVE an address
        part = d.filter(~pl.col("addr_missing")).select(pl.col(key).alias("k"), pl.col("q_idx").alias("q2"), pl.col("true_s1").alias("t2"),
                                                           pl.col("s1_idx").alias("s1_2"), pl.col("p_final").alias("p2"), pl.col("is_s3").alias("s3_2"))
        j = na.select("q_idx", pl.col(key).alias("k"), "true_s1", "ok_now", "is_s3", "q_is_val").filter(pl.col("k") != "").join(part, on="k")
        g = j.group_by("q_idx").agg(pl.first("true_s1"), pl.first("ok_now"), pl.first("q_is_val"), pl.len().alias("n_part"),
                                    pl.col("t2").n_unique().alias("n_true_s1"), (pl.col("t2") == pl.col("true_s1")).any().alias("any_true"),
                                    (pl.col("s3_2") != pl.col("is_s3")).any().alias("other_src"),
                                    # label-free proposal: confident partners' predicted S1
                                    pl.col("s1_2").filter(pl.col("p2") >= 0.9).unique().alias("props"))
        g = g.with_columns(pl.col("props").list.len().alias("n_props"))
        uniq = g.filter(pl.col("n_props") == 1).with_columns(pl.col("props").list.first().alias("prop"))
        uniq = uniq.with_columns((pl.col("prop") == pl.col("true_s1")).alias("prop_ok"))
        log(f"\n== key={key}")
        log(f"no-address records with >=1 address-bearing partner: {g.height:,} ({g.height / na.height:.3f}); "
            f"partner from other source: {g['other_src'].mean():.3f}")
        log(f"partners' TRUE S1 all equal to ours (purity): {(g['n_true_s1'] == 1).mean():.3f}; any partner shares our true S1: {g['any_true'].mean():.3f}")
        log(f"label-free unique confident proposal: {uniq.height:,} records; proposal correct: {uniq['prop_ok'].mean():.4f}")
        fb = uniq.filter(pl.col("q_is_val").fill_null(False))
        log(f"  fold B: proposals {fb.height:,}; correct {fb['prop_ok'].sum():,} ({fb['prop_ok'].mean():.4f}); "
            f"of which currently NOT correct/accepted: {fb.filter(~pl.col('ok_now') & pl.col('prop_ok')).height:,} would be FIXED; "
            f"wrong proposals on currently-ok records: {fb.filter(pl.col('ok_now') & ~pl.col('prop_ok')).height:,}; "
            f"wrong proposals on currently-not-ok: {fb.filter(~pl.col('ok_now') & ~pl.col('prop_ok')).height:,}")
        dist = g.group_by(pl.col("n_part").clip(0, 5).alias("n_part")).agg(pl.len(), pl.col("any_true").mean().round(3).alias("any_true"),
                                                                        (pl.col("n_true_s1") == 1).mean().round(3).alias("pure")).sort("n_part")
        log(f"by #partners:\n{dist}")
    (OUT / "diag_propagation.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
