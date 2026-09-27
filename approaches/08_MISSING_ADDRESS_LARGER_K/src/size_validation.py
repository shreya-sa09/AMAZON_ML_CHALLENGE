"""Count read-only missing-address queries in the established validation fold."""
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)


def main():
    s1 = pl.read_parquet(ROOT / "cache/train_source1.parquet", columns=["entity_id", "country"]).with_row_index("s1_idx")
    s1 = s1.with_columns((pl.col("entity_id").str.slice(3).cast(pl.Int64) % 5 == 0).alias("is_val"))
    q = pl.concat([pl.read_parquet(ROOT / f"cache/train_{src}.parquet", columns=["entity_id"])
                   .hstack(pl.read_parquet(ROOT / f"cache/train_{src}_norm.parquet", columns=["addr_missing", "country"]))
                   for src in ("source2", "source3")]).with_row_index("q_idx")
    q = q.filter(pl.col("addr_missing"))
    gt = pl.read_parquet(ROOT / "cache/train_ground_truth.parquet")
    matched = (gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
               .with_columns(pl.col("matched_entity_ids").str.split(","))
               .explode("matched_entity_ids", empty_as_null=True)
               .join(s1.select(pl.col("entity_id").alias("source1_entity_id"), "is_val"), on="source1_entity_id")
               .join(q.select(pl.col("entity_id").alias("matched_entity_ids"), "q_idx"), on="matched_entity_ids")
               .select("q_idx", pl.col("is_val").alias("true_is_val")))
    best = pl.read_parquet(ROOT / "cache/train_best.parquet", columns=["q_idx", "s1_idx"])
    cohort = (q.join(matched, on="q_idx", how="left")
              .join(best.join(s1.select("s1_idx", pl.col("is_val").alias("top1_is_val")), on="s1_idx"), on="q_idx")
              .with_columns(pl.coalesce("true_is_val", "top1_is_val").alias("q_is_val")))
    val = cohort.filter(pl.col("q_is_val"))
    result = {"all_missing_address_queries": cohort.height, "validation_missing_address_queries": val.height,
              "validation_matched": int(val["true_is_val"].is_not_null().sum()),
              "validation_unmatched": int(val["true_is_val"].is_null().sum()),
              "by_country": val.group_by("country").agg(pl.len().alias("n"),
                              pl.col("true_is_val").is_not_null().sum().alias("matched")).to_dicts()}
    (OUT / "validation_size.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
