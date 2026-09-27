"""Read-only measurement of saved train candidates and Approach 02 validation decisions."""
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
CACHE = ROOT / "cache"
STAGE2 = ROOT / "approaches/02_STAGE2_CROSSFIT_RUNNERUP/cache/validate_val_scored_cf1_ru0_r8000.parquet"


def source_rows(source):
    raw = pl.read_parquet(CACHE / f"train_{source}.parquet", columns=["entity_id"])
    norm = pl.read_parquet(CACHE / f"train_{source}_norm.parquet",
                           columns=["country", "addr_missing", "has_indic"])
    return raw.hstack(norm)


def main():
    s1 = pl.read_parquet(CACHE / "train_source1.parquet", columns=["entity_id", "country"]).with_row_index("s1_idx")
    q = pl.concat([source_rows("source2"), source_rows("source3")]).with_row_index("q_idx")
    gt = pl.read_parquet(CACHE / "train_ground_truth.parquet")
    singleton = int((gt["matched_entity_ids"].fill_null("") == "").sum())
    multi = int((gt["matched_entity_ids"].fill_null("").str.count_matches(",") >= 1).sum())
    true = (gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
            .with_columns(pl.col("matched_entity_ids").str.split(","))
            .explode("matched_entity_ids", empty_as_null=True)
            .join(s1.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("s1_idx").alias("true_s1")), on="source1_entity_id")
            .join(q.select(pl.col("entity_id").alias("matched_entity_ids"), "q_idx", "country", "addr_missing", "has_indic"), on="matched_entity_ids")
            .select("q_idx", "true_s1", "country", "addr_missing", "has_indic", "source1_entity_id", "matched_entity_ids"))
    assert true["q_idx"].n_unique() == true.height, "one query maps to multiple S1: adjust definitions"
    print(f"ground truth: {true.height:,} links, {singleton:,} singleton S1, {multi:,} multi-match S1", flush=True)

    # Scan each saved chunk independently; retain only true pairs, never materialize all 103M candidates.
    hits = []
    for i, path in enumerate(sorted((CACHE / "train_cands").glob("*.parquet")), 1):
        c = pl.read_parquet(path, columns=["q_idx", "s1_idx", "rank"])
        found = (c.join(true.select("q_idx", "true_s1"), on="q_idx", how="inner")
                 .filter(pl.col("s1_idx") == pl.col("true_s1"))
                 .select("q_idx", "rank"))
        hits.append(found)
        if i % 10 == 0:
            print(f"candidate chunks {i}", flush=True)
    hits = pl.concat(hits).unique(subset=["q_idx"])
    true = true.join(hits, on="q_idx", how="left").with_columns(
        pl.col("rank").is_not_null().alias("top10"), (pl.col("rank") == 1).fill_null(False).alias("retrieval_rank1"))
    best = pl.read_parquet(CACHE / "train_best.parquet", columns=["q_idx", "s1_idx", "p1", "p2"])
    true = true.join(best, on="q_idx", how="left").with_columns(
        (pl.col("s1_idx") == pl.col("true_s1")).fill_null(False).alias("model_top1"),
        ((pl.col("source1_entity_id").str.slice(3).cast(pl.Int64) % 5) == 0).alias("is_val"))
    val_scored = pl.read_parquet(STAGE2, columns=["q_idx", "p_stage2"])
    true = true.join(val_scored, on="q_idx", how="left").with_columns(
        pl.coalesce("p_stage2", "p1").alias("p_final"))
    true = true.with_columns(
        (pl.col("model_top1") & (pl.col("p_final") >= 0.7)).alias("accepted_070"),
        (pl.col("model_top1") & (pl.col("p_final") >= 0.5)).alias("accepted_050"))

    out = {"definitions": {"top1": "baseline LightGBM highest-scored S1, not cosine rank 1",
                           "stage2": "Approach 02 validation model for fold B in band; baseline p1 outside band; threshold 0.70",
                           "note": "Stage-2 validation scores exist for fold B only; full-corpus Stage-2 recall is unavailable."},
           "ground_truth_links": true.height, "queries_evaluated": true["q_idx"].n_unique(),
           "singleton_s1": singleton, "multi_match_s1": multi, "total_s1": gt.height,
           "groups": {}}
    for name, d in [("all", true), ("validation", true.filter(pl.col("is_val"))),
                    ("US", true.filter(pl.col("country") == "US")),
                    ("India", true.filter(pl.col("country") == "India")),
                    ("address_present", true.filter(~pl.col("addr_missing"))),
                    ("address_missing", true.filter(pl.col("addr_missing")))]:
        g = {"links": d.height, "top10": int(d["top10"].sum()),
             "retrieval_rank1": int(d["retrieval_rank1"].sum()), "model_top1": int(d["model_top1"].sum())}
        g["top10_recall"] = g["top10"] / g["links"] if g["links"] else None
        g["top1_recall"] = g["model_top1"] / g["links"] if g["links"] else None
        g["top10_to_top1_loss"] = g["top10"] - g["model_top1"]
        if name == "validation":
            g["stage2_scored_band_rows"] = int(d["p_stage2"].is_not_null().sum())
            for t in ("070", "050"):
                threshold = 0.7 if t == "070" else 0.5
                g[f"accepted_{t}"] = int(d[f"accepted_{t}"].sum())
                g[f"accepted_recall_{t}"] = g[f"accepted_{t}"] / g["links"]
                g[f"top1_to_stage2_loss_{t}"] = g["model_top1"] - g[f"accepted_{t}"]
                g[f"in_band_stage2_rejected_correct_top1_{t}"] = d.filter(
                    pl.col("model_top1") & pl.col("p_stage2").is_not_null() & (pl.col("p_stage2") < threshold)).height
                g[f"outside_band_baseline_rejected_correct_top1_{t}"] = d.filter(
                    pl.col("model_top1") & pl.col("p_stage2").is_null() & (pl.col("p1") < threshold)).height
        out["groups"][name] = g

    # A directly comparable validation funnel by country and address status.
    v = true.filter(pl.col("is_val"))
    for name, d in [("val_US", v.filter(pl.col("country") == "US")),
                    ("val_India", v.filter(pl.col("country") == "India")),
                    ("val_address_present", v.filter(~pl.col("addr_missing"))),
                    ("val_address_missing", v.filter(pl.col("addr_missing")))]:
        n = d.height
        out["groups"][name] = {"links": n, "top10": int(d["top10"].sum()),
                               "model_top1": int(d["model_top1"].sum()),
                               "accepted_070": int(d["accepted_070"].sum()),
                               "top10_recall": float(d["top10"].mean()),
                               "top1_recall": float(d["model_top1"].mean()),
                               "accepted_recall_070": float(d["accepted_070"].mean())}

    misses = true.filter(~pl.col("top10")).select("q_idx", "true_s1", "country", "addr_missing", "has_indic")
    # Compact deterministic sample and aggregate flags for diagnosis. Never duplicate the full training table.
    sample = misses.sort("q_idx").sample(n=min(200, misses.height), seed=42, shuffle=True).sort("q_idx")
    sample.write_csv(OUT / "top10_miss_sample.csv")
    # Fixed, balanced diagnostic sample for the targeted top-40 rerun.
    top40_sample = pl.concat([
        misses.filter(pl.col("country") == country).sort("q_idx").sample(n=1000, seed=42, shuffle=True)
        for country in ("US", "India")
    ]).sort("country", "q_idx")
    top40_sample.write_csv(OUT / "top40_sample_keys.csv")
    out["misses"] = {"total": misses.height,
                     "address_missing": int(misses["addr_missing"].sum()),
                     "has_indic": int(misses["has_indic"].sum()),
                     "US": misses.filter(pl.col("country") == "US").height,
                     "India": misses.filter(pl.col("country") == "India").height}
    out["top40_full_corpus"] = "NOT AVAILABLE: approximate top-40 lists were not cached"
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
