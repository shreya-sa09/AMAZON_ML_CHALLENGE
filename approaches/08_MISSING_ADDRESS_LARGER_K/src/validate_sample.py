"""End-to-end validation with K80/20 only on a fixed missing-address query sample.

All source caches and models are read-only. The full baseline validation decisions are
reused; only the selected queries are reretrieved, refeaturized, and rescored.
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[3]
APP = Path(__file__).resolve().parents[1]
OUT = APP / "results"
OUT.mkdir(exist_ok=True)
os.environ["ER_CACHE_DIR"] = str(ROOT / "cache")
os.environ["ER_OUTPUT_DIR"] = str(OUT)
sys.path.insert(0, str(ROOT / "business_entity_resolution/src"))
sys.path.insert(0, str(ROOT / "approaches/02_STAGE2_CROSSFIT_RUNNERUP/src"))
import stage2  # noqa: E402
from build_features import load_norm_tables  # noqa: E402
from features import FEATURES, compute_features, s1_extra_table  # noqa: E402
from metrics import macro_f05  # noqa: E402
from retrieval import CountryIndex  # noqa: E402
from train import best_per_query  # noqa: E402

T0 = time.time()


def log(message):
    print(f"[{time.time()-T0:6.0f}s] {message}", flush=True)


def flags():
    return pl.concat([
        pl.read_parquet(ROOT / f"cache/train_{src}_norm.parquet", columns=["country", "addr_missing"])
        for src in ("source2", "source3")
    ]).with_row_index("q_idx")


def old_candidates(keys):
    parts = []
    for path in sorted((ROOT / "cache/train_cands").glob("*.parquet")):
        parts.append(pl.read_parquet(path).join(keys, on="q_idx", how="semi"))
    return pl.concat(parts)


def retrieve(cohort, s1, qn, s1n):
    src = s1.select("s1_idx", "entity_id", "country").join(
        s1n.select("s1_idx", "name_core", "addr_norm"), on="s1_idx")
    queries = cohort.select("q_idx", "country").join(qn.select("q_idx", "name_core", "addr_norm"), on="q_idx")
    out = []
    for country in ("US", "India"):
        sc = src.filter(pl.col("country") == country)
        qc = queries.filter(pl.col("country") == country).sort("q_idx")
        log(f"retrieval {country}: index {sc.height:,} S1; query {qc.height:,}")
        index = CountryIndex(sc)
        Qn, Qa = index.encode(qc)
        found = index.candidates(Qn, Qa, K=80, k=20)
        qmap, smap = qc["q_idx"].to_numpy(), sc["s1_idx"].to_numpy()
        out.append(found.with_columns(
            pl.Series("q_idx", qmap[found["q_row"].to_numpy()], dtype=pl.UInt32),
            pl.Series("s1_idx", smap[found["s1_row"].to_numpy()], dtype=pl.UInt32),
        ).select("q_idx", "s1_idx", "cos_name", "cos_addr", "cos_comb", "rank"))
        del index, Qn, Qa
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pl.concat(out)


def score_candidates(candidates, cohort):
    log(f"loading baseline feature inputs for {candidates.height:,} union pairs")
    s1, q = load_norm_tables("train")
    extra = s1_extra_table(s1, ROOT / "cache/train_cands")
    model = lgb.Booster(model_file=str(ROOT / "cache/lgbm_model.txt"))
    ids = cohort["q_idx"].sort()
    best_blocks = []
    n_blocks = (len(ids) + 999) // 1000
    for bi, start in enumerate(range(0, len(ids), 1000), 1):
        keys = ids.slice(start, 1000)
        c = candidates.filter(pl.col("q_idx").is_in(keys))
        f = compute_features(c, q, s1, extra)
        p = model.predict(f.select(FEATURES).to_numpy().astype(np.float32), num_threads=16).astype(np.float32)
        scored = f.select("q_idx", "s1_idx").with_columns(pl.Series("p", p))
        b = best_per_query(scored).rename({"p1": "p1", "p2": "p2"})
        best_blocks.append(b.join(f, on=["q_idx", "s1_idx"]).select(
            "q_idx", "s1_idx", "p1", "p2", *[pl.col(x).alias("f_" + x) for x in FEATURES]))
        if bi % 2 == 0 or bi == n_blocks:
            log(f"features/model: {bi}/{n_blocks} blocks complete")
    del s1, q, extra, model
    gc.collect()
    return pl.concat(best_blocks).sort("q_idx")


def stage2_score(new_best, best, qn, s1n):
    log("learning Approach 02 validation token odds from fold A")
    te = stage2.learn_te(best, qn, s1n, ~pl.col("q_is_val") & pl.col("oos"))
    vocab = stage2.Vocab(s1n, s1n.height)
    new_best = new_best.with_columns((pl.col("p1") - pl.col("p2")).alias("gap"))
    in_band = new_best.filter(pl.col("p1").is_between(0.002, 0.995)).with_row_index("_r")
    pair = (in_band.select("_r", "q_idx", "s1_idx")
            .join(qn.select("q_idx", "name_core", "addr_norm", "addr_nums"), on="q_idx")
            .join(s1n.select("s1_idx", "name_core", "addr_norm", "addr_nums"), on="s1_idx", suffix="_s")
            .sort("_r"))
    e = stage2.edit_features(pair, vocab, *te)
    x = pl.concat([in_band.select("p1", "p2", "gap", *["f_" + z for z in FEATURES]), e], how="horizontal")
    model = lgb.Booster(model_file=str(ROOT / "approaches/02_STAGE2_CROSSFIT_RUNNERUP/cache/stage2_validation_model_cf1_ru0_r8000.txt"))
    assert x.columns == model.feature_name(), "Stage-2 feature order differs from saved model"
    p = model.predict(x.to_numpy().astype(np.float32), num_threads=16).astype(np.float32)
    scored = in_band.select("q_idx").with_columns(pl.Series("p_stage2", p))
    log(f"Stage-2 scored {scored.height:,} in-band new Top-1 rows")
    return new_best.join(scored, on="q_idx", how="left").with_columns(
        pl.coalesce("p_stage2", "p1").alias("p_final"))


def to_ids(rows, s1, q):
    return (rows.join(s1.select("s1_idx", pl.col("entity_id").alias("s1_id")), on="s1_idx")
            .join(q.select("q_idx", pl.col("entity_id").alias("m_id")), on="q_idx")
            .select("s1_id", "m_id"))


def main():
    s1, q, tp, best = stage2.train_rows()
    qn, s1n = stage2.norm_tables("train")
    eligible = best.join(flags(), on="q_idx").filter(pl.col("q_is_val") & pl.col("addr_missing"))
    cohort = pl.concat([eligible.filter(pl.col("country") == country).sort("q_idx")
                        .sample(n=5000, seed=42, shuffle=True)
                        for country in ("US", "India")]).sort("q_idx")
    log(f"fixed validation cohort: {cohort.height:,} missing-address queries, "
        f"{cohort['true_s1'].is_not_null().sum():,} matched")
    cohort.select("q_idx", "country", "true_s1").write_csv(OUT / "validation_sample_keys.csv")
    keys = cohort.select("q_idx")
    new = retrieve(cohort, s1, qn, s1n)
    old = old_candidates(keys)
    old_keys = old.select("q_idx", "s1_idx").unique()
    additions = new.join(old_keys, on=["q_idx", "s1_idx"], how="anti")
    combined = pl.concat([old, additions]).unique(subset=["q_idx", "s1_idx"], keep="first")
    combined = combined.with_columns(pl.col("cos_comb").rank("ordinal", descending=True).over("q_idx").cast(pl.Int16).alias("rank"))
    log(f"saved {old.height:,} pairs + {additions.height:,} additions = {combined.height:,} union pairs")

    truth_keys = cohort.filter(pl.col("true_s1").is_not_null()).select(
        "q_idx", pl.col("true_s1").alias("s1_idx"))
    old_true = old_keys.join(truth_keys, on=["q_idx", "s1_idx"]).height
    new_true = combined.select("q_idx", "s1_idx").join(truth_keys, on=["q_idx", "s1_idx"]).height
    assert new_true >= old_true
    new_top = score_candidates(combined, cohort)
    new_top = stage2_score(new_top, best, qn, s1n)
    old_top = cohort.select("q_idx", "s1_idx", "p1", "true_s1").join(
        pl.read_parquet(ROOT / "approaches/02_STAGE2_CROSSFIT_RUNNERUP/cache/validate_val_scored_cf1_ru0_r8000.parquet",
                        columns=["q_idx", "p_stage2"]), on="q_idx", how="left").with_columns(
            pl.coalesce("p_stage2", "p1").alias("p_final"))
    new_top = new_top.join(cohort.select("q_idx", "true_s1"), on="q_idx")
    tp_old = old_top.filter((pl.col("s1_idx") == pl.col("true_s1")) & (pl.col("p_final") >= 0.7)).height
    tp_new = new_top.filter((pl.col("s1_idx") == pl.col("true_s1")) & (pl.col("p_final") >= 0.7)).height
    log(f"sample true accepted: {tp_old:,} old, {tp_new:,} new")

    log("assembling full validation decisions for a directly comparable macro F0.5")
    val_scores = pl.read_parquet(ROOT / "approaches/02_STAGE2_CROSSFIT_RUNNERUP/cache/validate_val_scored_cf1_ru0_r8000.parquet",
                                 columns=["q_idx", "p_stage2"])
    all_old = best.select("q_idx", "s1_idx", "p1", "q_is_val").join(val_scores, on="q_idx", how="left")
    all_old = all_old.with_columns(pl.coalesce("p_stage2", "p1").alias("p_final"))
    pred_old = all_old.filter(pl.col("p_final") >= 0.7).select("q_idx", "s1_idx")
    pred_new = pl.concat([pred_old.join(keys, on="q_idx", how="anti"),
                          new_top.filter(pl.col("p_final") >= 0.7).select("q_idx", "s1_idx")])
    val_ids = s1.filter(pl.col("is_val"))["entity_id"]
    old_metric = macro_f05(to_ids(pred_old, s1, q), tp, val_ids)
    new_metric = macro_f05(to_ids(pred_new, s1, q), tp, val_ids)
    expected = 0.9856897798505745
    assert abs(old_metric["macro_f05"] - expected) < 1e-8, "baseline validation assembly does not reproduce Approach 02"
    old_per, new_per = old_metric["per_entity"], new_metric["per_entity"]
    old_single_fp = old_per.filter((pl.col("n_true") == 0) & (pl.col("n_pred") > 0)).height
    new_single_fp = new_per.filter((pl.col("n_true") == 0) & (pl.col("n_pred") > 0)).height
    matched = truth_keys.height
    summary = {
        "variant": "missing-address queries only: approximate Top-80, exact Top-20, union saved Top-10",
        "sample": "fixed 5000 validation-fold missing-address queries per country; other queries unchanged",
        "sample_queries": cohort.height, "sample_matched_queries": matched,
        "saved_candidate_pairs": old.height, "added_candidate_pairs": additions.height,
        "added_true_pairs": new_true - old_true,
        "added_negative_pairs": additions.height - (new_true - old_true),
        "candidate_recall_baseline": old_true / matched, "candidate_recall_new": new_true / matched,
        "candidate_true_baseline": old_true, "candidate_true_new": new_true,
        "sample_baseline_correct_top1": old_top.filter(pl.col("s1_idx") == pl.col("true_s1")).height,
        "sample_new_correct_top1": new_top.filter(pl.col("s1_idx") == pl.col("true_s1")).height,
        "sample_baseline_accepted_true": tp_old, "sample_new_accepted_true": tp_new,
        "sample_top1_changed": old_top.select("q_idx", pl.col("s1_idx").alias("old_s1")).join(
            new_top.select("q_idx", pl.col("s1_idx").alias("new_s1")), on="q_idx").filter(
                pl.col("old_s1") != pl.col("new_s1")).height,
        "validation_baseline": {"f05": old_metric["macro_f05"], "precision": old_metric["macro_precision"],
                                "recall": old_metric["macro_recall"], "singleton_acc": old_metric["singleton_acc"],
                                "false_singletons": old_single_fp, "accepted_pairs": pred_old.height},
        "validation_new_sample_only": {"f05": new_metric["macro_f05"], "precision": new_metric["macro_precision"],
                                       "recall": new_metric["macro_recall"], "singleton_acc": new_metric["singleton_acc"],
                                       "false_singletons": new_single_fp, "accepted_pairs": pred_new.height},
        "runtime_sec": round(time.time() - T0),
    }
    (OUT / "validation_sample_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    new_top.select("q_idx", "s1_idx", "p1", "p2", "p_stage2", "p_final", "true_s1").write_parquet(
        OUT / "validation_sample_new_best.parquet")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
