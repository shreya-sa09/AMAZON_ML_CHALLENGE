"""FINAL_BEST_APPROACH: Approach 02 Stage-2 + house-number (Exp 10) + sibling (Exp 13) features, trained on all
training folds, applied to the full test set. Writes ONLY inside approaches/FINAL_BEST_APPROACH/.

usage: python src/final_run.py [--smoke]
  --smoke : 60k training rows, 30 rounds, 50k test band rows, outputs to results/smoke/ (pipeline check only)
Reused read-only: Approach 02 final train/test row tables + config, shared cache (train/test best, norm, scored, ids).
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

import common  # noqa: F401  (sets sys.path for the baseline + Approach 02 modules)
import sib
from common import A02, A02_CFG, BASE_FEATURES, ROOT, SEED, log
from newfeats import HN_COLS, hn_features
from predict import write_grouped   # baseline writer (read-only)
from train import id_tables         # baseline (read-only)

APP = Path(__file__).resolve().parents[1]
SMOKE = "--smoke" in sys.argv
CACHE, RES = APP / "cache", APP / "results"
OUT = RES / "smoke" if SMOKE else APP / "output"
TAG = "smoke_" if SMOKE else ""
THR = json.load(open(APP / "config/final_config.json"))["threshold"]
FEATS = BASE_FEATURES + HN_COLS + sib.SIB_COLS
T0 = time.time()


def norm(split):
    qn = pl.concat([pl.read_parquet(ROOT / f"cache/{split}_{s}_norm.parquet", columns=["name_core", "addr_norm", "addr_nums"])
                    .with_columns(pl.lit(s == "source3").alias("is_s3")) for s in ("source2", "source3")]).with_row_index("q_idx")
    sn = pl.read_parquet(ROOT / f"cache/{split}_source1_norm.parquet", columns=["name_core", "addr_norm", "addr_nums"]).with_row_index("s1_idx")
    return qn, sn


def add_new_features(rows, split):
    qn, sn = norm(split)
    best = pl.read_parquet(ROOT / f"cache/{split}_best.parquet", columns=["q_idx", "s1_idx"])
    pairs = rows.select("q_idx", "s1_idx").with_row_index("_r").join(qn.drop("is_s3"), on="q_idx", how="left") \
        .join(sn, on="s1_idx", how="left", suffix="_s").sort("_r")
    hn = hn_features(pairs)
    sb = sib.sib_features(rows, best, qn.select("q_idx", "name_core", "addr_norm", "is_s3"), sn.select("s1_idx", "addr_norm"),
                          rows["f_ad_ratio"])
    log(f"{split}: hn + sib features for {rows.height:,} rows")
    return pl.concat([rows, hn, sb], how="horizontal")


def fit(T, path):
    X, y = T.select(FEATS).to_numpy().astype(np.float32), T["y"].to_numpy()
    cut = int(0.9 * len(y))
    params = {**A02_CFG["lgbm"], "seed": SEED}
    cat = [c for c in A02_CFG["categorical"] if c in FEATS]
    dtr = lgb.Dataset(X[:cut], y[:cut], feature_name=FEATS, categorical_feature=cat)
    dva = lgb.Dataset(X[cut:], y[cut:], reference=dtr)
    m = lgb.train(params, dtr, 30 if SMOKE else A02_CFG["max_rounds"], valid_sets=[dva], valid_names=["es"],
                  callbacks=[lgb.early_stopping(A02_CFG["early_stopping"], verbose=False), lgb.log_evaluation(500)])
    m.save_model(str(path))
    log(f"final model {path.name}: best_iteration {m.best_iteration}")
    return m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"FINAL {'SMOKE' if SMOKE else 'FULL'} run; threshold {THR}; {len(FEATS)} features")
    # 1. training rows: identical to Approach 02 final (both folds), same shuffle
    h0 = pl.read_parquet(A02 / "cache/final_train_rows_h0_cf1_ru0.parquet")
    h1 = pl.read_parquet(A02 / "cache/final_train_rows_h1_cf1_ru0.parquet")
    T = pl.concat([h0, h1]).sample(fraction=1.0, shuffle=True, seed=SEED)
    del h0, h1
    if SMOKE:
        T = T.head(60_000)
    T = add_new_features(T, "train")
    model = fit(T, CACHE / f"{TAG}final_model.txt")
    del T
    # 2. test band rows (Approach 02 final_test_rows) + new features
    X = pl.read_parquet(A02 / "cache/final_test_rows_cf1_ru0.parquet")
    if SMOKE:
        X = X.head(50_000)
    X = add_new_features(X, "test")
    pt = model.predict(X.select(FEATS).to_numpy().astype(np.float32), num_threads=16).astype(np.float32)
    tb = pl.read_parquet(ROOT / "cache/test_best.parquet")
    if SMOKE:
        tb = tb.join(X.select("q_idx"), on="q_idx", how="semi")
    scored = tb.join(X.select("q_idx", pl.Series("p_stage2", pt)), on="q_idx", how="left") \
               .with_columns(pl.coalesce("p_stage2", "p1").alias("p_final"))
    scored.write_parquet(CACHE / f"{TAG}test_final_scored.parquet")
    matches = scored.filter(pl.col("p_final") >= THR).select("s1_idx", "q_idx")
    log(f"test band rows scored {X.height:,}; matched records {matches.height:,}")
    # 3. comparison with the Approach 02 submission decisions (same threshold) for sanity
    a02 = pl.read_parquet(A02 / "cache/test_final_scored_cf1_ru0_r8000.parquet", columns=["q_idx", "s1_idx", "p_final"]) \
        .join(scored.select("q_idx"), on="q_idx", how="semi")
    a02_acc = a02.filter(pl.col("p_final") >= THR).select("q_idx")
    # 4. write submission files
    s1_t, q_t = id_tables("test")
    s1_ids, q_ids = s1_t["entity_id"], q_t["entity_id"]
    cand = pl.read_parquet(ROOT / "cache/test_scored.parquet", columns=["s1_idx", "q_idx"])
    if SMOKE:
        cand = cand.join(X.select("q_idx"), on="q_idx", how="semi")
    n, n_c = write_grouped(s1_ids, q_ids, cand, "candidate_entity_ids", OUT / "candidate_pairs.tsv")
    del cand
    n, n_m = write_grouped(s1_ids, q_ids, matches, "matched_entity_ids", OUT / "matching_results.tsv")
    ts1 = pl.read_parquet(ROOT / "cache/test_source1.parquet", columns=["country"]).with_row_index("s1_idx")
    per_c = matches.join(ts1, on="s1_idx").group_by("country").agg(pl.len().alias("matched_records"))
    acc_q = matches.select("q_idx")
    summary = {"threshold": THR, "smoke": SMOKE, "n_features": len(FEATS), "best_iteration": model.best_iteration,
               "test_band_rows": X.height, "matched_records": matches.height,
               "approach02_matched_records_same_threshold": a02_acc.height,
               "newly_accepted_vs_a02": acc_q.join(a02_acc, on="q_idx", how="anti").height,
               "no_longer_accepted_vs_a02": a02_acc.join(acc_q, on="q_idx", how="anti").height,
               "s1_rows": n, "s1_with_match": n_m, "s1_with_candidates": n_c,
               "per_country": dict(zip(per_c["country"].to_list(), per_c["matched_records"].to_list())),
               "output_dir": str(OUT), "runtime_sec": round(time.time() - T0)}
    json.dump(summary, open(RES / f"{TAG}final_summary.json", "w"), indent=2)
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
