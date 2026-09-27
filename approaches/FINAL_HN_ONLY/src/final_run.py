"""FINAL_HN_ONLY: Approach 02 Stage-2 + Exp 10 house-number features (109 features), threshold 0.70.
Sibling evidence (Exp 13/14) is excluded: it shifts test acceptances ~18x more than validation (Exp 14).

usage: python src/final_run.py --verify   validation reproduction with the saved Exp 10 R1 model (expects 0.986268)
       python src/final_run.py --smoke    tiny end-to-end pipeline check -> results/smoke/
       python src/final_run.py            FULL run: train on all folds, score full test -> output/
Writes ONLY inside approaches/FINAL_HN_ONLY/. Run with PYTHONDONTWRITEBYTECODE=1.
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

import common  # noqa: F401  (sys.path for baseline + Approach 02 modules)
from common import A02, A02_CFG, BASE_FEATURES, ROOT, SEED, Evaluator, log
from newfeats import HN_COLS, hn_features
from predict import write_grouped   # baseline writer (read-only)
from train import id_tables         # baseline (read-only)

APP = Path(__file__).resolve().parents[1]
SMOKE, VERIFY = "--smoke" in sys.argv, "--verify" in sys.argv
CACHE, RES = APP / "cache", APP / "results"
OUT = RES / "smoke" if SMOKE else APP / "output"
TAG = "smoke_" if SMOKE else ""
THR = json.load(open(APP / "config/final_config.json"))["threshold"]
FEATS = BASE_FEATURES + HN_COLS
T0 = time.time()


def add_hn(rows, split):
    qn = pl.concat([pl.read_parquet(ROOT / f"cache/{split}_{s}_norm.parquet", columns=["addr_nums"]) for s in ("source2", "source3")]).with_row_index("q_idx")
    sn = pl.read_parquet(ROOT / f"cache/{split}_source1_norm.parquet", columns=["addr_nums"]).with_row_index("s1_idx")
    pairs = rows.select("q_idx", "s1_idx").with_row_index("_r").join(qn, on="q_idx", how="left") \
        .join(sn, on="s1_idx", how="left", suffix="_s").sort("_r")
    log(f"{split}: hn features for {rows.height:,} rows")
    return pl.concat([rows, hn_features(pairs)], how="horizontal")


def verify():
    V = add_hn(pl.read_parquet(A02 / "cache/validate_val_rows_cf1_ru0.parquet"), "train")
    cached = V.select("q_idx").join(pl.read_parquet(ROOT / "approaches/10_STAGE2_ADDRESS_SIGNALS/cache/feat_hn_val.parquet"),
                                    on="q_idx", how="left", maintain_order="left")
    same = all(np.array_equal(V[c].to_numpy(), cached[c].to_numpy()) for c in HN_COLS)
    m = lgb.Booster(model_file=str(ROOT / "approaches/10_STAGE2_ADDRESS_SIGNALS/cache/model_R1_hn.txt"))
    assert m.feature_name() == FEATS, "saved Exp 10 R1 model feature vector differs from FINAL_HN_ONLY"
    res = Evaluator().evaluate("verify_final_hn_only", V, m.predict(V.select(FEATS).to_numpy().astype(np.float32), num_threads=16))
    res.update({"hn_features_identical_to_exp10_cache": same, "runtime_sec": round(time.time() - T0)})
    json.dump(res, open(RES / "verify_summary.json", "w"), indent=2)
    log(f"VERIFY: features identical {same}; F0.5 {res['f05']:.6f} (expected 0.986268)")


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
    if VERIFY:
        return verify()
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"FINAL_HN_ONLY {'SMOKE' if SMOKE else 'FULL'} run; threshold {THR}; {len(FEATS)} features")
    T = pl.concat([pl.read_parquet(A02 / "cache/final_train_rows_h0_cf1_ru0.parquet"),
                   pl.read_parquet(A02 / "cache/final_train_rows_h1_cf1_ru0.parquet")]).sample(fraction=1.0, shuffle=True, seed=SEED)
    if SMOKE:
        T = T.head(60_000)
    model = fit(add_hn(T, "train"), CACHE / f"{TAG}final_model.txt")
    del T
    X = pl.read_parquet(A02 / "cache/final_test_rows_cf1_ru0.parquet")
    if SMOKE:
        X = X.head(50_000)
    X = add_hn(X, "test")
    pt = model.predict(X.select(FEATS).to_numpy().astype(np.float32), num_threads=16).astype(np.float32)
    tb = pl.read_parquet(ROOT / "cache/test_best.parquet")
    if SMOKE:
        tb = tb.join(X.select("q_idx"), on="q_idx", how="semi")
    scored = tb.join(X.select("q_idx", pl.Series("p_stage2", pt)), on="q_idx", how="left") \
               .with_columns(pl.coalesce("p_stage2", "p1").alias("p_final"))
    scored.write_parquet(CACHE / f"{TAG}test_final_scored.parquet")
    matches = scored.filter(pl.col("p_final") >= THR).select("s1_idx", "q_idx")
    log(f"test band rows scored {X.height:,}; matched records {matches.height:,}")
    a02 = pl.read_parquet(A02 / "cache/test_final_scored_cf1_ru0_r8000.parquet", columns=["q_idx", "p_final"]) \
        .join(scored.select("q_idx"), on="q_idx", how="semi")
    a02_acc = a02.filter(pl.col("p_final") >= THR).select("q_idx")
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
