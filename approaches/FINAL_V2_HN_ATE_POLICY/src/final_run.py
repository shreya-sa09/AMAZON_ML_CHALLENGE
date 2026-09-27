"""FINAL_V2_HN_ATE_POLICY: Approach 02 Stage-2 + house-number (Exp 10) + cross-fitted address token odds (Exp 11),
115 features, group-aware decision rule (Exp 15): threshold 0.85 if another record points to the same S1 with
p_final >= 0.90, else 0.70.

usage: python src/final_run.py --verify   rebuild validation features through this code path, check identity with the
                                          Exp 10/11 caches, score the saved Exp 11 R2 model, plain + LB-proxy F0.5
       python src/final_run.py --smoke    tiny end-to-end pipeline check -> results/smoke/
       python src/final_run.py            FULL run -> output/
Writes ONLY inside approaches/FINAL_V2_HN_ATE_POLICY/. Run with PYTHONDONTWRITEBYTECODE=1.
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

import ate
import common  # noqa: F401  (sys.path for baseline + Approach 02 modules)
from common import A02, A02_CFG, BASE_FEATURES, ROOT, SEED, log, stage2
from newfeats import HN_COLS, hn_features
from predict import write_grouped   # baseline writer (read-only)
from train import id_tables         # baseline (read-only)

APP = Path(__file__).resolve().parents[1]
SMOKE, VERIFY = "--smoke" in sys.argv, "--verify" in sys.argv
CACHE, RES = APP / "cache", APP / "results"
OUT = RES / "smoke" if SMOKE else APP / "output"
TAG = "smoke_" if SMOKE else ""
CFG = json.load(open(APP / "config/final_config.json"))
POL = CFG["policy"]
FEATS = BASE_FEATURES + HN_COLS + ate.ATE_COLS
T0 = time.time()


def norm(split):
    q = pl.concat([pl.read_parquet(ROOT / f"cache/{split}_{s}_norm.parquet", columns=["addr_norm", "addr_nums"]) for s in ("source2", "source3")]).with_row_index("q_idx")
    s = pl.read_parquet(ROOT / f"cache/{split}_source1_norm.parquet", columns=["addr_norm", "addr_nums"]).with_row_index("s1_idx")
    return q, s


def pairs_of(rows, qn, sn):
    return rows.select("q_idx", "s1_idx").with_row_index("_r").join(qn, on="q_idx", how="left") \
        .join(sn, on="s1_idx", how="left", suffix="_s").sort("_r")


def learn_tables(pool_mask):
    """cross-fit ATE tables on training top-1 rows exactly like stage2.build_training: (te for half-0 rows, te for half-1 rows, te_all)."""
    _, _, _, best = stage2.train_rows()
    best = best.with_columns(pl.col("q_is_val").fill_null(False))
    pool = best.filter(pool_mask).select("q_idx", "s1_idx", "y", "te_half")
    qn, sn = norm("train")
    pp = pairs_of(pool, qn, sn).join(pool.with_row_index("_r").select("_r", "y", "te_half"), on="_r")
    log(f"ATE pool rows {pp.height:,}")
    return ate.learn(pp.filter(pl.col("te_half"))), ate.learn(pp.filter(~pl.col("te_half"))), ate.learn(pp), best.select("q_idx", "te_half")


def add_feats(rows, split, te=None, te_h1=None, te_h0=None, half=None):
    qn, sn = norm(split)
    p = pairs_of(rows, qn, sn)
    hn = hn_features(p)
    if half is None:
        a = ate.apply(p, te)
    else:  # cross-fitted training rows: half-0 rows use tables learned on half 1 and vice versa
        p = p.join(rows.select("q_idx").with_row_index("_r").join(half, on="q_idx", how="left").select("_r", "te_half"), on="_r").sort("_r")
        a0 = ate.apply(p.filter(~pl.col("te_half")).sort("_r"), te_h1).with_columns(p.filter(~pl.col("te_half")).sort("_r")["_r"])
        a1 = ate.apply(p.filter(pl.col("te_half")).sort("_r"), te_h0).with_columns(p.filter(pl.col("te_half")).sort("_r")["_r"])
        a = pl.concat([a0, a1]).sort("_r").drop("_r")
    log(f"{split}: hn + ate features for {rows.height:,} rows")
    return pl.concat([rows, hn, a], how="horizontal")


def policy(scored):
    n = ((pl.col("p_final") >= POL["conf"]).cast(pl.Int32).sum().over("s1_idx") - (pl.col("p_final") >= POL["conf"]).cast(pl.Int32))
    return scored.with_columns(pl.when(n > 0).then(pl.lit(POL["t_group"])).otherwise(pl.lit(POL["t_alone"])).alias("thr"))


def verify():
    _, _, te_all, _ = learn_tables(~pl.col("q_is_val") & pl.col("oos"))
    V = add_feats(pl.read_parquet(A02 / "cache/validate_val_rows_cf1_ru0.parquet"), "train", te=te_all)
    cache = V.select("q_idx").join(pl.read_parquet(ROOT / "approaches/10_STAGE2_ADDRESS_SIGNALS/cache/feat_hn_val.parquet"), on="q_idx", how="left", maintain_order="left") \
             .join(pl.read_parquet(ROOT / "approaches/11_STAGE2_ADDRESS_TOKEN_ODDS/cache/feat_ate_val.parquet"), on="q_idx", how="left", maintain_order="left")
    diffs = {c: float(np.nanmax(np.abs(V[c].to_numpy() - cache[c].to_numpy()))) for c in HN_COLS + ate.ATE_COLS}
    m = lgb.Booster(model_file=str(ROOT / "approaches/11_STAGE2_ADDRESS_TOKEN_ODDS/cache/model_R2_hn_ate.txt"))
    assert m.feature_name() == FEATS, "saved Exp 11 R2 model feature vector differs"
    p = m.predict(V.select(FEATS).to_numpy().astype(np.float32), num_threads=16)
    sys.path.insert(0, str(ROOT / "approaches/15_DENSITY_MATCHED_EVAL/src"))
    from density import DensityEval
    ev = DensityEval()
    b = policy(ev.frame(pl.DataFrame({"q_idx": V["q_idx"], "p_stage2": p})))
    plain, prox = ev.evaluate(b, pl.col("thr")), ev.evaluate(b, pl.col("thr"), 2.4, "spread")
    u7 = ev.evaluate(b, pl.lit(0.7))
    res = {"feature_max_abs_diff_vs_exp10_11_caches": diffs, "features_identical": all(d == 0 for d in diffs.values()),
           "plain_policy": plain, "lb_proxy_policy": prox["f05"] - 0.0019, "plain_uniform_0.7": u7,
           "expected": {"lb_proxy_policy": 0.982344, "plain_uniform_0.7": 0.986354}, "runtime_sec": round(time.time() - T0)}
    json.dump(res, open(RES / "verify_summary.json", "w"), indent=2, default=float)
    log(f"VERIFY identical={res['features_identical']} plain(policy) {plain['f05']:.6f} LB-proxy(policy) {res['lb_proxy_policy']:.6f} "
        f"plain(u0.7) {u7['f05']:.6f}  [expected proxy 0.982344, u0.7 0.986354]")


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
    log(f"FINAL_V2 {'SMOKE' if SMOKE else 'FULL'} run; policy {POL}; {len(FEATS)} features")
    te_h1, te_h0, te_all, half = learn_tables(pl.col("oos"))      # final pool = all training folds (as Approach 02 final)
    T = pl.concat([pl.read_parquet(A02 / "cache/final_train_rows_h0_cf1_ru0.parquet"),
                   pl.read_parquet(A02 / "cache/final_train_rows_h1_cf1_ru0.parquet")]).sample(fraction=1.0, shuffle=True, seed=SEED)
    if SMOKE:
        T = T.head(60_000)
    model = fit(add_feats(T, "train", te_h1=te_h1, te_h0=te_h0, half=half), CACHE / f"{TAG}final_model.txt")
    del T
    X = pl.read_parquet(A02 / "cache/final_test_rows_cf1_ru0.parquet")
    if SMOKE:
        X = X.head(50_000)
    X = add_feats(X, "test", te=te_all)
    pt = model.predict(X.select(FEATS).to_numpy().astype(np.float32), num_threads=16).astype(np.float32)
    tb = pl.read_parquet(ROOT / "cache/test_best.parquet")
    if SMOKE:
        tb = tb.join(X.select("q_idx"), on="q_idx", how="semi")
    scored = policy(tb.join(X.select("q_idx", pl.Series("p_stage2", pt)), on="q_idx", how="left")
                    .with_columns(pl.coalesce("p_stage2", "p1").alias("p_final")))
    scored.write_parquet(CACHE / f"{TAG}test_final_scored.parquet")
    matches = scored.filter(pl.col("p_final") >= pl.col("thr")).select("s1_idx", "q_idx")
    # test-shift gate vs the submitted FINAL_HN_ONLY scores (same records)
    hn = pl.read_parquet(ROOT / "approaches/FINAL_HN_ONLY/cache/test_final_scored.parquet", columns=["q_idx", "p_final"]) \
        .join(scored.select("q_idx"), on="q_idx", how="semi")
    gate = {"hn_only_uniform0.7": int((hn["p_final"] >= 0.7).sum()), "v2_uniform0.7": int((scored["p_final"] >= 0.7).sum()),
            "v2_policy": matches.height}
    gate["ate_shift_pct_uniform0.7"] = 100 * (gate["v2_uniform0.7"] / gate["hn_only_uniform0.7"] - 1)
    gate["policy_vs_hn_only_pct"] = 100 * (gate["v2_policy"] / gate["hn_only_uniform0.7"] - 1)
    log(f"test-shift gate: {gate}")
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
    summary = {"policy": POL, "smoke": SMOKE, "n_features": len(FEATS), "best_iteration": model.best_iteration,
               "test_band_rows": X.height, "matched_records": matches.height, "test_shift_gate": gate,
               "s1_rows": n, "s1_with_match": n_m, "s1_with_candidates": n_c,
               "per_country": dict(zip(per_c["country"].to_list(), per_c["matched_records"].to_list())),
               "output_dir": str(OUT), "runtime_sec": round(time.time() - T0)}
    json.dump(summary, open(RES / f"{TAG}final_summary.json", "w"), indent=2)
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
