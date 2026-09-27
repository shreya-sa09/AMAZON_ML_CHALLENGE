"""FINAL_V5_CHAR8: Approach 02 Stage-2 + house-number (Exp 10) + address token odds (Exp 11) + cross-fitted
4-epoch char-level pair model score (Exp 17/21) + confident-anchor sibling features (Exp 19), 122 features;
expected-F0.5 per-S1 decision rule with density calibration c=0.5 (Exp 18).

usage: python src/final_run.py --verify   validation reproduction through this code path (saved Exp 21 char models + R7)
       python src/final_run.py --smoke    tiny end-to-end pipeline check -> results/smoke/
       python src/final_run.py            FULL run -> output/
Writes ONLY inside approaches/FINAL_V5_CHAR8/. Run with PYTHONDONTWRITEBYTECODE=1.
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import torch

import asib
import ate
import charmodel as cm
import common  # noqa: F401
from common import A02, A02_CFG, BASE_FEATURES, ROOT, SEED, log, stage2
from newfeats import HN_COLS, hn_features
from predict import write_grouped
from train import id_tables

APP = Path(__file__).resolve().parents[1]
SMOKE, VERIFY = "--smoke" in sys.argv, "--verify" in sys.argv
CACHE, RES = APP / "cache", APP / "results"
OUT = RES / "smoke" if SMOKE else APP / "output"
TAG = "smoke_" if SMOKE else ""
CFG = json.load(open(APP / "config/final_config.json"))
C_ODDS = CFG["decision"]["calibration_odds_multiplier"]
FEATS = BASE_FEATURES + HN_COLS + ate.ATE_COLS + ["char_p"] + asib.ASIB_COLS
E21 = ROOT / "approaches/24_CHAR_8EPOCH/cache"
E19 = ROOT / "approaches/19_ANCHOR_SIBLINGS/cache"
T0 = time.time()


def norm(split):
    cols = ["name_core", "addr_norm", "addr_nums"]
    q = pl.concat([pl.read_parquet(ROOT / f"cache/{split}_{s}_norm.parquet", columns=cols) for s in ("source2", "source3")]).with_row_index("q_idx")
    s = pl.read_parquet(ROOT / f"cache/{split}_source1_norm.parquet", columns=cols).with_row_index("s1_idx")
    return q, s


def pairs_of(rows, qn, sn):
    return rows.select("q_idx", "s1_idx").with_row_index("_r").join(qn, on="q_idx", how="left") \
        .join(sn, on="s1_idx", how="left", suffix="_s").sort("_r")


def char_arrays(p):
    return {"qn": cm.encode(p["name_core"].fill_null("").str.replace_all(" ", "").to_list(), cm.LN),
            "sn": cm.encode(p["name_core_s"].fill_null("").str.replace_all(" ", "").to_list(), cm.LN),
            "qa": cm.encode(p["addr_norm"].fill_null("").to_list(), cm.LA),
            "sa": cm.encode(p["addr_norm_s"].fill_null("").to_list(), cm.LA)}


def learn_tables(pool_mask):
    _, _, _, best = stage2.train_rows()
    best = best.with_columns(pl.col("q_is_val").fill_null(False))
    pool = best.filter(pool_mask).select("q_idx", "s1_idx", "y", "te_half")
    qn, sn = norm("train")
    pp = pairs_of(pool, qn, sn).join(pool.with_row_index("_r").select("_r", "y", "te_half"), on="_r")
    log(f"ATE pool rows {pp.height:,}")
    return ate.learn(pp.filter(pl.col("te_half"))), ate.learn(pp.filter(~pl.col("te_half"))), ate.learn(pp), best.select("q_idx", "te_half")


def ate_crossfit(p, rows, half, te_h1, te_h0):
    p = p.join(rows.select("q_idx").with_row_index("_r").join(half, on="q_idx", how="left").select("_r", "te_half"), on="_r").sort("_r")
    parts = []
    for flag, te in ((False, te_h1), (True, te_h0)):
        sub = p.filter(pl.col("te_half") == flag).sort("_r")
        parts.append(ate.apply(sub, te).with_columns(sub["_r"]))
    return pl.concat(parts).sort("_r").drop("_r")


def ef05_accept(b, c):
    """Expected-F0.5 prefix per S1 vs empty set, on density-calibrated probabilities (Exp 18)."""
    p = b["p_final"].clip(1e-6, 1 - 1e-6)
    odds = p / (1 - p) * c
    x = b.select("s1_idx").with_row_index("_r").with_columns(pl.Series("pc", odds / (1 + odds)))
    x = x.sort(["s1_idx", "pc"], descending=[False, True]).with_columns(
        pl.col("pc").cum_sum().over("s1_idx").alias("cum"), pl.col("pc").sum().over("s1_idx").alias("tot"),
        pl.int_range(1, pl.len() + 1).over("s1_idx").alias("k"), (1 - pl.col("pc")).log().sum().over("s1_idx").exp().alias("p_none"))
    x = x.with_columns((1.25 * pl.col("cum") / (0.25 * pl.col("tot") + pl.col("k"))).alias("val"))
    x = x.with_columns(pl.col("val").max().over("s1_idx").alias("vmax"))
    kb = x.filter(pl.col("val") == pl.col("vmax")).group_by("s1_idx").agg(pl.col("k").min().alias("kb"))
    x = x.join(kb, on="s1_idx").filter((pl.col("k") <= pl.col("kb")) & (pl.col("vmax") > pl.col("p_none")))
    acc = np.zeros(b.height, dtype=bool)
    acc[x["_r"].to_numpy()] = True
    return acc


def load_char(tag):
    m = cm.CharPair().to(cm.DEV)
    m.load_state_dict(torch.load(E21 / f"char8_model_{tag}.pt", map_location=cm.DEV))
    return m


def verify():
    V = pl.read_parquet(A02 / "cache/validate_val_rows_cf1_ru0.parquet")
    qn, sn = norm("train")
    p = pairs_of(V, qn, sn)
    X = char_arrays(p)
    cp = (cm.predict(load_char("A"), X) + cm.predict(load_char("B"), X)) / 2
    ref = V.select("q_idx").join(pl.read_parquet(E21 / "char8_val.parquet"), on="q_idx", how="left", maintain_order="left")["char_p"].to_numpy()
    best_tr = pl.read_parquet(ROOT / "cache/train_best.parquet", columns=["q_idx", "s1_idx", "p1"])
    av = asib.asib_features(V, best_tr, qn)
    aref = V.select("q_idx").join(pl.read_parquet(E19 / "feat_asib_val.parquet"), on="q_idx", how="left", maintain_order="left")
    asib_diff = max(float(np.nanmax(np.abs(av[c].to_numpy() - aref[c].to_numpy()))) for c in asib.ASIB_COLS)
    _, _, te_all, _ = learn_tables(~pl.col("q_is_val") & pl.col("oos"))
    V = pl.concat([V, hn_features(p), ate.apply(p, te_all), pl.DataFrame({"char_p": cp.astype(np.float32)}), av], how="horizontal")
    m = lgb.Booster(model_file=str(E21 / "model_R10_s42.txt"))
    assert m.feature_name() == FEATS
    pv = m.predict(V.select(FEATS).to_numpy().astype(np.float32), num_threads=16)
    sys.path.insert(0, str(ROOT / "approaches/15_DENSITY_MATCHED_EVAL/src"))
    from density import DensityEval
    ev = DensityEval()
    b = ev.frame(pl.DataFrame({"q_idx": V["q_idx"], "p_stage2": pv}))
    bb = b.with_columns(pl.Series("p_final", ef05_accept(b, C_ODDS).astype(np.float64)))
    res = {"char_p_max_abs_diff_vs_exp21": float(np.max(np.abs(cp - ref))), "char_p_mean_abs_diff": float(np.mean(np.abs(cp - ref))), "asib_max_abs_diff_vs_exp19": asib_diff,
           "plain_u0.7": ev.evaluate(b, pl.lit(0.7))["f05"], "plain_ef05": ev.evaluate(bb, pl.lit(0.5))["f05"],
           "lb_proxy_ef05": ev.evaluate(bb, pl.lit(0.5), 2.4, "spread")["f05"] - 0.0019,
           "expected": {"plain_ef05": 0.9882, "lb_proxy_ef05": 0.985049}, "runtime_sec": round(time.time() - T0)}
    json.dump(res, open(RES / "verify_summary.json", "w"), indent=2)
    log(f"VERIFY {json.dumps(res)}")


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
    ep = 1 if SMOKE else CFG["char_model"]["epochs"]
    log(f"FINAL_V5 {'SMOKE' if SMOKE else 'FULL'} run; {len(FEATS)} features; EF0.5 c={C_ODDS}")
    te_h1, te_h0, te_all, half = learn_tables(pl.col("oos"))
    h0 = pl.read_parquet(A02 / "cache/final_train_rows_h0_cf1_ru0.parquet")
    h1 = pl.read_parquet(A02 / "cache/final_train_rows_h1_cf1_ru0.parquet")
    if SMOKE:
        h0, h1 = h0.head(30_000), h1.head(30_000)
    X = pl.read_parquet(A02 / "cache/final_test_rows_cf1_ru0.parquet")
    if SMOKE:
        X = X.head(50_000)
    qn, sn = norm("train")
    p0, p1 = pairs_of(h0, qn, sn), pairs_of(h1, qn, sn)
    A0, A1 = char_arrays(p0), char_arrays(p1)
    # cross-fitted char model (test arrays are encoded only after training, to limit RAM/VRAM pressure)
    mA = cm.train_model(A0, h0["y"].to_numpy(), epochs=ep, log=log)
    c1 = cm.predict(mA, A1)
    torch.save(mA.state_dict(), CACHE / f"{TAG}char_model_A.pt")
    del mA
    torch.cuda.empty_cache()
    mB = cm.train_model(A1, h1["y"].to_numpy(), epochs=ep, log=log)
    c0 = cm.predict(mB, A0)
    torch.save(mB.state_dict(), CACHE / f"{TAG}char_model_B.pt")
    del mB, A0, A1
    torch.cuda.empty_cache()
    qt, st = norm("test")
    pt_ = pairs_of(X, qt, st)
    AT = char_arrays(pt_)
    cT = []
    for tag in ("A", "B"):
        m = cm.CharPair().to(cm.DEV)
        m.load_state_dict(torch.load(CACHE / f"{TAG}char_model_{tag}.pt", map_location=cm.DEV))
        cT.append(cm.predict(m, AT))
        del m
    cA, cB = cT
    del AT
    torch.cuda.empty_cache()
    log("char models trained and scored")
    # training rows: A02 final halves + hn + ate (cross-fitted) + char OOF
    h0 = pl.concat([h0, hn_features(p0), pl.DataFrame({"char_p": c0.astype(np.float32)})], how="horizontal")
    h1 = pl.concat([h1, hn_features(p1), pl.DataFrame({"char_p": c1.astype(np.float32)})], how="horizontal")
    T = pl.concat([h0, h1]).sample(fraction=1.0, shuffle=True, seed=SEED)
    del h0, h1
    pT = pairs_of(T, qn, sn)
    T = pl.concat([T, ate_crossfit(pT, T, half, te_h1, te_h0)], how="horizontal")
    T = pl.concat([T, asib.asib_features(T, pl.read_parquet(ROOT / "cache/train_best.parquet", columns=["q_idx", "s1_idx", "p1"]), qn)], how="horizontal")
    del pT, qn, sn
    log(f"train rows {T.height:,} with {len(FEATS)} features")
    model = fit(T, CACHE / f"{TAG}final_model.txt")
    del T
    X = pl.concat([X, hn_features(pt_), ate.apply(pt_, te_all), pl.DataFrame({"char_p": ((cA + cB) / 2).astype(np.float32)}),
                   asib.asib_features(X, pl.read_parquet(ROOT / "cache/test_best.parquet", columns=["q_idx", "s1_idx", "p1"]), qt)], how="horizontal")
    pt = model.predict(X.select(FEATS).to_numpy().astype(np.float32), num_threads=16).astype(np.float32)
    tb = pl.read_parquet(ROOT / "cache/test_best.parquet")
    if SMOKE:
        tb = tb.join(X.select("q_idx"), on="q_idx", how="semi")
    scored = tb.join(X.select("q_idx", pl.Series("p_stage2", pt)), on="q_idx", how="left").with_columns(pl.coalesce("p_stage2", "p1").alias("p_final"))
    scored = scored.with_columns(pl.Series("accept", ef05_accept(scored, C_ODDS)))
    scored.write_parquet(CACHE / f"{TAG}test_final_scored.parquet")
    matches = scored.filter(pl.col("accept")).select("s1_idx", "q_idx")
    # test-shift gates
    hn = pl.read_parquet(ROOT / "approaches/FINAL_HN_ONLY/cache/test_final_scored.parquet", columns=["q_idx", "p_final"]).join(scored.select("q_idx"), on="q_idx", how="semi")
    cty = pl.concat([pl.read_parquet(ROOT / f"cache/test_{s}_norm.parquet", columns=["country"]) for s in ("source2", "source3")]).with_row_index("q_idx")
    xc = X.select("q_idx", "char_p", "p1").join(cty, on="q_idx", how="left")
    gate = {"hn_only_uniform0.7": int((hn["p_final"] >= 0.7).sum()), "v4_uniform0.7": int((scored["p_final"] >= 0.7).sum()),
            "v4_ef05": matches.height,
            "char_p_mean_by_country_test_band": dict(xc.group_by("country").agg(pl.col("char_p").mean()).iter_rows()),
            "char_p_corr_with_p1_by_country": dict(xc.group_by("country").agg(pl.corr("char_p", "p1")).iter_rows())}
    gate["v3_file_ef05_accepted"] = int(pl.read_parquet(ROOT / "approaches/FINAL_V3_CHAR_EF05/cache/test_final_scored.parquet", columns=["accept"])["accept"].sum()) if not SMOKE else None
    gate["model_shift_pct_uniform0.7"] = 100 * (gate["v4_uniform0.7"] / gate["hn_only_uniform0.7"] - 1)
    gate["ef05_vs_hn_only_pct"] = 100 * (gate["v4_ef05"] / gate["hn_only_uniform0.7"] - 1)
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
    summary = {"decision": CFG["decision"], "smoke": SMOKE, "n_features": len(FEATS), "best_iteration": model.best_iteration,
               "test_band_rows": X.height, "matched_records": matches.height, "test_shift_gate": gate,
               "s1_rows": n, "s1_with_match": n_m, "s1_with_candidates": n_c,
               "per_country": dict(zip(per_c["country"].to_list(), per_c["matched_records"].to_list())),
               "output_dir": str(OUT), "runtime_sec": round(time.time() - T0)}
    json.dump(summary, open(RES / f"{TAG}final_summary.json", "w"), indent=2, default=float)
    log(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
