"""
02_STAGE2_CROSSFIT_RUNNERUP — 01 + cross-fitted token odds + converged round budget. (runner_up context was measured and is OFF by default.)

  cross_fit_te : the token target encoding is cross-fitted (learned on half 1 -> features of half 0 and
                 vice versa), so stage 2 trains on BOTH halves of the held-out records (01 used one
                 half only). Validation/test rows use the encoding learned on all training rows.
  runner_up    : the top-1 row also carries the 64 pair features and 36 edit features of the record's
                 #2 candidate (prefix r2_), so stage 2 sees whether a near-equal alternative exists.

CLI overrides for ablations: --no-crossfit, --no-runner-up, --runner-up, --max-rounds=N.

Base design (unchanged from 01):

Stage 1 (reused, read-only): the 00_BASELINE LightGBM scores in the shared cache
    (train_best / test_best: top-1 S1 per record with p1, p2; *_feats: the 64 pair features).
Stage 2 (this approach): for records whose p1 is in [band_lo, band_hi], a second LightGBM re-scores the
    top-1 pair from  p1, p2, p1-p2  +  the 64 baseline pair features  +  36 edit-type features
    (edit_features.py). Outside the band the stage-1 decision is kept. Accept when p >= threshold.

Leakage control
  * Stage 2 is trained only on records the stage-1 model never saw (hash(q_idx) % 1000 >= 100), so its
    input p1 is out-of-sample exactly as on test.
  * The token target encoding (e_te_*) is learned on one hash half of those records and the stage-2
    model is trained on the other half.
  * validate: everything is learned from fold A (S1 id % 5 != 0); fold B is only scored.
    final:    the same recipe on both folds, applied to test.

usage (from the approach folder):
    python src/stage2.py validate [--smoke]
    python src/stage2.py final    [--smoke]
    python src/stage2.py all      [--smoke]
"""
import json
import sys
import time

import paths  # noqa: F401  (must come first: sets ER_CACHE_DIR / ER_OUTPUT_DIR, sys.path)
from paths import CONFIG, OUTPUT_DIR, RESULTS_DIR, mine, shared

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from edit_features import EDIT_COLS, Vocab, edit_features, token_te
from features import FEATURES as PAIR_FEATURES          # baseline (read-only)
from metrics import macro_f05                           # baseline (read-only)
from predict import write_grouped                       # baseline (read-only)
from train import best_per_query, id_tables, label_table, query_folds  # baseline (read-only)

C = CONFIG
SEED = C["seed"]
SMOKE = "--smoke" in sys.argv
CROSS_FIT = C["cross_fit_te"] and "--no-crossfit" not in sys.argv
RUNNER_UP = (C["runner_up"] or "--runner-up" in sys.argv) and "--no-runner-up" not in sys.argv
VARIANT = f"cf{int(CROSS_FIT)}_ru{int(RUNNER_UP)}"          # feature-table variant
MAX_ROUNDS = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--max-rounds=")), C["max_rounds"]))
MODEL_VARIANT = VARIANT if MAX_ROUNDS == 3000 else f"{VARIANT}_r{MAX_ROUNDS}"  # model / result variant
BASE_FEATURES = ["p1", "p2", "gap"] + ["f_" + c for c in PAIR_FEATURES] + EDIT_COLS
R2_FEATURES = ["r2_f_" + c for c in PAIR_FEATURES] + ["r2_" + c for c in EDIT_COLS] + ["r2_missing"]
STAGE2_FEATURES = BASE_FEATURES + (R2_FEATURES if RUNNER_UP else [])
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def tag(name):
    return f"smoke_{name}" if SMOKE else name


def vtag(name):
    """feature tables (depend on the feature variant)"""
    return tag(f"{name}_{VARIANT}")


def mtag(name):
    """models and results (depend on the feature variant and the number of rounds)"""
    return tag(f"{name}_{MODEL_VARIANT}")


# --------------------------------------------------------------------------- shared inputs
def norm_tables(split):
    cols = ["name_core", "addr_norm", "addr_nums"]
    q = pl.concat([pl.read_parquet(shared(f"{split}_{s}_norm.parquet"), columns=cols) for s in ("source2", "source3")]).with_row_index("q_idx")
    s1 = pl.read_parquet(shared(f"{split}_source1_norm.parquet"), columns=cols).with_row_index("s1_idx")
    return q, s1


def pair_feats(split, keys):
    keys = keys.select(pl.col("q_idx").cast(pl.UInt32), pl.col("s1_idx").cast(pl.UInt32)).unique()
    parts = [pl.read_parquet(f).join(keys, on=["q_idx", "s1_idx"], how="semi")
             for f in sorted(shared(f"{split}_feats").glob("*.parquet"))]
    return pl.concat(parts).rename({c: "f_" + c for c in PAIR_FEATURES})


def build(split, rows, qn, s1n, vocab, te, cache_name):
    """rows: q_idx, s1_idx, p1, p2 (+ y). Returns rows + stage-2 features. Cached (resume)."""
    path = mine(cache_name + ".parquet")
    if path.exists():
        log(f"reuse {path.name}")
        return pl.read_parquet(path)
    rows = rows.with_columns(pl.col("q_idx").cast(pl.UInt32), pl.col("s1_idx").cast(pl.UInt32))
    log(f"{cache_name}: computing edit features for {rows.height:,} top-1 pairs")
    keep = [c for c in ("q_idx", "s1_idx", "p1", "p2", "y") if c in rows.columns]
    parts = [rows.select(keep), pair_block(split, rows, qn, s1n, vocab, te, "s1_idx", "")]
    if RUNNER_UP:
        rows = rows.join(second_best(split, rows), on="q_idx", how="left", maintain_order="left")
        log(f"{cache_name}: computing edit features for {rows['s1_2'].is_not_null().sum():,} runner-up pairs")
        parts += [pair_block(split, rows, qn, s1n, vocab, te, "s1_2", "r2_"),
                  rows.select(pl.col("s1_2").is_null().cast(pl.Float32).alias("r2_missing"))]
    out = pl.concat(parts, how="horizontal").with_columns((pl.col("p1") - pl.col("p2")).alias("gap"))
    out.write_parquet(path)
    return out


def second_best(split, qset):
    """(q_idx, s1_2): the #2 candidate of each record under the stage-1 scores (shared cache)."""
    sc = pl.read_parquet(shared(f"{split}_scored.parquet")).join(
        qset.select(pl.col("q_idx").cast(pl.UInt32)).unique(), on="q_idx", how="semi")
    return (sc.sort(["q_idx", "p"], descending=[False, True]).group_by("q_idx", maintain_order=True)
              .agg(pl.col("s1_idx").get(1).alias("s1_2")).drop_nulls())


def pair_block(split, rows, qn, s1n, vocab, te, s1_col, prefix):
    """pair features + edit features of (q_idx, rows[s1_col]); columns prefixed; row-aligned with rows."""
    keys = rows.select("q_idx", pl.col(s1_col).alias("s1_idx")).drop_nulls()
    df = rows.select("q_idx", pl.col(s1_col).alias("s1_idx")).with_row_index("_r")
    df = df.join(pair_feats(split, keys), on=["q_idx", "s1_idx"], how="left")
    df = df.join(qn, on="q_idx", how="left").join(s1n, on="s1_idx", how="left", suffix="_s")
    df = df.sort("_r").with_columns([pl.col(c).fill_null("") for c in
                                     ("name_core", "name_core_s", "addr_norm", "addr_norm_s", "addr_nums", "addr_nums_s")])
    ef = edit_features(df, vocab, *te)
    out = pl.concat([df.select(*["f_" + c for c in PAIR_FEATURES]), ef], how="horizontal")
    return out.rename({c: prefix + c for c in out.columns}) if prefix else out


def in_band(col="p1"):
    return pl.col(col).is_between(C["band_lo"], C["band_hi"])


def train_rows():
    """Stage-1 top-1 predictions on train with labels, fold, and leakage-control hashes."""
    s1, q, lab, tp = label_table()
    qf = query_folds(lab, s1)
    best = pl.read_parquet(shared("train_best.parquet")).join(qf, on="q_idx", how="left").with_columns(
        (pl.col("s1_idx") == pl.col("true_s1")).fill_null(False).alias("y"),
        ((pl.col("q_idx").hash(SEED) % 1000) >= C["production_train_hash_cut"]).alias("oos"),
        ((pl.col("q_idx").hash(SEED + 11) % 2) == 1).alias("te_half"))
    return s1, q, tp, best


def smoke_limit(df, n):
    return df.head(n) if SMOKE else df


# --------------------------------------------------------------------------- model
def fit(T, model_path):
    if model_path.exists() and not SMOKE:
        log(f"reuse {model_path.name}")
        return lgb.Booster(model_file=str(model_path))
    X, y = T.select(STAGE2_FEATURES).to_numpy().astype(np.float32), T["y"].to_numpy()
    cut = int(0.9 * len(y))
    params = {**C["lgbm"], "seed": SEED}
    cat = [c for c in C["categorical"] if c in STAGE2_FEATURES]
    dtr = lgb.Dataset(X[:cut], y[:cut], feature_name=STAGE2_FEATURES, categorical_feature=cat)
    dva = lgb.Dataset(X[cut:], y[cut:], reference=dtr)
    rounds = 50 if SMOKE else MAX_ROUNDS
    m = lgb.train(params, dtr, rounds, valid_sets=[dva], valid_names=["es"],
                  callbacks=[lgb.early_stopping(C["early_stopping"]), lgb.log_evaluation(250)])
    m.save_model(str(model_path))
    imp = sorted(zip(STAGE2_FEATURES, m.feature_importance("gain")), key=lambda x: -x[1])
    tot = sum(g for _, g in imp) or 1.0
    pl.DataFrame({"feature": [f for f, _ in imp], "gain_pct": [100 * g / tot for _, g in imp]}).write_csv(
        RESULTS_DIR / (model_path.stem + "_importance.csv"))
    log(f"model {model_path.name}: best_iteration {m.best_iteration}")
    return m


def build_training(split, best, qn, s1n, vocab, pool, name):
    """Stage-2 training rows from `pool` (a mask over best). Returns (rows, TE for scoring new rows)."""
    if not CROSS_FIT:  # 01 recipe: TE on half 1, train on half 0
        te = learn_te(best, qn, s1n, pool & pl.col("te_half"))
        trn = smoke_limit(best.filter(pool & ~pl.col("te_half") & in_band()), 100_000)
        return build(split, trn.select("q_idx", "s1_idx", "p1", "p2", "y"), qn, s1n, vocab, te, vtag(name)), te
    te1 = learn_te(best, qn, s1n, pool & pl.col("te_half"))
    te0 = learn_te(best, qn, s1n, pool & ~pl.col("te_half"))
    h0 = smoke_limit(best.filter(pool & ~pl.col("te_half") & in_band()), 50_000)
    h1 = smoke_limit(best.filter(pool & pl.col("te_half") & in_band()), 50_000)
    t0 = build(split, h0.select("q_idx", "s1_idx", "p1", "p2", "y"), qn, s1n, vocab, te1, vtag(name + "_h0"))
    t1 = build(split, h1.select("q_idx", "s1_idx", "p1", "p2", "y"), qn, s1n, vocab, te0, vtag(name + "_h1"))
    te_all = learn_te(best, qn, s1n, pool)
    # shuffle so the 10% early-stopping tail mixes both halves
    return pl.concat([t0, t1]).sample(fraction=1.0, shuffle=True, seed=SEED), te_all


def learn_te(best, qn, s1n, mask):
    src = best.filter(mask).select("q_idx", "s1_idx", "y")
    src = smoke_limit(src, 200_000)
    src = src.join(qn.select("q_idx", "name_core"), on="q_idx").join(s1n.select("s1_idx", pl.col("name_core").alias("name_core_s")), on="s1_idx")
    te = token_te(src, C["te_smoothing"])
    log(f"token TE learned from {src.height:,} rows: {len(te[0]):,} query-only / {len(te[1]):,} S1-only tokens")
    return te


# --------------------------------------------------------------------------- validate
def to_ids(pairs, s1, q):
    return pairs.join(s1.select("s1_idx", pl.col("entity_id").alias("s1_id")), on="s1_idx") \
                .join(q.select("q_idx", pl.col("entity_id").alias("m_id")), on="q_idx").select("s1_id", "m_id")


def metrics_row(ids, s1, tp, t, name):
    val_s1 = s1.filter(pl.col("is_val"))
    m = macro_f05(ids, tp, val_s1["entity_id"])
    pe = m["per_entity"]
    row = {"run": name, "threshold": t, "f05": m["macro_f05"], "precision": m["macro_precision"], "recall": m["macro_recall"],
           "singleton_acc": m["singleton_acc"],
           "fp_singleton": pe.filter((pl.col("n_true") == 0) & (pl.col("n_pred") > 0)).height,
           "predictions": int(pe["n_pred"].sum()), "empty_s1": pe.filter(pl.col("n_pred") == 0).height}
    for c in ("India", "US"):
        row[f"f05_{c}"] = macro_f05(ids, tp, val_s1.filter(pl.col("country") == c)["entity_id"])["macro_f05"]
    return row


def validate():
    log("VALIDATE: learn on fold A, score fold B")
    s1, q, tp, best = train_rows()
    qn, s1n = norm_tables("train")
    vocab = Vocab(s1n, s1n.height)
    foldA = ~pl.col("q_is_val") & pl.col("oos")
    T, te = build_training("train", best, qn, s1n, vocab, foldA, "validate_train_rows")
    val = best.filter(pl.col("q_is_val"))
    vband = smoke_limit(val.filter(in_band()), 50_000)
    V = build("train", vband.select("q_idx", "s1_idx", "p1", "p2", "y"), qn, s1n, vocab, te, vtag("validate_val_rows"))
    del qn
    log(f"stage-2 train rows {T.height:,}, validation band rows {V.height:,}")
    model = fit(T, mine(mtag("stage2_validation_model") + ".txt"))
    pv = model.predict(V.select(STAGE2_FEATURES).to_numpy().astype(np.float32), num_threads=16).astype(np.float32)
    V.select("q_idx", "s1_idx", "p1", "y").with_columns(pl.Series("p_stage2", pv)).write_parquet(mine(mtag("validate_val_scored") + ".parquet"))
    auc = {"auc_band_p1": float(roc_auc_score(V["y"], V["p1"])), "auc_band_stage2": float(roc_auc_score(V["y"], pv))}
    log(f"validation band AUC: stage-1 p1 {auc['auc_band_p1']:.4f} -> stage-2 {auc['auc_band_stage2']:.4f}")

    # assemble decisions: fold-B records use stage 2 inside the band; fold-A records keep the stage-1
    # decision (they can still create false merges on fold-B S1, exactly as the baseline evaluation counts)
    newp = val.join(V.select("q_idx", pl.Series("p_stage2", pv)), on="q_idx", how="left") \
              .with_columns(pl.coalesce("p_stage2", "p1").alias("p_final"))
    fold_a = to_ids(best.filter(~pl.col("q_is_val") & (pl.col("p1") >= C["threshold"])).select("q_idx", "s1_idx"), s1, q)
    rows = [metrics_row(pl.concat([to_ids(val.filter(pl.col("p1") >= C["threshold"]).select("q_idx", "s1_idx"), s1, q), fold_a]),
                        s1, tp, C["threshold"], "baseline_stage1")]
    for t in C["report_thresholds"]:
        ids = pl.concat([to_ids(newp.filter(pl.col("p_final") >= t).select("q_idx", "s1_idx"), s1, q), fold_a])
        rows.append(metrics_row(ids, s1, tp, t, "stage2"))
        log(f"t={t}: F0.5 {rows[-1]['f05']:.6f} (P {rows[-1]['precision']:.4f} R {rows[-1]['recall']:.4f}) "
            f"delta vs baseline {rows[-1]['f05'] - rows[0]['f05']:+.6f}")
    pl.DataFrame(rows).write_csv(RESULTS_DIR / (mtag("validation_sweep") + ".csv"))
    main_row = next(r for r in rows if r["run"] == "stage2" and r["threshold"] == C["threshold"])
    summary = {"baseline": rows[0], "stage2_at_threshold": main_row, "delta_f05": main_row["f05"] - rows[0]["f05"],
               **auc, "variant": MODEL_VARIANT, "max_rounds": MAX_ROUNDS, "cross_fit_te": CROSS_FIT, "runner_up": RUNNER_UP, "train_rows": T.height, "val_band_rows": V.height, "smoke": SMOKE,
               "best_iteration": model.best_iteration, "runtime_sec": round(time.time() - T0)}
    json.dump(summary, open(RESULTS_DIR / (mtag("validation") + ".json"), "w"), indent=2)
    log(f"VALIDATION F0.5 @ {C['threshold']}: {main_row['f05']:.6f} (baseline {rows[0]['f05']:.6f}, delta {summary['delta_f05']:+.6f})")


# --------------------------------------------------------------------------- final (test)
def final():
    log("FINAL: learn on all training folds, predict test")
    s1, q, tp, best = train_rows()
    qn, s1n = norm_tables("train")
    ref_size = s1n.height
    vocab = Vocab(s1n, ref_size)
    T, te = build_training("train", best, qn, s1n, vocab, pl.col("oos"), "final_train_rows")
    del qn, s1n, best
    model = fit(T, mine(mtag("stage2_final_model") + ".txt"))
    del T

    tb = pl.read_parquet(shared("test_best.parquet"))
    qn, s1n = norm_tables("test")
    vocab_t = Vocab(s1n, ref_size)
    band = smoke_limit(tb.filter(in_band()), 50_000)
    X = build("test", band.select("q_idx", "s1_idx", "p1", "p2"), qn, s1n, vocab_t, te, vtag("final_test_rows"))
    del qn, s1n
    pt = model.predict(X.select(STAGE2_FEATURES).to_numpy().astype(np.float32), num_threads=16).astype(np.float32)
    scored = tb.join(X.select("q_idx", pl.Series("p_stage2", pt)), on="q_idx", how="left") \
               .with_columns(pl.coalesce("p_stage2", "p1").alias("p_final"))
    scored.write_parquet(mine(mtag("test_final_scored") + ".parquet"))
    t = C["threshold"]
    matches = scored.filter(pl.col("p_final") >= t).select("s1_idx", "q_idx")

    s1_t, q_t = id_tables("test")
    s1_ids, q_ids = s1_t["entity_id"], q_t["entity_id"]
    del s1_t, q_t
    out_dir = RESULTS_DIR / "smoke" if SMOKE else OUTPUT_DIR
    out_dir.mkdir(exist_ok=True)
    cand = pl.read_parquet(shared("test_scored.parquet"), columns=["s1_idx", "q_idx"])
    if SMOKE:  # tiny files, only to exercise the writer
        keep = matches.head(2000).select("q_idx")
        cand, matches = cand.join(keep, on="q_idx", how="semi"), matches.head(2000)
    n, n_c = write_grouped(s1_ids, q_ids, cand, "candidate_entity_ids", out_dir / "candidate_pairs.tsv")
    del cand
    n, n_m = write_grouped(s1_ids, q_ids, matches, "matched_entity_ids", out_dir / "matching_results.tsv")
    ts1 = pl.read_parquet(shared("test_source1.parquet"), columns=["country"]).with_row_index("s1_idx")
    per_c = matches.join(ts1, on="s1_idx").group_by("country").agg(pl.len().alias("matched_records"))
    summary = {"threshold": t, "test_band_rows": X.height, "matched_records": matches.height,
               "stage1_matched_records": int((tb["p1"] >= t).sum()), "s1_rows": n, "s1_with_match": n_m,
               "per_country": dict(zip(per_c["country"].to_list(), per_c["matched_records"].to_list())),
               "output_dir": str(out_dir), "smoke": SMOKE, "runtime_sec": round(time.time() - T0)}
    json.dump(summary, open(RESULTS_DIR / (mtag("final_summary") + ".json"), "w"), indent=2)
    log(f"wrote {out_dir / 'matching_results.tsv'}: {matches.height:,} matched records over {n_m:,}/{n:,} S1 "
        f"(stage 1 alone: {summary['stage1_matched_records']:,})")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    json.dump({**C, "variant": MODEL_VARIANT, "max_rounds": MAX_ROUNDS, "cross_fit_te": CROSS_FIT, "runner_up": RUNNER_UP},
              open(RESULTS_DIR / (mtag("config_used") + ".json"), "w"), indent=2)
    if step in ("validate", "all"):
        validate()
    if step in ("final", "all"):
        final()
