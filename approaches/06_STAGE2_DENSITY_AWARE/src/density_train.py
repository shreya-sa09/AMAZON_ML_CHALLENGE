"""
06_STAGE2_DENSITY_AWARE — Experiment 3: density-aware stage-2 training.

ONE change vs Approach 02: in the stage-2 training (and early-stopping) rows, every decoy row
(a record with no true S1 at all) gets sample weight `decoy_weight` (2.0), i.e. the model is trained as if
there were ~2x as many decoys per S1, which is the estimated test density (Experiment 1).
Features, training rows, row order, parameters, round cap and early stopping are exactly 02's; the
feature tables are 02's cached validation tables (read-only).

Evaluation (fold B, same assembly as Experiment 1): fold-B records use the stage-2 probability inside
the band [0.002, 0.995] and stage-1 p1 outside; fold-A records use stage-1 p1 at the same threshold
("consistent", primary) or are excluded (optimistic). Plain metric and the x2 decoy-density simulation.

usage (from the approach folder):
    python src/density_train.py --smoke   # controlled check: w=1 vs w=2 on the same 200k-row subset, 300 rounds
    python src/density_train.py           # full: w=2 on all rows, 8,000-round cap; baseline = 02's saved model
"""
import json
import sys
import time

import paths  # noqa: F401
from paths import APPROACH_DIR, CONFIG, RESULTS_DIR, mine, shared

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score

import stage2 as s2                                       # 02 code (copied): feature list, categorical handling
from metrics import macro_f05                             # baseline (read-only)
from train import label_table, query_folds                # baseline (read-only)

C = CONFIG
SMOKE = "--smoke" in sys.argv
SEED = C["seed"]
T0 = time.time()
R = C["reuse_02_cache"]


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


# --------------------------------------------------------------------------- data
s1, q, lab, tp = label_table()
qf = query_folds(lab, s1)
decoy = qf.select("q_idx", pl.col("true_s1").is_null().alias("decoy"))
T = pl.concat([pl.read_parquet(APPROACH_DIR / R["train_h0"]), pl.read_parquet(APPROACH_DIR / R["train_h1"])]) \
      .sample(fraction=1.0, shuffle=True, seed=SEED)          # identical to 02's build_training()
T = T.join(decoy, on="q_idx", how="left", maintain_order="left")
V = pl.read_parquet(APPROACH_DIR / R["val_rows"]).join(decoy, on="q_idx", how="left", maintain_order="left")
FE = s2.STAGE2_FEATURES
assert len(FE) == 103, len(FE)
log(f"train rows {T.height:,}: positives {int(T['y'].sum()):,}, decoy rows {int(T['decoy'].sum()):,}, "
    f"wrong-S1 negatives {int((~T['y'] & ~T['decoy']).sum()):,}; validation band rows {V.height:,}")

best = pl.read_parquet(shared("train_best.parquet")).join(qf, on="q_idx", how="left")
SID = s1.select("s1_idx", pl.col("entity_id").alias("s1_id"))
QID = q.select("q_idx", pl.col("entity_id").alias("m_id"))
val_s1 = s1.filter("is_val")
VAL_IDS = val_s1["entity_id"]
COUNTRY = val_s1.select(pl.col("entity_id").alias("s1_id"), "country")


def fit(Tsub, w, rounds, name):
    X, y = Tsub.select(FE).to_numpy().astype(np.float32), Tsub["y"].to_numpy()
    wt = np.where(Tsub["decoy"].to_numpy(), w, 1.0).astype(np.float32)
    cut = int(0.9 * len(y))
    cat = [c for c in C["categorical"] if c in FE]
    dtr = lgb.Dataset(X[:cut], y[:cut], weight=wt[:cut], feature_name=FE, categorical_feature=cat)
    dva = lgb.Dataset(X[cut:], y[cut:], weight=wt[cut:], reference=dtr)
    t = time.time()
    m = lgb.train({**C["lgbm"], "seed": SEED}, dtr, rounds, valid_sets=[dva], valid_names=["es"],
                  callbacks=[lgb.early_stopping(C["early_stopping"]), lgb.log_evaluation(500)])
    m.save_model(str(mine(name + ".txt")))
    info = {"model": name, "decoy_weight": w, "train_rows": cut, "early_stop_rows": len(y) - cut,
            "best_iteration": m.best_iteration, "fit_sec": round(time.time() - t)}
    log(f"{name}: {info}")
    return m, info


# --------------------------------------------------------------------------- evaluation
def assemble(p_val, t, fold_a):
    """p_val: (q_idx, p_stage2) for the band rows"""
    b = best.join(p_val, on="q_idx", how="left").with_columns(pl.coalesce("p_stage2", "p1").alias("p"))
    parts = [b.filter(pl.col("q_is_val") & (pl.col("p") >= t))]
    if fold_a == "consistent":
        parts.append(b.filter(~pl.col("q_is_val") & (pl.col("p1") >= t)))
    return pl.concat([x.select("q_idx", "s1_idx", "true_s1") for x in parts]).join(SID, on="s1_idx").join(QID, on="q_idx")


def metrics(ids, k=1):
    pairs = ids.select("s1_id", "m_id")
    if k > 1:
        dec = ids.filter(pl.col("true_s1").is_null())
        pairs = pl.concat([pairs] + [dec.select("s1_id", (pl.col("m_id") + f"#dup{j}").alias("m_id")) for j in range(1, k)])
    m = macro_f05(pairs, tp, VAL_IDS)
    pe = m["per_entity"]
    row = {"f05": m["macro_f05"], "precision": m["macro_precision"], "recall": m["macro_recall"], "singleton_acc": m["singleton_acc"],
           "false_singletons": pe.filter((pl.col("n_true") == 0) & (pl.col("n_pred") > 0)).height,
           "predicted_pairs": int(pe["n_pred"].sum()), "empty_s1": pe.filter(pl.col("n_pred") == 0).height}
    for c, f in pe.join(COUNTRY, on="s1_id").group_by("country").agg(pl.col("f").mean()).sort("country").rows():
        row[f"f05_{c}"] = f
    return row


def evaluate(name, p_val, thresholds):
    rows = []
    for t in thresholds:
        for fold_a in ("consistent", "excluded"):
            ids = assemble(p_val, t, fold_a)
            for k in ((1, 2) if fold_a == "consistent" else (1,)):
                rows.append({"model": name, "threshold": t, "fold_a": fold_a, "decoys": f"x{k}", **metrics(ids, k)})
        r = [x for x in rows if x["threshold"] == t]
        log(f"{name:>10s} t={t}: consistent x1 {r[0]['f05']:.6f} | x2 {r[1]['f05']:.6f} | excluded x1 {r[2]['f05']:.6f}")
    return rows


def band_quality(name, p):
    y, dec = V["y"].to_numpy(), V["decoy"].to_numpy()
    return {"model": name, "band_auc": float(roc_auc_score(y, p)), "band_logloss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
            "mean_p_decoy_rows": float(p[dec].mean()), "mean_p_true_rows": float(p[y].mean()),
            "accepted_decoy_rows_0.7": int((p[dec] >= 0.7).sum()), "accepted_true_rows_0.7": int((p[y] >= 0.7).sum())}


def run():
    out = {"smoke": SMOKE}
    if SMOKE:  # controlled: same subset, same rounds, only the weight differs
        sub = T.head(C["smoke_rows"])
        preds = {}
        for w in (1.0, C["decoy_weight"]):
            m, info = fit(sub, w, C["smoke_rounds"], f"smoke_w{w:g}")
            preds[f"w{w:g}"] = (m.predict(V.select(FE).to_numpy().astype(np.float32), num_threads=16).astype(np.float32), info)
        thresholds = C.get("smoke_thresholds", [0.7])
    else:
        base = pl.read_parquet(APPROACH_DIR / R["baseline_val_scored"])
        vb = V.select("q_idx").join(base.select("q_idx", "p_stage2"), on="q_idx", how="left", maintain_order="left")
        preds = {"baseline_02": (vb["p_stage2"].to_numpy(), {"model": "02 stage2_validation_model_cf1_ru0_r8000", "decoy_weight": 1.0,
                                                               "best_iteration": 4460, "train_rows": int(0.9 * T.height)})}
        m, info = fit(T, C["decoy_weight"], C["max_rounds"], f"stage2_validation_model_decoyw{C['decoy_weight']:g}")
        preds["density_w2"] = (m.predict(V.select(FE).to_numpy().astype(np.float32), num_threads=16).astype(np.float32), info)
        thresholds = C["eval_thresholds"]
    rows, quality, infos = [], [], []
    for name, (p, info) in preds.items():
        quality.append(band_quality(name, p))
        infos.append(info)
        rows += evaluate(name, V.select("q_idx").with_columns(pl.Series("p_stage2", p)), thresholds)
        log(f"{name}: band {quality[-1]}")
    tag = "smoke_" if SMOKE else ""
    pl.DataFrame(rows).write_csv(RESULTS_DIR / f"{tag}validation_grid.csv")
    out.update(models=infos, band_quality=quality, grid=rows, runtime_sec=round(time.time() - T0))
    json.dump(out, open(RESULTS_DIR / f"{tag}validation.json", "w"), indent=2, default=float)
    log("done")


if __name__ == "__main__":
    run()
