"""Shared, read-only experiment harness for Stage-2 feature experiments (Experiments 10+).

* Training / validation rows: the saved Approach 02 validate tables (fold A train halves h0/h1, fold-B band rows).
* Training recipe: identical to Approach 02 `stage2.fit` (same shuffle, 90/10 early-stopping split, LightGBM
  params, categorical columns, 8000 rounds, early stopping 100). Extra feature columns are appended.
* Evaluation: identical decision assembly to Experiment 08/09 (fold-B band rows use Stage-2, everything else
  keeps its Approach 02 decision), macro F0.5 over fold-B S1 at threshold 0.70.
Nothing under approaches/02_* or cache/ is written.
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
A02 = ROOT / "approaches/02_STAGE2_CROSSFIT_RUNNERUP"
sys.path.insert(0, str(ROOT / "business_entity_resolution/src"))
sys.path.insert(0, str(A02 / "src"))
import stage2  # noqa: E402  (read-only: train_rows, to_ids, STAGE2_FEATURES)
from metrics import macro_f05  # noqa: E402

A02_CFG = json.load(open(A02 / "config/config.json", encoding="utf-8"))
BASE_FEATURES = list(stage2.BASE_FEATURES)
THR = 0.70
SEED = 42
EXPECTED_BASE = 0.9856897798505745
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def load_rows():
    """(T, V): Approach 02 training rows (shuffled exactly as stage2.build_training) and fold-B band rows."""
    h0 = pl.read_parquet(A02 / "cache/validate_train_rows_h0_cf1_ru0.parquet")
    h1 = pl.read_parquet(A02 / "cache/validate_train_rows_h1_cf1_ru0.parquet")
    T = pl.concat([h0, h1]).sample(fraction=1.0, shuffle=True, seed=SEED)
    V = pl.read_parquet(A02 / "cache/validate_val_rows_cf1_ru0.parquet")
    return T, V


def fit(T, features, model_path, params_override=None):
    X, y = T.select(features).to_numpy().astype(np.float32), T["y"].to_numpy()
    cut = int(0.9 * len(y))
    params = {**A02_CFG["lgbm"], "seed": SEED, **(params_override or {})}
    cat = [c for c in A02_CFG["categorical"] if c in features]
    dtr = lgb.Dataset(X[:cut], y[:cut], feature_name=features, categorical_feature=cat)
    dva = lgb.Dataset(X[cut:], y[cut:], reference=dtr)
    t = time.time()
    m = lgb.train(params, dtr, A02_CFG["max_rounds"], valid_sets=[dva], valid_names=["es"],
                  callbacks=[lgb.early_stopping(A02_CFG["early_stopping"], verbose=False), lgb.log_evaluation(1000)])
    m.save_model(str(model_path))
    log(f"fit {model_path.name}: best_iteration {m.best_iteration}, {time.time() - t:.0f}s")
    imp = sorted(zip(features, m.feature_importance("gain")), key=lambda x: -x[1])
    tot = sum(g for _, g in imp) or 1.0
    pl.DataFrame({"feature": [f for f, _ in imp], "gain_pct": [100 * g / tot for _, g in imp]}).write_csv(
        model_path.with_suffix(".importance.csv"))
    return m


class Evaluator:
    """Loads the fold-B decision context once; evaluate(p_stage2 for band rows) -> metrics dict."""

    def __init__(self):
        s1, q, tp, best = stage2.train_rows()
        flags = pl.concat([pl.read_parquet(ROOT / f"cache/train_{s}_norm.parquet", columns=["addr_missing"])
                           for s in ("source2", "source3")]).with_row_index("q_idx")
        self.s1, self.q, self.tp = s1, q, tp
        self.best = best.select("q_idx", "s1_idx", "p1", "y", "q_is_val").join(flags, on="q_idx", how="left")
        self.val_ids = s1.filter(pl.col("is_val"))["entity_id"]

    def evaluate(self, name, V, p):
        """V: fold-B band rows (q_idx, y, f_q_addr_missing); p: Stage-2 probabilities aligned with V."""
        sc = V.select("q_idx", "y", "f_q_addr_missing").with_columns(pl.Series("p_stage2", np.asarray(p, dtype=np.float32)))
        b = self.best.join(sc.select("q_idx", "p_stage2"), on="q_idx", how="left").with_columns(
            pl.coalesce("p_stage2", "p1").alias("p_final"))
        pred = b.filter(pl.col("p_final") >= THR).select("q_idx", "s1_idx")
        m = macro_f05(stage2.to_ids(pred, self.s1, self.q), self.tp, self.val_ids)
        pe = m["per_entity"]
        ap = sc.filter(pl.col("f_q_addr_missing") == 0)
        am = sc.filter(pl.col("f_q_addr_missing") > 0)
        out = {"variant": name, "f05": m["macro_f05"], "precision": m["macro_precision"], "recall": m["macro_recall"],
               "singleton_acc": m["singleton_acc"],
               "false_singletons": pe.filter((pl.col("n_true") == 0) & (pl.col("n_pred") > 0)).height,
               "accepted_pairs": pred.height,
               "band_auc": float(roc_auc_score(sc["y"], sc["p_stage2"])),
               "band_auc_addr_present": float(roc_auc_score(ap["y"], ap["p_stage2"])),
               "band_auc_addr_missing": float(roc_auc_score(am["y"], am["p_stage2"])),
               "ap_rejected_correct": ap.filter(pl.col("y") & (pl.col("p_stage2") < THR)).height,
               "ap_accepted_wrong": ap.filter(~pl.col("y") & (pl.col("p_stage2") >= THR)).height,
               "am_rejected_correct": am.filter(pl.col("y") & (pl.col("p_stage2") < THR)).height,
               "am_accepted_wrong": am.filter(~pl.col("y") & (pl.col("p_stage2") >= THR)).height}
        for c in ("India", "US"):
            ids = self.s1.filter(pl.col("is_val") & (pl.col("country") == c))["entity_id"]
            out[f"f05_{c}"] = macro_f05(stage2.to_ids(pred, self.s1, self.q), self.tp, ids)["macro_f05"]
        log(f"{name}: F0.5 {out['f05']:.6f} P {out['precision']:.5f} R {out['recall']:.5f} "
            f"FS {out['false_singletons']} AUC {out['band_auc']:.5f} ap_rej_corr {out['ap_rejected_correct']} ap_acc_wrong {out['ap_accepted_wrong']}")
        return out


def norm_pairs(rows):
    """Attach normalized query / S1 fields to (q_idx, s1_idx) rows, order-preserving."""
    qn = pl.concat([pl.read_parquet(ROOT / f"cache/train_{s}_norm.parquet", columns=["name_core", "addr_norm", "addr_nums"])
                    for s in ("source2", "source3")]).with_row_index("q_idx")
    sn = pl.read_parquet(ROOT / "cache/train_source1_norm.parquet", columns=["name_core", "addr_norm", "addr_nums"]).with_row_index("s1_idx")
    return rows.select("q_idx", "s1_idx").with_row_index("_r").join(qn, on="q_idx", how="left") \
        .join(sn, on="s1_idx", how="left", suffix="_s").sort("_r")
