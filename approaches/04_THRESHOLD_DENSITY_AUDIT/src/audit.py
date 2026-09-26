"""
04_THRESHOLD_DENSITY_AUDIT — Experiment 1: is threshold 0.5 right, and does a test-like decoy density
support a different threshold?  Reuses Approach 02's probabilities; nothing is retrained.

Validation assembly (fold B = S1 id % 5 == 0; the leaderboard metric on fold-B S1 entities):
  * fold-B records: 02's stage-2 probability inside the band [0.002, 0.995], stage-1 p1 outside it
    (exactly what 02 does on test).
  * fold-A records whose top-1 is a fold-B S1 can create false links on fold-B entities. Stage 2 has no
    out-of-fold score for them (they trained it), so three treatments are reported:
      consistent : stage-1 p1 at the SAME threshold  (pessimistic: stage 1 is weaker than stage 2)
      fixed_0.7  : stage-1 p1 at 0.7 whatever the threshold (what the earlier 02 sweep did)
      excluded   : dropped  (optimistic: their false links vanish)

Decoy-density simulation (test has about 2x the decoys per S1 of train, see test_density()):
  every ACCEPTED decoy link (a record with no true S1) is counted k times (fresh, never-true ids) on the
  same S1 ("same"), or its k-1 copies go to uniformly random fold-B S1 entities ("spread").
  This scales the expected number of false links from decoys by k while keeping the decoy acceptance
  rate; k = 1 reproduces the plain metric exactly (checked).

usage (from the approach folder):  python src/audit.py [--smoke] [--no-submissions]
"""
import json
import sys
import time

import paths  # noqa: F401
from paths import APPROACH_DIR, CONFIG, OUTPUT_DIR, RESULTS_DIR, shared

import numpy as np
import polars as pl

from metrics import macro_f05                   # baseline (read-only)
from predict import write_grouped               # baseline (read-only)
from train import id_tables, label_table, query_folds  # baseline (read-only)

C = CONFIG
SMOKE = "--smoke" in sys.argv
T0 = time.time()
RNG = np.random.default_rng(C["seed"])


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def tag(n):
    return f"smoke_{n}" if SMOKE else n


# --------------------------------------------------------------------------- data
s1, q, lab, tp = label_table()
qf = query_folds(lab, s1)
best = pl.read_parquet(shared("train_best.parquet")).join(qf, on="q_idx", how="left")
vs = pl.read_parquet(APPROACH_DIR / C["imported"]["val_scored"]).select("q_idx", "p_stage2")
best = best.join(vs, on="q_idx", how="left").with_columns(pl.coalesce("p_stage2", "p1").alias("p"))
val_s1 = s1.filter("is_val")
if SMOKE:
    val_s1 = val_s1.head(20_000)
VAL_IDS = val_s1["entity_id"]
SID = s1.select("s1_idx", pl.col("entity_id").alias("s1_id"))
QID = q.select("q_idx", pl.col("entity_id").alias("m_id"))
COUNTRY = val_s1.select(pl.col("entity_id").alias("s1_id"), "country")
log(f"loaded: {best.height:,} records, {val_s1.height:,} validation S1")


def pairs_at(t, fold_a):
    v = best.filter(pl.col("q_is_val") & (pl.col("p") >= t))
    parts = [v]
    if fold_a == "consistent":
        parts.append(best.filter(~pl.col("q_is_val") & (pl.col("p1") >= t)))
    elif fold_a == "fixed_0.7":
        parts.append(best.filter(~pl.col("q_is_val") & (pl.col("p1") >= 0.7)))
    return pl.concat([x.select("q_idx", "s1_idx", "true_s1") for x in parts]).join(SID, on="s1_idx").join(QID, on="q_idx")


def with_decoys(ids, k, mode):
    """count every accepted decoy link k times"""
    if k == 1:
        return ids.select("s1_id", "m_id")
    dec = ids.filter(pl.col("true_s1").is_null() & pl.col("s1_id").is_in(VAL_IDS.implode()))
    extra = []
    for j in range(1, k):
        if mode == "same":
            extra.append(dec.select("s1_id", (pl.col("m_id") + f"#dup{j}").alias("m_id")))
        else:  # spread to random validation S1 entities
            r = RNG.integers(0, len(VAL_IDS), dec.height)
            extra.append(pl.DataFrame({"s1_id": VAL_IDS.gather(r), "m_id": (dec["m_id"] + f"#dup{j}")}))
    return pl.concat([ids.select("s1_id", "m_id")] + extra)


def metrics(pairs):
    m = macro_f05(pairs, tp, VAL_IDS)
    pe = m["per_entity"]
    row = {"f05": m["macro_f05"], "precision": m["macro_precision"], "recall": m["macro_recall"],
           "singleton_acc": m["singleton_acc"],
           "false_singletons": pe.filter((pl.col("n_true") == 0) & (pl.col("n_pred") > 0)).height,
           "predicted_pairs": int(pe["n_pred"].sum()), "empty_s1": pe.filter(pl.col("n_pred") == 0).height}
    byc = pe.join(COUNTRY, on="s1_id").group_by("country").agg(pl.col("f").mean())
    for c, f in byc.rows():
        row[f"f05_{c}"] = f
    return row, pe.sort("s1_id")["f"].to_numpy()


# --------------------------------------------------------------------------- 1. correctness checks of the simulation
def checks():
    out = {}
    ids = pairs_at(0.7, "consistent")
    plain, _ = metrics(ids.select("s1_id", "m_id"))
    k1, _ = metrics(with_decoys(ids, 1, "same"))
    out["k1_equals_plain"] = abs(plain["f05"] - k1["f05"]) < 1e-12
    k2 = with_decoys(ids, 2, "same")
    dec = ids.filter(pl.col("true_s1").is_null() & pl.col("s1_id").is_in(VAL_IDS.implode()))
    out["accepted_decoy_links_at_0.7"] = dec.height
    out["k2_added_rows"] = k2.height - ids.height
    out["k2_added_rows_equals_decoys"] = (k2.height - ids.height) == dec.height
    dup_true = k2.join(tp, on=["s1_id", "m_id"], how="semi").height
    out["k2_true_positives_unchanged"] = dup_true == ids.select("s1_id", "m_id").join(tp, on=["s1_id", "m_id"], how="semi").height
    m1, m2 = macro_f05(ids.select("s1_id", "m_id"), tp, VAL_IDS)["per_entity"], macro_f05(k2, tp, VAL_IDS)["per_entity"]
    j = m1.join(m2, on="s1_id", suffix="_k2")
    out["k2_tp_identical_per_entity"] = bool((j["tp"] == j["tp_k2"]).all())
    out["k2_npred_never_decreases"] = bool((j["n_pred_k2"] >= j["n_pred"]).all())
    log(f"simulation checks: {out}")
    return out


# --------------------------------------------------------------------------- 2. test decoy density estimate
def test_density():
    tsc = pl.read_parquet(APPROACH_DIR / C["imported"]["test_scored"])
    tq = pl.concat([pl.read_parquet(shared(f"test_{s}_norm.parquet"), columns=["country"]) for s in ("source2", "source3")]).with_row_index("q_idx")
    ts1 = pl.read_parquet(shared("test_source1.parquet"), columns=["country"])
    tr_q = pl.concat([pl.read_parquet(shared(f"train_{s}_norm.parquet"), columns=["country"]) for s in ("source2", "source3")])
    tr_s1 = pl.read_parquet(shared("train_source1.parquet"), columns=["country"])
    n_true = lab.height
    val_best = best.filter(pl.col("q_is_val"))
    rec_rate = float(((val_best["s1_idx"] == val_best["true_s1"]) & (val_best["p"] >= 0.7)).sum() / val_best["true_s1"].is_not_null().sum())
    fp_rate = float(val_best.filter(pl.col("true_s1").is_null())["p"].ge(0.7).mean())
    acc_test = tsc.join(tq, on="q_idx").group_by("country").agg((pl.col("p_final") >= 0.7).mean().alias("acc"), pl.len().alias("q"))
    rows = []
    for c, acc, nq in acc_test.rows():
        ns1 = int((ts1["country"] == c).sum())
        match_share = (acc - fp_rate) / (rec_rate - fp_rate)       # method 1: invert acceptance
        rows.append({"country": c, "records_per_s1": nq / ns1, "accept_rate_0.7": acc,
                     "est_decoy_share": 1 - match_share, "est_decoys_per_s1": nq / ns1 * (1 - match_share)})
    train = {"records_per_s1": tr_q.height / tr_s1.height, "decoy_share": 1 - n_true / tr_q.height,
             "decoys_per_s1": (tr_q.height - n_true) / tr_s1.height}
    res = {"train": train, "test": rows, "val_record_recall_0.7": rec_rate, "val_decoy_accept_0.7": fp_rate}
    log(f"train: {train}")
    for r in rows:
        log(f"test {r['country']}: {r['records_per_s1']:.2f} records/S1, est. decoy share {r['est_decoy_share']:.3f}, "
            f"est. decoys/S1 {r['est_decoys_per_s1']:.2f} (x{r['est_decoys_per_s1'] / train['decoys_per_s1']:.2f} train)")
    return res


# --------------------------------------------------------------------------- 3. threshold x density grid + bootstrap
def grid():
    rows, fvec = [], {}
    for mode in ("consistent", "fixed_0.7", "excluded"):
        for t in C["thresholds"]:
            ids = pairs_at(t, mode)
            for k in C["decoy_multipliers"]:
                for spread in (("same", "spread") if k == 2 else ("same",)):
                    if mode != "consistent" and (k != 1 or spread != "same"):
                        continue  # density runs only for the primary treatment
                    r, f = metrics(with_decoys(ids, k, spread))
                    r.update(fold_a=mode, threshold=t, decoys=f"x{k}" + ("" if spread == "same" else "_spread"))
                    rows.append(r)
                    fvec[(mode, t, r["decoys"])] = f
            log(f"{mode:10s} t={t}: " + " | ".join(f"{r['decoys']} {r['f05']:.6f}" for r in rows if r["fold_a"] == mode and r["threshold"] == t))
    df = pl.DataFrame(rows).select("fold_a", "decoys", "threshold", *[c for c in rows[0] if c not in ("fold_a", "decoys", "threshold")])
    df.write_csv(RESULTS_DIR / tag("threshold_grid.csv"))
    # paired bootstrap (resample S1 entities): F(t) - F(0.5) and F(t) - F(0.7)
    n = len(next(iter(fvec.values())))
    B = 50 if SMOKE else C["bootstrap_resamples"]
    idx = RNG.integers(0, n, (B, n))
    boot = []
    for dens in ("x1", "x2", "x3"):
        for ref in (0.5, 0.7):
            base = fvec[("consistent", ref, dens)]
            for t in C["thresholds"]:
                d = fvec[("consistent", t, dens)] - base
                bs = np.array([d[i].mean() for i in idx])
                boot.append({"decoys": dens, "vs": ref, "threshold": t, "delta": float(d.mean()),
                             "ci_low": float(np.quantile(bs, 0.025)), "ci_high": float(np.quantile(bs, 0.975))})
    pl.DataFrame(boot).write_csv(RESULTS_DIR / tag("bootstrap_deltas.csv"))
    return df, pl.DataFrame(boot)


# --------------------------------------------------------------------------- 4. alternative test submissions from 02's probabilities
def submissions():
    tsc = pl.read_parquet(APPROACH_DIR / C["imported"]["test_scored"])
    s1_t, q_t = id_tables("test")
    ts1 = pl.read_parquet(shared("test_source1.parquet"), columns=["country"]).with_row_index("s1_idx")
    out = []
    for t in C["thresholds"]:
        m = tsc.filter(pl.col("p_final") >= t).select("s1_idx", "q_idx")
        path = OUTPUT_DIR / f"matching_results_t{t:.2f}.tsv"
        if SMOKE:
            m, path = m.head(1000), RESULTS_DIR / f"smoke_matching_results_t{t:.2f}.tsv"
        n, n_m = write_grouped(s1_t["entity_id"], q_t["entity_id"], m, "matched_entity_ids", path)
        byc = m.join(ts1, on="s1_idx").group_by("country").len()
        out.append({"threshold": t, "file": str(path), "matched_records": m.height, "s1_with_match": n_m, "empty_s1": n - n_m,
                    **{f"matched_{c}": k for c, k in byc.rows()}})
        log(f"test t={t}: {m.height:,} matched records, {n - n_m:,} empty S1 -> {path.name}")
    pl.DataFrame(out).write_csv(RESULTS_DIR / tag("test_submissions.csv"))
    return out


if __name__ == "__main__":
    summary = {"checks": checks(), "test_density": test_density()}
    df, boot = grid()
    if "--no-submissions" not in sys.argv:
        summary["submissions"] = submissions()
    summary["runtime_sec"] = round(time.time() - T0)
    json.dump(summary, open(RESULTS_DIR / tag("audit_summary.json"), "w"), indent=2, default=str)
    pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
    print(df.select("fold_a", "decoys", "threshold", "f05", "precision", "recall", "singleton_acc", "false_singletons",
                    "predicted_pairs", "empty_s1", "f05_India", "f05_US"))
    print(boot)
    log("done")
