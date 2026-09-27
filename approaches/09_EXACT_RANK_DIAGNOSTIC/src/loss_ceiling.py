"""Experiment 09b: oracle F0.5 ceilings per error bucket + Stage-2 rejection profile (validation fold B).

Read-only. Reassembles the Approach 02 validation decisions exactly as Experiment 08 did (asserted to
reproduce 0.98568978), then replaces ONE bucket of fold-B query decisions with the oracle decision and
recomputes macro F0.5. This bounds the maximum gain any method targeting that bucket could deliver.
Writes only to approaches/09_EXACT_RANK_DIAGNOSTIC/results/.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
A02 = ROOT / "approaches/02_STAGE2_CROSSFIT_RUNNERUP"
sys.path.insert(0, str(ROOT / "business_entity_resolution/src"))
sys.path.insert(0, str(A02 / "src"))
import stage2  # noqa: E402  (read-only use: train_rows, to_ids)
from metrics import macro_f05  # noqa: E402

THR = 0.70
EXPECTED = 0.9856897798505745
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def main():
    s1, q, tp, best = stage2.train_rows()
    flags = pl.concat([pl.read_parquet(ROOT / f"cache/train_{s}_norm.parquet", columns=["addr_missing"])
                       for s in ("source2", "source3")]).with_row_index("q_idx")
    scored = pl.read_parquet(A02 / "cache/validate_val_scored_cf1_ru0_r8000.parquet", columns=["q_idx", "p_stage2"])
    b = (best.select("q_idx", "s1_idx", "p1", "true_s1", "q_is_val", "y")
         .join(scored, on="q_idx", how="left").join(flags, on="q_idx", how="left")
         .with_columns(pl.coalesce("p_stage2", "p1").alias("p_final"), pl.col("q_is_val").fill_null(False)))
    val_pairs = b.filter(pl.col("q_is_val") & pl.col("true_s1").is_not_null()).select("q_idx", pl.col("true_s1").alias("s1_idx"))
    log(f"locating true S1 in saved Top-10 for {val_pairs.height:,} fold-B matched queries")
    in10 = pl.concat([pl.read_parquet(f, columns=["q_idx", "s1_idx"]).join(val_pairs, on=["q_idx", "s1_idx"], how="semi")
                      for f in sorted((ROOT / "cache/train_cands").glob("*.parquet"))]).select("q_idx").unique()
    b = b.join(in10.with_columns(pl.lit(True).alias("t10")), on="q_idx", how="left").with_columns(
        pl.col("t10").fill_null(False), (pl.col("p_final") >= THR).alias("acc"), pl.col("p_stage2").is_not_null().alias("band"))
    val_ids = s1.filter(pl.col("is_val"))["entity_id"]

    V, Y, M = pl.col("q_is_val"), pl.col("y"), pl.col("true_s1").is_not_null()
    T10, BAND, ACC = pl.col("t10"), pl.col("band"), pl.col("acc")
    groups = {"all": pl.lit(True), "addr_present": ~pl.col("addr_missing"), "addr_missing": pl.col("addr_missing")}

    def evaluate(name, acc_expr, s1_expr=pl.col("s1_idx")):
        pred = b.with_columns(s1_expr.alias("_s1"), acc_expr.alias("_acc")).filter(pl.col("_acc")).select("q_idx", pl.col("_s1").alias("s1_idx"))
        m = macro_f05(stage2.to_ids(pred, s1, q), tp, val_ids)
        pe = m["per_entity"]
        row = {"variant": name, "f05": m["macro_f05"], "precision": m["macro_precision"], "recall": m["macro_recall"],
               "singleton_acc": m["singleton_acc"],
               "false_singletons": pe.filter((pl.col("n_true") == 0) & (pl.col("n_pred") > 0)).height,
               "accepted_pairs_all_folds": pred.height}
        log(f"{name:34s} F0.5 {row['f05']:.6f}")
        return row

    base = evaluate("base_approach02_t0.70", ACC)
    assert abs(base["f05"] - EXPECTED) < 1e-8, f"reconstruction {base['f05']} != {EXPECTED}"
    rows = [base]
    for g, G in groups.items():
        sel = V & G
        rows.append(evaluate(f"stage2_oracle_inband__{g}", pl.when(sel & BAND).then(Y).otherwise(ACC)))
        rows.append(evaluate(f"accept_all_correct_top1__{g}", pl.when(sel & Y).then(True).otherwise(ACC)))
        rows.append(evaluate(f"reject_all_wrong_top1__{g}", pl.when(sel & ~Y).then(False).otherwise(ACC)))
        fix_rank = sel & M & T10 & ~Y
        rows.append(evaluate(f"oracle_rerank_top10__{g}", pl.when(fix_rank).then(True).otherwise(ACC),
                             pl.when(fix_rank).then(pl.col("true_s1")).otherwise(pl.col("s1_idx"))))
        fix_ret = sel & M & ~T10
        rows.append(evaluate(f"oracle_retrieval_miss__{g}", pl.when(fix_ret).then(True).otherwise(ACC),
                             pl.when(fix_ret).then(pl.col("true_s1")).otherwise(pl.col("s1_idx"))))
        rows.append(evaluate(f"perfect__{g}", pl.when(sel).then(M).otherwise(ACC),
                             pl.when(sel & M).then(pl.col("true_s1")).otherwise(pl.col("s1_idx"))))
    for r in rows:
        r["delta_vs_base"] = r["f05"] - base["f05"]
    pl.DataFrame(rows).write_csv(OUT / "loss_ceiling.csv")

    # fold-B bucket counts (queries), the denominators behind each ceiling
    vb = b.filter(V)
    counts = {}
    for g, G in groups.items():
        d = vb.filter(G)
        counts[g] = {
            "queries": d.height, "matched": d.filter(M).height,
            "accepted_correct": d.filter(ACC & pl.col("y")).height,
            "accepted_wrong": d.filter(ACC & ~pl.col("y")).height,
            "accepted_wrong_unmatched_query": d.filter(ACC & ~M).height,
            "accepted_wrong_other_s1": d.filter(ACC & M & ~pl.col("y")).height,
            "rejected_correct_top1": d.filter(~ACC & pl.col("y")).height,
            "wrong_top1_true_in_top10": d.filter(M & T10 & ~pl.col("y")).height,
            "true_not_in_top10": d.filter(M & ~T10).height,
        }

    # profile of in-band Stage-2 rows (rejected correct vs accepted correct vs accepted wrong)
    feats = ["f_cos_name", "f_cos_addr", "f_rank", "f_comb_gap", "f_nm_exact", "f_ad_tset", "f_num_first_eq",
             "f_num_conflict", "f_s1_name_dup", "f_n_name_ties", "f_q_is_s3", "f_q_indic", "f_q_domain", "f_nm_tset",
             "e_hn_edit", "e_ocr_hits", "p1", "gap"]
    vr = pl.read_parquet(A02 / "cache/validate_val_rows_cf1_ru0.parquet", columns=["q_idx", "y", "f_q_addr_missing", *feats]) \
        .join(scored, on="q_idx").join(b.select("q_idx", "true_s1"), on="q_idx", how="left")
    vr = vr.with_columns(
        pl.when(pl.col("y") & (pl.col("p_stage2") < THR)).then(pl.lit("rejected_correct"))
          .when(pl.col("y")).then(pl.lit("accepted_correct"))
          .when(pl.col("p_stage2") >= THR).then(pl.lit("accepted_wrong"))
          .otherwise(pl.lit("rejected_wrong")).alias("cls"),
        pl.when(pl.col("f_q_addr_missing") > 0).then(pl.lit("addr_missing")).otherwise(pl.lit("addr_present")).alias("grp"))
    prof = vr.group_by("grp", "cls").agg(pl.len().alias("n"), *[pl.col(f).mean().round(4).alias(f) for f in feats]).sort("grp", "cls")
    prof.write_csv(OUT / "stage2_band_profile.csv")
    edges = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    hist = vr.with_columns(pl.col("p_stage2").cut(edges[1:-1], left_closed=True).alias("p_bin")) \
        .group_by("grp", "p_bin").agg(pl.col("y").sum().alias("correct_top1"), (~pl.col("y")).sum().alias("wrong_top1")) \
        .sort("grp", "p_bin")
    hist.write_csv(OUT / "stage2_band_hist.csv")
    # what makes rejected-correct address-present rows different: flag rates
    rc = vr.filter((pl.col("grp") == "addr_present") & (pl.col("cls") == "rejected_correct"))
    flag_rates = {
        "n": rc.height,
        "s1_name_dup>1": rc.filter(pl.col("f_s1_name_dup") > 1).height,
        "n_name_ties>=1": rc.filter(pl.col("f_n_name_ties") >= 1).height,
        "rank>1_in_retrieval": rc.filter(pl.col("f_rank") > 1).height,
        "num_conflict": rc.filter(pl.col("f_num_conflict") > 0).height,
        "first_num_not_equal": rc.filter(pl.col("f_num_first_eq") < 1).height,
        "cos_name<0.5": rc.filter(pl.col("f_cos_name") < 0.5).height,
        "cos_addr<0.5": rc.filter(pl.col("f_cos_addr") < 0.5).height,
        "p_stage2_in_[0.5,0.7)": rc.filter(pl.col("p_stage2") >= 0.5).height,
        "p_stage2_<0.2": rc.filter(pl.col("p_stage2") < 0.2).height,
    }
    summary = {"threshold": THR, "base": base, "bucket_counts_foldB_queries": counts,
               "ceilings": {r["variant"]: round(r["delta_vs_base"], 6) for r in rows[1:]},
               "addr_present_rejected_correct_flags": flag_rates, "runtime_sec": round(time.time() - T0)}
    (OUT / "loss_ceiling_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float), flush=True)
    print(prof, flush=True)
    print(hist, flush=True)


if __name__ == "__main__":
    main()
