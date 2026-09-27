"""Experiment 11 runner (harness copied from Experiment 10; recipe identical to Approach 02 stage2.fit).

usage: python src/run.py <variant_name> [group ...]     groups: hn (Exp 10 house-number), ate (address token odds)
Writes: cache/feat_<group>_<table>.parquet, cache/model_<variant>.txt, results/runs.jsonl
"""
import json
import sys
import time
from pathlib import Path

import polars as pl

import ate
import common
from common import BASE_FEATURES, Evaluator, fit, load_rows, log, norm_pairs
from newfeats import GROUPS

APP = Path(__file__).resolve().parents[1]
CACHE, RES = APP / "cache", APP / "results"
EXP10_CACHE = common.ROOT / "approaches/10_STAGE2_ADDRESS_SIGNALS/cache"  # read-only reuse of hn features


def hn_table(name, rows):
    for path in (EXP10_CACHE / f"feat_hn_{name}.parquet", CACHE / f"feat_hn_{name}.parquet"):
        if path.exists():
            return pl.read_parquet(path)
    f = pl.concat([rows.select("q_idx"), GROUPS["hn"][1](norm_pairs(rows))], how="horizontal")
    f.write_parquet(CACHE / f"feat_hn_{name}.parquet")
    return f


def ate_tables(T, V):
    pt, pv = CACHE / "feat_ate_train.parquet", CACHE / "feat_ate_val.parquet"
    if pt.exists() and pv.exists():
        return pl.read_parquet(pt), pl.read_parquet(pv)
    t = time.time()
    _, _, _, best = common.stage2.train_rows()
    best = best.with_columns(pl.col("q_is_val").fill_null(False))
    pool = best.filter(~pl.col("q_is_val") & pl.col("oos")).select("q_idx", "s1_idx", "y", "te_half")
    pp = norm_pairs(pool).join(pool.with_row_index("_r").select("_r", "y", "te_half"), on="_r")
    log(f"ATE pool rows {pp.height:,}")
    te_h1 = ate.learn(pp.filter(pl.col("te_half")))     # applied to half-0 rows
    te_h0 = ate.learn(pp.filter(~pl.col("te_half")))    # applied to half-1 rows
    te_all = ate.learn(pp)                              # applied to validation rows
    log(f"ATE tables: qo {te_all['qo'].height:,} / so {te_all['so'].height:,} tokens")
    del pp, pool
    half = best.select("q_idx", "te_half")
    Tr = T.select("q_idx", "s1_idx").join(half, on="q_idx", how="left", maintain_order="left")
    parts = []
    for flag, te in ((False, te_h1), (True, te_h0)):
        rows = Tr.filter(pl.col("te_half") == flag)
        parts.append(pl.concat([rows.select("q_idx"), ate.apply(norm_pairs(rows), te)], how="horizontal"))
    ft = pl.concat(parts)
    fv = pl.concat([V.select("q_idx"), ate.apply(norm_pairs(V), te_all)], how="horizontal")
    ft.write_parquet(pt)
    fv.write_parquet(pv)
    log(f"ATE features built in {time.time() - t:.0f}s")
    return ft, fv


def main():
    variant, groups = sys.argv[1], sys.argv[2:]
    t0 = time.time()
    T, V = load_rows()
    feats = list(BASE_FEATURES)
    if "hn" in groups:
        T = T.join(hn_table("train", T), on="q_idx", how="left", maintain_order="left")
        V = V.join(hn_table("val", V), on="q_idx", how="left", maintain_order="left")
        feats += GROUPS["hn"][0]
    if "ate" in groups:
        ft, fv = ate_tables(T, V)
        T = T.join(ft, on="q_idx", how="left", maintain_order="left")
        V = V.join(fv, on="q_idx", how="left", maintain_order="left")
        feats += ate.ATE_COLS
    log(f"{variant}: {len(feats)} features, train {T.height:,}, val {V.height:,}")
    m = fit(T, feats, CACHE / f"model_{variant}.txt")
    p = m.predict(V.select(feats).to_numpy().astype("float32"), num_threads=16)
    pl.DataFrame({"q_idx": V["q_idx"], "p_stage2": p}).write_parquet(CACHE / f"val_scored_{variant}.parquet")
    del T
    res = Evaluator().evaluate(variant, V, p)
    res.update({"groups": groups, "n_features": len(feats), "best_iteration": m.best_iteration,
                "runtime_sec": round(time.time() - t0), "delta_vs_a02_saved": res["f05"] - common.EXPECTED_BASE})
    with open(RES / "runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(res) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
