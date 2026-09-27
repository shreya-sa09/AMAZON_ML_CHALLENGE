"""Experiment 10 runner: controlled Stage-2 refits on the Approach 02 validation tables.

usage: python src/run.py <variant_name> [group ...]      e.g.  run.py R0_base      run.py R1_hn hn
Every variant uses the identical rows / shuffle / recipe; only the appended feature groups differ.
Writes: cache/feat_<group>_<table>.parquet, cache/model_<variant>.txt, results/runs.jsonl
"""
import json
import sys
import time
from pathlib import Path

import polars as pl

import common
from common import BASE_FEATURES, Evaluator, fit, load_rows, log, norm_pairs
from newfeats import GROUPS

APP = Path(__file__).resolve().parents[1]
CACHE, RES = APP / "cache", APP / "results"


def group_table(group, name, rows):
    path = CACHE / f"feat_{group}_{name}.parquet"
    if path.exists():
        return pl.read_parquet(path)
    cols, fn = GROUPS[group]
    t = time.time()
    f = pl.concat([rows.select("q_idx"), fn(norm_pairs(rows))], how="horizontal")
    f.write_parquet(path)
    log(f"features {group}/{name}: {f.height:,} rows in {time.time() - t:.0f}s")
    return f


def main():
    variant, groups = sys.argv[1], sys.argv[2:]
    t0 = time.time()
    T, V = load_rows()
    feats = list(BASE_FEATURES)
    for g in groups:
        T = T.join(group_table(g, "train", T), on="q_idx", how="left", maintain_order="left")
        V = V.join(group_table(g, "val", V), on="q_idx", how="left", maintain_order="left")
        feats += GROUPS[g][0]
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
