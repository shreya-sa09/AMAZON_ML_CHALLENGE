"""Experiment 13 runner (harness copied from Exp 11; recipe identical to Approach 02 stage2.fit; ER_LGB_SEED optional).

usage: python src/run.py <variant_name> [group ...]     groups: hn (Exp 10, reused read-only), sib (sibling evidence)
Writes: cache/feat_sib_<table>.parquet, cache/model_<variant>.txt, cache/val_scored_<variant>.parquet, results/runs.jsonl
"""
import json
import sys
import time
from pathlib import Path

import polars as pl

import common
import sib
from common import BASE_FEATURES, Evaluator, fit, load_rows, log
from newfeats import GROUPS

APP = Path(__file__).resolve().parents[1]
CACHE, RES = APP / "cache", APP / "results"
EXP10_CACHE = common.ROOT / "approaches/10_STAGE2_ADDRESS_SIGNALS/cache"


def sib_tables(T, V):
    pt, pv = CACHE / "feat_sib_train.parquet", CACHE / "feat_sib_val.parquet"
    if pt.exists() and pv.exists():
        return pl.read_parquet(pt), pl.read_parquet(pv)
    t = time.time()
    best = pl.read_parquet(common.ROOT / "cache/train_best.parquet", columns=["q_idx", "s1_idx"])
    qn = pl.concat([pl.read_parquet(common.ROOT / f"cache/train_{s}_norm.parquet", columns=["name_core", "addr_norm"])
                    .with_columns(pl.lit(s == "source3").alias("is_s3")) for s in ("source2", "source3")]).with_row_index("q_idx")
    sn = pl.read_parquet(common.ROOT / "cache/train_source1_norm.parquet", columns=["addr_norm"]).with_row_index("s1_idx")
    out = []
    for name, rows, path in (("train", T, pt), ("val", V, pv)):
        f = pl.concat([rows.select("q_idx"), sib.sib_features(rows, best, qn, sn, rows["f_ad_ratio"])], how="horizontal")
        f.write_parquet(path)
        log(f"sib features {name}: {f.height:,} rows")
        out.append(f)
    log(f"sib features built in {time.time() - t:.0f}s")
    return out


def main():
    variant, groups = sys.argv[1], sys.argv[2:]
    t0 = time.time()
    T, V = load_rows()
    feats = list(BASE_FEATURES)
    if "hn" in groups:
        T = T.join(pl.read_parquet(EXP10_CACHE / "feat_hn_train.parquet"), on="q_idx", how="left", maintain_order="left")
        V = V.join(pl.read_parquet(EXP10_CACHE / "feat_hn_val.parquet"), on="q_idx", how="left", maintain_order="left")
        feats += GROUPS["hn"][0]
    if "sib" in groups:
        ft, fv = sib_tables(T, V)
        T = T.join(ft, on="q_idx", how="left", maintain_order="left")
        V = V.join(fv, on="q_idx", how="left", maintain_order="left")
        feats += sib.SIB_COLS
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
