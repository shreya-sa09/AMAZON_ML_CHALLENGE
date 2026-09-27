"""Test-shift attribution: does the house-number-only model (Exp 10 R1, fold A) also shift test acceptances?
Plus raw examples of US test rows newly accepted by R4 vs Approach 02 (plausibility check). Read-only."""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

import common
from common import A02, BASE_FEATURES
from newfeats import HN_COLS, hn_features

ROOT = common.ROOT
OUT = Path(__file__).resolve().parents[1] / "results"
sys.stdout.reconfigure(encoding="utf-8")
lines = []


def main():
    X = pl.read_parquet(A02 / "cache/final_test_rows_cf1_ru0.parquet")
    qn = pl.concat([pl.read_parquet(ROOT / f"cache/test_{s}_norm.parquet", columns=["addr_nums"]) for s in ("source2", "source3")]).with_row_index("q_idx")
    sn = pl.read_parquet(ROOT / "cache/test_source1_norm.parquet", columns=["addr_nums"]).with_row_index("s1_idx")
    pairs = X.select("q_idx", "s1_idx").with_row_index("_r").join(qn, on="q_idx", how="left").join(sn, on="s1_idx", how="left", suffix="_s").sort("_r")
    X = pl.concat([X, hn_features(pairs)], how="horizontal")
    cty = pl.concat([pl.read_parquet(ROOT / f"cache/test_{s}.parquet", columns=["country", "business_name", "business_address"])
                     for s in ("source2", "source3")]).with_row_index("q_idx")
    X = X.join(cty.select("q_idx", "country"), on="q_idx", how="left", maintain_order="left")
    pa = lgb.Booster(model_file=str(A02 / "cache/stage2_validation_model_cf1_ru0_r8000.txt")).predict(
        X.select(BASE_FEATURES).to_numpy().astype(np.float32), num_threads=16)
    p1 = lgb.Booster(model_file=str(ROOT / "approaches/10_STAGE2_ADDRESS_SIGNALS/cache/model_R1_hn.txt")).predict(
        X.select(BASE_FEATURES + HN_COLS).to_numpy().astype(np.float32), num_threads=16)
    X = X.with_columns(pl.Series("pa", pa), pl.Series("p1hn", p1))
    g = X.group_by("country").agg((pl.col("pa") >= 0.7).sum().alias("A02"), (pl.col("p1hn") >= 0.7).sum().alias("R1_hn")).sort("country")
    lines.append(f"TEST band acceptances, fold-A models:\n{g}")
    tot_a, tot_1 = int(g["A02"].sum()), int(g["R1_hn"].sum())
    lines.append(f"R1_hn vs A02 on test: {100 * (tot_1 / tot_a - 1):+.2f}%  (validation R1 vs A02: see runs; R3 +26.6%, R4 +25.2%)")
    # raw examples: US rows accepted by R4 but not A02
    r4 = pl.read_parquet(Path(__file__).resolve().parents[1] / "cache/model_R4_hn_sibrobust.txt".replace("model_R4_hn_sibrobust.txt", "val_scored_R4.parquet"))
    del r4  # (validation scores only; R4 test scores recomputed below would need sib features - use R1_hn new acceptances instead)
    new = X.filter((pl.col("country") == "US") & (pl.col("p1hn") >= 0.7) & (pl.col("pa") < 0.7))
    lines.append(f"\nUS test rows newly accepted by R1_hn: {new.height:,}")
    s1r = pl.read_parquet(ROOT / "cache/test_source1.parquet", columns=["business_name", "business_address"]).with_row_index("s1_idx")
    ex = new.sample(n=min(20, new.height), seed=11).select("q_idx", "s1_idx", "pa", "p1hn", "hn_absd1", "hn_par1") \
        .join(cty, on="q_idx").join(s1r, on="s1_idx", suffix="_s1").sort("p1hn")
    for r in ex.iter_rows(named=True):
        lines.append(f"A02 p={r['pa']:.3f} -> R1 p={r['p1hn']:.3f}  d={r['hn_absd1']:.0f} same_parity={r['hn_par1']:.0f}\n"
                     f"  Q : {r['business_name']} | {r['business_address']}\n  S1: {r['business_name_s1']} | {r['business_address_s1']}")
    text = "\n".join(lines)
    (OUT / "shift_hn_and_examples.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
