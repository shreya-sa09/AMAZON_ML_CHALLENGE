"""Raw examples of the remaining address-present errors of the current best model (Exp 10 R1), by house-number group.
Read-only. Also prints the error counts per group so the next target is chosen by size."""
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "results"
A02 = ROOT / "approaches/02_STAGE2_CROSSFIT_RUNNERUP"
E10 = ROOT / "approaches/10_STAGE2_ADDRESS_SIGNALS/cache"
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_formatting("ASCII_MARKDOWN"); pl.Config.set_tbl_rows(40)


def main():
    v = pl.read_parquet(A02 / "cache/validate_val_rows_cf1_ru0.parquet", columns=["q_idx", "s1_idx", "y", "f_q_addr_missing", "f_nm_exact"]) \
        .join(pl.read_parquet(E10 / "val_scored_R1_hn.parquet"), on="q_idx") \
        .join(pl.read_parquet(E10 / "feat_hn_val.parquet", columns=["q_idx", "hn_absd1", "hn_par1"]), on="q_idx") \
        .filter(pl.col("f_q_addr_missing") == 0)
    v = v.with_columns(
        pl.when(pl.col("hn_absd1") == 0).then(pl.lit("d=0")).when(pl.col("hn_absd1").is_between(1, 2)).then(pl.lit("d=1-2"))
          .when(pl.col("hn_absd1") < 0).then(pl.lit("na")).otherwise(pl.lit("d>2")).alias("hn_grp"),
        pl.when(pl.col("y") & (pl.col("p_stage2") < 0.7)).then(pl.lit("rejected_correct"))
          .when(~pl.col("y") & (pl.col("p_stage2") >= 0.7)).then(pl.lit("accepted_wrong")).otherwise(pl.lit("ok")).alias("err"))
    lines = ["error counts (address present, fold-B band, R1):",
             str(v.filter(pl.col("err") != "ok").group_by("err", "hn_grp").len().sort("err", "hn_grp"))]
    rq = pl.concat([pl.read_parquet(ROOT / f"cache/train_{s}.parquet", columns=["business_name", "business_address"])
                    for s in ("source2", "source3")]).with_row_index("q_idx")
    rs = pl.read_parquet(ROOT / "cache/train_source1.parquet", columns=["business_name", "business_address", "country"]).with_row_index("s1_idx")
    for err in ("rejected_correct", "accepted_wrong"):
        for grp in ("d=0", "d=1-2", "d>2", "na"):
            d = v.filter((pl.col("err") == err) & (pl.col("hn_grp") == grp))
            if d.height == 0:
                continue
            ex = d.sample(n=min(8, d.height), seed=3).join(rq, on="q_idx").join(rs, on="s1_idx", suffix="_s1").sort("p_stage2")
            lines.append(f"\n===== {err} / {grp} (n={d.height:,}) =====")
            for r in ex.iter_rows(named=True):
                lines.append(f"p={r['p_stage2']:.3f} [{r['country']}] d={r['hn_absd1']:.0f}\n  Q : {r['business_name']} | {r['business_address']}\n"
                             f"  S1: {r['business_name_s1']} | {r['business_address_s1']}")
    text = "\n".join(lines)
    (OUT / "r1_error_examples.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
