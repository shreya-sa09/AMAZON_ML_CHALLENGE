"""Experiment 10 pre-check: does each candidate new signal separate correct vs wrong in-band Top-1 rows?
Uses fold-A Stage-2 TRAINING rows only (no validation leakage into feature selection). Read-only."""
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "results"
A02 = ROOT / "approaches/02_STAGE2_CROSSFIT_RUNNERUP"
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_formatting("ASCII_MARKDOWN"); pl.Config.set_tbl_rows(60); pl.Config.set_tbl_width_chars(250)
lines = []


def show(title, df):
    lines.append(f"\n## {title}\n{df}")


def main():
    t = pl.read_parquet(A02 / "cache/validate_train_rows_h0_cf1_ru0.parquet",
                        columns=["q_idx", "s1_idx", "y", "p1", "f_q_addr_missing", "f_q_indic", "e_hn_edit", "e_hn_rel"])
    qn = pl.concat([pl.read_parquet(ROOT / f"cache/train_{s}_norm.parquet", columns=["addr_norm", "addr_nums"])
                    for s in ("source2", "source3")]).with_row_index("q_idx")
    sn = pl.read_parquet(ROOT / "cache/train_source1_norm.parquet", columns=["addr_norm", "addr_nums", "country"]).with_row_index("s1_idx")
    d = t.join(qn, on="q_idx").join(sn, on="s1_idx", suffix="_s")
    ap = d.filter(pl.col("f_q_addr_missing") == 0)
    lines.append(f"fold-A h0 in-band rows: {t.height:,}; address present: {ap.height:,}; y-rate {ap['y'].mean():.4f}")
    # (a) house-number parity / absolute difference among rows whose first numbers differ
    h = ap.with_columns(pl.col("addr_nums").str.extract(r"^(\d+)").cast(pl.Int64, strict=False).alias("a"),
                        pl.col("addr_nums_s").str.extract(r"^(\d+)").cast(pl.Int64, strict=False).alias("b")) \
          .filter(pl.col("a").is_not_null() & pl.col("b").is_not_null() & (pl.col("a") != pl.col("b")))
    h = h.with_columns((pl.col("a") - pl.col("b")).abs().alias("absd"), ((pl.col("a") - pl.col("b")) % 2 == 0).alias("same_parity"))
    h = h.with_columns(pl.col("absd").cut([1, 2, 4, 10, 20, 50, 100, 1000], left_closed=False).alias("absd_bin"))
    show("first house number differs: y-rate by |a-b| and parity (US)", h.filter(pl.col("country") == "US")
         .group_by("absd_bin", "same_parity").agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("y_rate")).sort("absd_bin", "same_parity"))
    show("first house number differs: y-rate by |a-b| and parity (India)", h.filter(pl.col("country") == "India")
         .group_by("absd_bin", "same_parity").agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("y_rate")).sort("absd_bin", "same_parity"))
    # does parity add information beyond the existing edit code? y-rate by (edit code, parity)
    show("y-rate by existing e_hn_edit x parity (US)", h.filter(pl.col("country") == "US")
         .group_by("e_hn_edit", "same_parity").agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("y_rate")).sort("e_hn_edit", "same_parity"))
    # (b) query-only / S1-only address tokens (non-numeric): frequency and y-rate
    tok = ap.select("y", "country", pl.col("addr_norm").str.split(" ").alias("qa"), pl.col("addr_norm_s").str.split(" ").alias("sa"))
    tok = tok.with_columns(pl.col("qa").list.set_difference("sa").alias("qo"), pl.col("sa").list.set_difference("qa").alias("so"))
    for side in ("qo", "so"):
        g = tok.select("y", side).explode(side).drop_nulls().filter(~pl.col(side).str.contains(r"^\d+$") & (pl.col(side) != "")) \
               .group_by(side).agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("y_rate")).sort("n", descending=True)
        show(f"most frequent non-numeric {side} address tokens (y-rate; base {ap['y'].mean():.3f})", g.head(40))
        lines.append(f"{side}: {g.height:,} distinct tokens; tokens with n>=200: {g.filter(pl.col('n') >= 200).height:,}; "
                     f"rows covered by a token with n>=200 and |y_rate-base|>0.15: "
                     f"{g.filter((pl.col('n') >= 200) & ((pl.col('y_rate') - ap['y'].mean()).abs() > 0.15))['n'].sum():,}")
    # rows with no non-numeric address difference at all
    tok = tok.with_columns(pl.col("qo").list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+$"))).list.len().alias("n_qo_w"),
                           pl.col("so").list.eval(pl.element().filter(~pl.element().str.contains(r"^\d+$"))).list.len().alias("n_so_w"))
    show("y-rate by #non-numeric query-only / S1-only address tokens", tok.with_columns(pl.col("n_qo_w").clip(0, 3), pl.col("n_so_w").clip(0, 3))
         .group_by("n_qo_w", "n_so_w").agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("y_rate")).sort("n_qo_w", "n_so_w"))
    # (c) Indic queries in band
    show("in-band y-rate by q_indic (address present)", ap.group_by("country", "f_q_indic").agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("y_rate")).sort("country", "f_q_indic"))
    text = "\n".join(lines)
    (OUT / "diag_signals.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
