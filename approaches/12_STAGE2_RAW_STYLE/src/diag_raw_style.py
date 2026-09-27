"""Pre-check for Experiment 12: do RAW formatting signals (lost by normalization) separate correct vs wrong
in-band address-present Top-1 rows, overall and inside the ambiguous house-number groups (offset 0 / 1-2)?
Fold-A Stage-2 TRAINING rows (h0) only; read-only."""
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "results"
A02 = ROOT / "approaches/02_STAGE2_CROSSFIT_RUNNERUP"
E10 = ROOT / "approaches/10_STAGE2_ADDRESS_SIGNALS/cache"
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_formatting("ASCII_MARKDOWN"); pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(250)


def flags(prefix, name, addr):
    return [
        (pl.col(addr) == pl.col(addr).str.to_uppercase()).alias(f"{prefix}_addr_upper"),
        (pl.col(name) == pl.col(name).str.to_uppercase()).alias(f"{prefix}_name_upper"),
        pl.col(addr).str.contains(r"^\s*[#]?\d").alias(f"{prefix}_starts_num"),
        pl.col(addr).str.count_matches(",").alias(f"{prefix}_commas"),
        pl.col(addr).str.contains(r"(?i)\b(unit|apt|ste|suite|fl|floor|pmb|#)\b|#").alias(f"{prefix}_unit"),
        pl.col(addr).str.contains(r"(?i)\b(street|avenue|road|drive|lane|boulevard|court|place)\b").alias(f"{prefix}_long_sfx"),
        pl.col(name).str.contains(r"(?i)\b(llc|inc|ltd|limited|corp|co|pvt|private|lp|llp|pc)\b\.?").alias(f"{prefix}_legal"),
    ]


def main():
    t = pl.read_parquet(A02 / "cache/validate_train_rows_h0_cf1_ru0.parquet", columns=["q_idx", "s1_idx", "y", "f_q_addr_missing"]) \
        .filter(pl.col("f_q_addr_missing") == 0).join(pl.read_parquet(E10 / "feat_hn_train.parquet", columns=["q_idx", "hn_absd1"]), on="q_idx")
    rq = pl.concat([pl.read_parquet(ROOT / f"cache/train_{s}.parquet", columns=["business_name", "business_address"])
                    for s in ("source2", "source3")]).with_row_index("q_idx")
    rs = pl.read_parquet(ROOT / "cache/train_source1.parquet", columns=["business_name", "business_address", "country"]).with_row_index("s1_idx")
    d = t.join(rq, on="q_idx").join(rs, on="s1_idx", suffix="_s").with_columns(
        pl.col("business_name").fill_null(""), pl.col("business_address").fill_null(""),
        pl.col("business_name_s").fill_null(""), pl.col("business_address_s").fill_null(""))
    d = d.with_columns(flags("q", "business_name", "business_address") + flags("s", "business_name_s", "business_address_s"))
    d = d.with_columns((pl.col("q_starts_num") == pl.col("s_starts_num")).alias("same_order"),
                       (pl.col("q_commas").cast(pl.Int32) - pl.col("s_commas").cast(pl.Int32)).clip(-2, 2).alias("comma_diff"),
                       (pl.col("q_legal") == pl.col("s_legal")).alias("legal_same"),
                       (pl.col("q_unit") == pl.col("s_unit")).alias("unit_same"),
                       (pl.col("business_name") == pl.col("business_name_s")).alias("name_raw_eq"),
                       pl.when(pl.col("hn_absd1") == 0).then(pl.lit("d=0")).when(pl.col("hn_absd1").is_between(1, 2)).then(pl.lit("d=1-2"))
                         .when(pl.col("hn_absd1") < 0).then(pl.lit("na")).otherwise(pl.lit("d>2")).alias("hn_grp"))
    lines = [f"address-present in-band fold-A h0 rows: {d.height:,}; y-rate {d['y'].mean():.3f}"]
    for f in ["q_addr_upper", "q_name_upper", "same_order", "comma_diff", "q_unit", "s_unit", "unit_same", "q_long_sfx",
              "s_long_sfx", "legal_same", "name_raw_eq"]:
        g = d.group_by("country", "hn_grp", f).agg(pl.len().alias("n"), pl.col("y").mean().round(3).alias("y_rate")) \
             .filter(pl.col("n") >= 500).sort("country", "hn_grp", f)
        lines.append(f"\n## {f}\n{g}")
    text = "\n".join(lines)
    (OUT / "diag_raw_style.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
