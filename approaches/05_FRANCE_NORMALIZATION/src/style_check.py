"""
Label-free evaluation of the France fix (validation has no France records, so F0.5 cannot be measured).

1. Address-style exchangeability test. Whether a France S2/S3 record writes the departement, the region,
   or neither is a formatting choice of the noise generator and should be independent of whether the
   record is a true match. So, among France records, the acceptance rate (and mean probability) of the
   "departement" group should equal that of the "region" group. Before the fix the departement group
   is penalised by the address mismatch; if the fix is right, the gap should close.
   Balance checks confirm the groups are otherwise comparable (source, name length, house number).
2. US / India identity: every non-France row of this approach's matching_results.tsv must equal the
   Approach 02 @0.70 file of Experiment 1 (04_THRESHOLD_DENSITY_AUDIT), which is read-only here.

usage (from the approach folder): python src/style_check.py [--smoke]
"""
import json
import sys

import paths  # noqa: F401
from paths import APPROACH_DIR, CONFIG, OUTPUT_DIR, RESULTS_DIR, mine, shared

import polars as pl

from normalization import ascii_fold  # baseline (read-only)

C = CONFIG
SMOKE = "--smoke" in sys.argv
T = C["threshold"]
DEP = set(C["france"]["departement_to_region"])
REG = {"hautsdefrance", "nouvelleaquitaine", "paysdelaloire"}


def key(c):
    return "".join(ch for ch in ascii_fold(c).lower() if ch.isalnum())


def style(addr):
    if not addr:
        return "none"
    ks = {key(p) for p in addr.split(",")}
    if ks & DEP:
        return "departement"
    if ks & REG:
        return "region"
    return "none"


tag = (lambda n: f"smoke_{n}") if SMOKE else (lambda n: n)
after = pl.read_parquet(mine(tag("france_final_scored") + ".parquet")).select("q_idx", "s1_idx", pl.col("p_final").alias("p_after"),
                                                                            pl.col("p1").alias("p1_after"))
before = pl.read_parquet(APPROACH_DIR / C["france"]["stage2_scored_other_countries"]).select(
    "q_idx", pl.col("s1_idx").alias("s1_before"), pl.col("p_final").alias("p_before"), pl.col("p1").alias("p1_before"))
raw = pl.concat([pl.read_parquet(shared(f"test_{s}.parquet"), columns=["business_address", "business_name"]).with_columns(pl.lit(s == "source3").alias("is_s3"))
                 for s in ("source2", "source3")]).with_row_index("q_idx")
norm = pl.concat([pl.read_parquet(shared(f"test_{s}_norm.parquet"), columns=["addr_nums", "name_core"]) for s in ("source2", "source3")]).with_row_index("q_idx")
d = after.join(before, on="q_idx").join(raw, on="q_idx").join(norm, on="q_idx")
d = d.with_columns(pl.col("business_address").map_elements(style, return_dtype=pl.Utf8).alias("style"))

agg = d.group_by("style").agg(
    pl.len().alias("records"),
    (pl.col("p_before") >= T).mean().alias("accept_before"), (pl.col("p_after") >= T).mean().alias("accept_after"),
    pl.col("p_before").mean().alias("mean_p_before"), pl.col("p_after").mean().alias("mean_p_after"),
    (pl.col("p1_before") >= T).mean().alias("stage1_accept_before"), (pl.col("p1_after") >= T).mean().alias("stage1_accept_after"),
    ((pl.col("s1_idx") != pl.col("s1_before"))).mean().alias("top1_changed"),
    # balance checks (should not depend on style)
    pl.col("is_s3").mean().alias("share_s3"), pl.col("name_core").str.len_chars().mean().alias("name_len"),
    (pl.col("addr_nums") != "").mean().alias("has_number"),
).sort("style")
pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(20)
print(agg)
g = {r["style"]: r for r in agg.to_dicts()}
res = {"threshold": T, "groups": g}
if "departement" in g and "region" in g:
    for k in ("accept", "mean_p", "stage1_accept"):
        b = g["departement"][f"{k}_before"] - g["region"][f"{k}_before"]
        a = g["departement"][f"{k}_after"] - g["region"][f"{k}_after"]
        res[f"gap_{k}_dep_minus_reg"] = {"before": b, "after": a}
        print(f"{k:14s} departement - region gap: before {b:+.4f}  after {a:+.4f}")
res["france_accepted_before"] = int((d["p_before"] >= T).sum())
res["france_accepted_after"] = int((d["p_after"] >= T).sum())
res["france_records"] = d.height

# US / India identity against Experiment 1's 02 @0.70 file
ref = APPROACH_DIR.parent / "04_THRESHOLD_DENSITY_AUDIT" / "output" / "matching_results_t0.70.tsv"
mine_out = (RESULTS_DIR / "smoke" / "matching_results.tsv") if SMOKE else (OUTPUT_DIR / "matching_results.tsv")
if not SMOKE and ref.exists() and mine_out.exists():
    ts1 = pl.read_parquet(shared("test_source1.parquet"), columns=["entity_id", "country"])
    rd = lambda p: pl.read_csv(p, separator="\t", quote_char=None, schema={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8})
    a, b = rd(mine_out), rd(ref)
    j = a.join(b, on="source1_entity_id", suffix="_ref").join(ts1, left_on="source1_entity_id", right_on="entity_id")
    j = j.with_columns((pl.col("matched_entity_ids").fill_null("") == pl.col("matched_entity_ids_ref").fill_null("")).alias("same"))
    ident = j.group_by("country").agg(pl.len().alias("s1_rows"), pl.col("same").mean().alias("identical_share"))
    print(ident.sort("country"))
    res["identity_vs_02_t070"] = {c: {"rows": n, "identical_share": s} for c, n, s in ident.rows()}
json.dump(res, open(RESULTS_DIR / tag("style_check.json"), "w"), indent=2, default=float)
