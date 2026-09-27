"""Describe observed properties of a fixed 200-row sample of saved top-10 misses."""
import json
import re
from collections import Counter
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "results"


def first_num(value):
    m = re.search(r"\d+", value or "")
    return m.group(0) if m else ""


def main():
    sample = pl.read_csv(OUT / "top10_miss_sample.csv", schema_overrides={"q_idx": pl.UInt32, "true_s1": pl.UInt32})
    q = pl.concat([pl.read_parquet(ROOT / f"cache/train_{src}_norm.parquet",
                                   columns=["name_core", "addr_norm", "addr_nums"])
                   for src in ("source2", "source3")]).with_row_index("q_idx")
    s1 = pl.read_parquet(ROOT / "cache/train_source1_norm.parquet",
                         columns=["country", "name_core", "addr_norm", "addr_nums"]).with_row_index("true_s1")
    dup = (s1.group_by("country", "name_core").len()
           .rename({"name_core": "name_core_s1", "len": "same_core_s1_count"}))
    rows = (sample.join(q, on="q_idx")
            .join(s1, on="true_s1", suffix="_s1")
            .join(dup, on=["country", "name_core_s1"]).sort("q_idx"))
    detail = []
    flags = Counter()
    for r in rows.iter_rows(named=True):
        qname, sname = r["name_core"], r["name_core_s1"]
        qa, sa = r["addr_norm"], r["addr_norm_s1"]
        qn, sn = first_num(r["addr_nums"]), first_num(r["addr_nums_s1"])
        name_ratio = fuzz.ratio(qname, sname)
        name_tset = fuzz.token_set_ratio(qname, sname)
        addr_ratio = fuzz.token_set_ratio(qa, sa) if qa and sa else None
        numeric_name_diff = set(re.findall(r"\d+", qname)) != set(re.findall(r"\d+", sname))
        obs = {"missing_query_address": r["addr_missing"],
               "duplicate_s1_core_name": r["same_core_s1_count"] > 1,
               "common_s1_core_name_ge10": r["same_core_s1_count"] >= 10,
               "indic_query": r["has_indic"],
               "low_name_similarity_lt60": name_ratio < 60,
               "low_name_token_set_lt60": name_tset < 60,
               "house_number_conflict": bool(qn and sn and qn != sn),
               "low_address_similarity_lt60": bool(addr_ratio is not None and addr_ratio < 60),
               "numeric_name_token_difference": numeric_name_diff}
        flags.update(k for k, v in obs.items() if v)
        detail.append({"q_idx": r["q_idx"], "true_s1": r["true_s1"], "country": r["country"],
                       "query_name_core": qname, "s1_name_core": sname,
                       "query_addr_norm": qa, "s1_addr_norm": sa,
                       "same_core_s1_count": r["same_core_s1_count"],
                       "name_ratio": round(name_ratio, 1), "name_token_set": round(name_tset, 1),
                       "addr_token_set": round(addr_ratio, 1) if addr_ratio is not None else None, **obs})
    pl.DataFrame(detail).write_csv(OUT / "top10_miss_sample_details.csv")
    summary = {"sample_size": rows.height, "method": "fixed seed sample; flags may overlap and are observations, not causal labels",
               "flag_counts": dict(flags)}
    (OUT / "failure_sample_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
