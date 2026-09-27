"""Bounded K=80/20 diagnostic on Experiment 07's fixed known misses with missing addresses."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
E07 = ROOT / "approaches/07_CANDIDATE_RECALL_DECOMPOSITION/results"
sys.path.insert(0, str(ROOT / "business_entity_resolution/src"))
from retrieval import CountryIndex  # noqa: E402


def main():
    start = time.time()
    sample = pl.read_csv(E07 / "top40_sample_keys.csv", schema_overrides={"q_idx": pl.UInt32, "true_s1": pl.UInt32})
    sample = sample.filter(pl.col("addr_missing"))
    prior = pl.read_csv(E07 / "top40_sample_diagnostic.csv", schema_overrides={"q_idx": pl.UInt32})
    sample = sample.join(prior.select("q_idx", pl.col("in_top40").alias("in_old_top40")), on="q_idx")
    s1 = pl.read_parquet(ROOT / "cache/train_source1_norm.parquet",
                         columns=["entity_id", "country", "name_core", "addr_norm"]).with_row_index("s1_idx")
    q = pl.concat([pl.read_parquet(ROOT / f"cache/train_{src}_norm.parquet",
                                   columns=["country", "name_core", "addr_norm"])
                   for src in ("source2", "source3")]).with_row_index("q_idx")
    records = []
    new_pairs = []
    for country in ("US", "India"):
        sc = s1.filter(pl.col("country") == country)
        qc = sample.filter(pl.col("country") == country).join(q.select("q_idx", "name_core", "addr_norm"), on="q_idx").sort("q_idx")
        print(f"{country}: {sc.height:,} indexed S1; {qc.height:,} known missing-address Top-10 misses", flush=True)
        index = CountryIndex(sc)
        qn, qa = index.encode(qc)
        nb = index.approx_topk(qn, qa, K=80)
        local = dict(zip(sc["s1_idx"].to_list(), range(sc.height)))
        true_local = np.array([local[x] for x in qc["true_s1"].to_list()], dtype=np.int32)
        qi = np.repeat(np.arange(qc.height, dtype=np.int32), 80)
        di = nb.ravel()
        cn, ca = index.exact_scores(qn, qa, qi, di)
        scored = pl.DataFrame({"q_row": qi, "s1_row": di, "cos_name": cn, "cos_addr": ca})
        scored = scored.with_columns((0.5 * pl.col("cos_name") + 0.5 * pl.col("cos_addr")).alias("cos_comb"))
        scored = scored.with_columns(pl.col("cos_comb").rank("ordinal", descending=True).over("q_row").alias("rank"))
        true_rows = pl.DataFrame({"q_row": np.arange(qc.height, dtype=np.int32), "true_s1_row": true_local})
        found = scored.join(true_rows, on="q_row").filter(pl.col("s1_row") == pl.col("true_s1_row"))
        keep = scored.filter(pl.col("rank") <= 20)
        qmap = qc["q_idx"].to_numpy()
        smap = sc["s1_idx"].to_numpy()
        new_pairs.append(pl.DataFrame({"q_idx": qmap[keep["q_row"].to_numpy()],
                                       "s1_idx": smap[keep["s1_row"].to_numpy()]}))
        ranks = np.full(qc.height, 999, dtype=np.int32)
        ranks[found["q_row"].to_numpy()] = found["rank"].to_numpy()
        for row, rank in zip(qc.iter_rows(named=True), ranks):
            records.append({"q_idx": row["q_idx"], "country": country, "true_s1": row["true_s1"],
                            "in_old_top40": row["in_old_top40"], "rank_k80": int(rank) if rank != 999 else None,
                            "new_top20": bool(rank <= 20)})
        del index, qn, qa, nb, scored
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    data = pl.DataFrame(records)
    data.write_csv(OUT / "smoke_k80_pairs.csv")
    new_pairs = pl.concat(new_pairs).unique()
    old = []
    keys = sample.select("q_idx")
    for path in sorted((ROOT / "cache/train_cands").glob("*.parquet")):
        old.append(pl.read_parquet(path, columns=["q_idx", "s1_idx"]).join(keys, on="q_idx", how="semi"))
    old = pl.concat(old).unique()
    additions = new_pairs.join(old, on=["q_idx", "s1_idx"], how="anti")
    true_keys = sample.select("q_idx", pl.col("true_s1").alias("s1_idx"))
    added_true = additions.join(true_keys, on=["q_idx", "s1_idx"], how="inner").height
    assert added_true == int(data["new_top20"].sum())
    n = data.height
    def count(expr):
        return data.filter(expr).height
    summary = {"variant": "missing-address only; approximate Top-80, exact Top-20",
               "sample": "Experiment 07 fixed known Top-10 misses, filtered to missing-address queries",
               "sample_size": n, "US": count(pl.col("country") == "US"), "India": count(pl.col("country") == "India"),
               "old_top40": count(pl.col("in_old_top40")), "in_new_top80": count(pl.col("rank_k80").is_not_null()),
               "recovered_top20": count(pl.col("new_top20")),
               "rank_1_10": count(pl.col("rank_k80").is_between(1, 10)),
               "rank_11_20": count(pl.col("rank_k80").is_between(11, 20)),
               "rank_21_30": count(pl.col("rank_k80").is_between(21, 30)),
               "rank_31_80": count(pl.col("rank_k80").is_between(31, 80)),
               "absent_top80": count(pl.col("rank_k80").is_null()),
               "saved_candidate_pairs": old.height, "new_top20_pairs": new_pairs.height,
               "new_pairs_not_in_saved_top10": additions.height,
               "union_candidate_pairs": old.height + additions.height,
               "added_true_pairs": added_true,
               "added_negative_pairs_if_union_scored": additions.height - added_true,
               "runtime_sec": round(time.time() - start),
               "device": "cuda" if torch.cuda.is_available() else "cpu"}
    (OUT / "smoke_k80_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
