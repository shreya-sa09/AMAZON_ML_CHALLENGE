"""Targeted diagnostic only: rerun the existing approximate top-40 on fixed top-10 misses."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(ROOT / "business_entity_resolution/src"))
from retrieval import CountryIndex  # noqa: E402


def main():
    t0 = time.time()
    sample = pl.read_csv(OUT / "top40_sample_keys.csv", schema_overrides={"q_idx": pl.UInt32, "true_s1": pl.UInt32})
    s1 = pl.read_parquet(ROOT / "cache/train_source1_norm.parquet",
                         columns=["entity_id", "country", "name_core", "addr_norm"]).with_row_index("s1_idx")
    q = pl.concat([pl.read_parquet(ROOT / f"cache/train_{src}_norm.parquet",
                                   columns=["country", "name_core", "addr_norm"])
                   for src in ("source2", "source3")]).with_row_index("q_idx")
    records = []
    for country in ("US", "India"):
        country_s1 = s1.filter(pl.col("country") == country)
        country_q = sample.filter(pl.col("country") == country).join(q, on="q_idx", how="left").sort("q_idx")
        print(f"{country}: indexing {country_s1.height:,} S1, querying {country_q.height:,} known misses", flush=True)
        idx = CountryIndex(country_s1)
        qn, qa = idx.encode(country_q)
        nb = idx.approx_topk(qn, qa, K=40)
        local = dict(zip(country_s1["s1_idx"].to_list(), range(country_s1.height)))
        true_local = np.array([local.get(x, -1) for x in country_q["true_s1"].to_list()], dtype=np.int32)
        in_40 = (nb == true_local[:, None]).any(axis=1)
        qi = np.repeat(np.arange(country_q.height, dtype=np.int32), 40)
        di = nb.ravel()
        cn, ca = idx.exact_scores(qn, qa, qi, di)
        # Use the same Polars ordinal ranking expression as retrieval.CountryIndex.candidates.
        scored = pl.DataFrame({"q_row": qi, "s1_row": di, "cos_name": cn, "cos_addr": ca})
        scored = scored.with_columns((0.5 * pl.col("cos_name") + 0.5 * pl.col("cos_addr")).alias("cos_comb"))
        scored = scored.with_columns(pl.col("cos_comb").rank("ordinal", descending=True).over("q_row").alias("rank"))
        true_rows = pl.DataFrame({"q_row": np.arange(country_q.height, dtype=np.int32), "true_s1_row": true_local})
        found_rows = scored.join(true_rows, on="q_row").filter(pl.col("s1_row") == pl.col("true_s1_row"))
        ranks = np.full(country_q.height, 999, dtype=np.int32)
        ranks[found_rows["q_row"].to_numpy()] = found_rows["rank"].to_numpy()
        for row, found, rank in zip(country_q.iter_rows(named=True), in_40, ranks):
            records.append({"q_idx": row["q_idx"], "true_s1": row["true_s1"], "country": country,
                            "in_top40": bool(found), "exact_rank_within_top40": int(rank) if found else None,
                            "in_rerun_top10": bool(found and rank <= 10)})
        del idx, qn, qa, nb, scored
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    frame = pl.DataFrame(records)
    frame.write_csv(OUT / "top40_sample_diagnostic.csv")
    out = {"sample_size": frame.height, "sampling": "1000 fixed-seed saved top-10 misses per country",
           "device": "cuda" if torch.cuda.is_available() else "cpu",
           "top40_recovered": int(frame["in_top40"].sum()),
           "top10_recovered_on_rerun": int(frame["in_rerun_top10"].sum()),
           "top40_to_top10_loss": int((frame["in_top40"] & ~frame["in_rerun_top10"]).sum()),
           "by_country": {}, "runtime_sec": round(time.time() - t0)}
    for c in ("US", "India"):
        d = frame.filter(pl.col("country") == c)
        out["by_country"][c] = {"n": d.height, "top40_recovered": int(d["in_top40"].sum()),
                                "top10_recovered_on_rerun": int(d["in_rerun_top10"].sum())}
    (OUT / "top40_sample_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
