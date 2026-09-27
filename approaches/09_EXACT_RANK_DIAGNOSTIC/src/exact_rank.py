"""Experiment 09a: exact (unsketched) rank of the true S1 for the fixed Experiment 07 Top-10 miss sample.

Read-only. Rebuilds the production CountryIndex TF-IDF views (same vectorizers) for US and India, then
brute-forces exact cosine of every sampled query against EVERY S1 of its country. For each miss:
  exact rank of the true S1 under the production score 0.5*cos_name + 0.5*cos_addr,
  number of S1 strictly beating it, number tied with it, number of exact-name twins of the query,
  and the same rank under name-only scoring.
Writes only to approaches/09_EXACT_RANK_DIAGNOSTIC/results/.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "business_entity_resolution/src"))
from retrieval import name_text  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

EPS = 1e-6
CHUNK = 64
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def vectorizers():
    # identical to retrieval.CountryIndex.__init__ (no sketch, no GPU index needed here)
    nv = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=2, max_df=0.05, sublinear_tf=True, dtype=np.float32)
    av = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=1, max_df=0.10, sublinear_tf=True, dtype=np.float32)
    return nv, av


def bucket(n):
    for lim, lab in ((0, "0"), (9, "1-9"), (39, "10-39"), (99, "40-99"), (999, "100-999")):
        if n <= lim:
            return lab
    return ">=1000"


def main():
    sample = pl.read_csv(ROOT / "approaches/07_CANDIDATE_RECALL_DECOMPOSITION/results/top40_sample_keys.csv",
                         schema_overrides={"q_idx": pl.UInt32, "true_s1": pl.UInt32})
    top40 = pl.read_csv(ROOT / "approaches/07_CANDIDATE_RECALL_DECOMPOSITION/results/top40_sample_diagnostic.csv",
                        schema_overrides={"q_idx": pl.UInt32}).select("q_idx", "in_top40")
    s1 = pl.read_parquet(ROOT / "cache/train_source1_norm.parquet",
                         columns=["country", "name_core", "addr_norm"]).with_row_index("s1_idx")
    q = pl.concat([pl.read_parquet(ROOT / f"cache/train_{src}_norm.parquet", columns=["name_core", "addr_norm"])
                   for src in ("source2", "source3")]).with_row_index("q_idx")
    q = q.join(sample.select("q_idx"), on="q_idx", how="semi")
    records = []
    for country in ("US", "India"):
        cs1 = s1.filter(pl.col("country") == country)
        cq = sample.filter(pl.col("country") == country).join(q, on="q_idx", how="left").sort("q_idx")
        log(f"{country}: fitting TF-IDF on {cs1.height:,} S1; {cq.height:,} queries")
        nv, av = vectorizers()
        Xn = nv.fit_transform(name_text(cs1)).tocsr()
        Xa = av.fit_transform(cs1["addr_norm"].to_list()).tocsr()
        Qn = nv.transform(name_text(cq)).tocsr()
        Qa = av.transform(cq["addr_norm"].to_list()).tocsr()
        local = dict(zip(cs1["s1_idx"].to_list(), range(cs1.height)))
        true_local = np.array([local[x] for x in cq["true_s1"].to_list()], dtype=np.int64)
        s1_core = cs1["name_core"].to_numpy()
        core_counts = cs1.group_by("name_core").len()
        core_count = dict(zip(core_counts["name_core"].to_list(), core_counts["len"].to_list()))
        XnT, XaT = Xn.T.tocsr(), Xa.T.tocsr()
        log(f"{country}: vocab name {Xn.shape[1]:,}, addr {Xa.shape[1]:,}; exact brute force")
        for s in range(0, cq.height, CHUNK):
            e = min(s + CHUNK, cq.height)
            sn = (Qn[s:e] @ XnT).toarray()  # (chunk, n_s1)
            sa = (Qa[s:e] @ XaT).toarray()
            comb = 0.5 * sn + 0.5 * sa
            for j in range(e - s):
                i = s + j
                t = true_local[i]
                tc, tn, ta = comb[j, t], sn[j, t], sa[j, t]
                n_gt = int((comb[j] > tc + EPS).sum())
                n_tie = int((np.abs(comb[j] - tc) <= EPS).sum()) - 1
                n_gt_name = int((sn[j] > tn + EPS).sum())
                n_name_twins = int((sn[j] >= 0.999).sum())  # S1 whose name vector equals the query's
                row = cq.row(i, named=True)
                records.append({
                    "q_idx": row["q_idx"], "true_s1": row["true_s1"], "country": country,
                    "addr_missing": row["addr_missing"], "has_indic": row["has_indic"],
                    "true_cos_name": float(tn), "true_cos_addr": float(ta), "true_cos_comb": float(tc),
                    "max_cos_comb": float(comb[j].max()), "max_cos_name": float(sn[j].max()),
                    "n_beat": n_gt, "n_tie": n_tie, "exact_rank_best": n_gt + 1, "exact_rank_worst": n_gt + n_tie + 1,
                    "n_beat_name_only": n_gt_name, "n_query_name_twins": n_name_twins,
                    "true_core_dups": int(core_count.get(s1_core[t], 1)),
                    "query_core_eq_true": row["name_core"] == s1_core[t],
                })
            if (s // CHUNK) % 5 == 0:
                log(f"{country}: {e:,}/{cq.height:,}")
        del Xn, Xa, XnT, XaT
    df = pl.DataFrame(records).join(top40, on="q_idx", how="left")
    df = df.with_columns(
        pl.when(pl.col("n_beat") + pl.col("n_tie") < 10).then(pl.lit("A_exact_top10_clear"))
          .when(pl.col("n_beat") < 10).then(pl.lit("B_exact_top10_only_via_ties"))
          .when(pl.col("n_tie") >= 1).then(pl.lit("C_outscored_and_tied"))
          .otherwise(pl.lit("D_outscored")).alias("category"),
        pl.col("n_beat").map_elements(bucket, return_dtype=pl.Utf8).alias("beat_bucket"))
    df.write_csv(OUT / "exact_rank_sample.csv")

    def summarize(d):
        return {
            "n": d.height,
            "category": dict(d.group_by("category").len().sort("category").iter_rows()),
            "n_beat_bucket": dict(d.group_by("beat_bucket").len().iter_rows()),
            "exact_rank1_clear(n_beat=0,n_tie=0)": d.filter((pl.col("n_beat") == 0) & (pl.col("n_tie") == 0)).height,
            "tied_at_top(n_beat=0,n_tie>=1)": d.filter((pl.col("n_beat") == 0) & (pl.col("n_tie") >= 1)).height,
            "query_core_eq_true": int(d["query_core_eq_true"].sum()),
            "true_core_dups_ge2": d.filter(pl.col("true_core_dups") >= 2).height,
            "true_core_dups_ge10": d.filter(pl.col("true_core_dups") >= 10).height,
            "query_name_twins_ge10": d.filter(pl.col("n_query_name_twins") >= 10).height,
            "median_n_beat": float(d["n_beat"].median()),
            "median_true_cos_name": float(d["true_cos_name"].median()),
            "median_true_cos_addr": float(d["true_cos_addr"].median()),
            "name_only_rank<=10": d.filter(pl.col("n_beat_name_only") < 10).height,
            "in_top40_sketch": int(d["in_top40"].sum()),
        }
    out = {"sample": "Experiment 07 fixed 2,000 saved Top-10 misses (1,000 US, 1,000 India)",
           "score": "exact 0.5*cos_name + 0.5*cos_addr against every S1 of the country (production TF-IDF views)",
           "categories": {"A": "exact rank <=10 even counting all ties -> lost only to CountSketch approximation",
                          "B": "within top-10 only if ties break favourably",
                          "C": ">=10 S1 strictly beat it and it is also tied",
                          "D": ">=10 S1 strictly beat it (true score genuinely lower)"},
           "all": summarize(df)}
    for key, filt in (("addr_missing", pl.col("addr_missing")), ("addr_present", ~pl.col("addr_missing")),
                      ("US", pl.col("country") == "US"), ("India", pl.col("country") == "India")):
        out[key] = summarize(df.filter(filt))
    out["runtime_sec"] = round(time.time() - T0)
    (OUT / "exact_rank_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
