"""
03_FRANCE_NORMALIZATION — France-only normalization fixes on top of 02_STAGE2_CROSSFIT_RUNNERUP.

Why (measured on the test data, which is the only France data):
  * 32% of France S2/S3 records name the *departement* (Nord, Pas-de-Calais, Gironde, Loire-Atlantique)
    where every France S1 record names the *region* (Hauts-de-France, Nouvelle-Aquitaine, Pays de la
    Loire). US / India never had this (state names are normalized), so both models read it as an
    address mismatch. Stage 2 rejected 3.2% of the France records stage 1 had accepted (US 0.7%).
  * Short digit-for-letter legal forms are not repaired by the baseline typo rule (it needs >= 4
    letters): 5arl, 5as, c1ub.

What this step does (France records only; US / India decisions are copied unchanged from 02):
  1. address components that are a departement are rewritten to their region, then re-normalized
     with the baseline normalize_address; digit-for-letter tokens in names are repaired when the
     repaired token is a legal word or a common France S1 name token.
  2. the 64 baseline pair features are recomputed for all France candidate pairs (baseline
     compute_features, unchanged code) with the corrected records.
  3. stage 1 is re-scored with the unchanged baseline model; the top-1 is re-derived.
  4. stage 2 (02's final model, unchanged, imported) re-scores the France top-1 pairs in the band.
  5. France decisions replace 02's France decisions; outputs are written to this approach's output/.

Not changed: retrieval / candidates (region and departement tokens are above the TF-IDF max_df and
never were in the retrieval vocabulary), any model, any training, US / India.

usage (from the approach folder):  python src/france_fix.py [--smoke]
"""
import json
import sys
import time

import paths  # noqa: F401  (sets ER_CACHE_DIR / ER_OUTPUT_DIR and sys.path first)
from paths import APPROACH_DIR, CONFIG, OUTPUT_DIR, RESULTS_DIR, mine, shared

import lightgbm as lgb
import numpy as np
import polars as pl

import stage2 as s2                                        # 02 code (copied into this folder)
from edit_features import Vocab
from features import FEATURES, compute_features, s1_extra_table   # baseline (read-only)
from normalization import LEGAL_WORDS, _LEET, ascii_fold, normalize_address  # baseline (read-only)
from predict import write_grouped                          # baseline (read-only)
from train import best_per_query, id_tables                # baseline (read-only)

C = CONFIG
F = C["france"]
SMOKE = "--smoke" in sys.argv
T0 = time.time()
QCOLS = ["name_core", "name_norm", "addr_norm", "addr_nums", "is_domain", "has_indic", "addr_missing"]


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def tag(name):
    return f"smoke_{name}" if SMOKE else name


def comp_key(c: str) -> str:
    return "".join(ch for ch in ascii_fold(c).lower() if ch.isalnum())


def fix_address(raw: str) -> str:
    parts = raw.split(",")
    out = [F["departement_to_region"].get(comp_key(p), p) for p in parts]
    return ",".join(out)


def fix_name_tokens(name_norm: str, vocab: dict):
    """repair digit-for-letter tokens the baseline rule skips; returns (name_norm, name_core, changed)"""
    toks, changed = [], False
    for t in name_norm.split():
        if not t.isalpha() and not t.isdigit():
            r = t.translate(_LEET)
            if r.isalpha() and (r in LEGAL_WORDS or vocab.get(r, 0) >= F["leet_min_s1_count"]):
                t, changed = r, True
        toks.append(t)
    core = [t for t in toks if t not in LEGAL_WORDS] or toks
    return " ".join(toks), " ".join(core), changed


# --------------------------------------------------------------------------- step 1: corrected records
def corrected_tables():
    path = mine(tag("test_q_norm_france_fixed") + ".parquet")
    q = pl.concat([pl.read_parquet(shared(f"test_{s}_norm.parquet"), columns=["country"] + QCOLS).with_columns(pl.lit(s == "source3").alias("is_s3"))
                   for s in ("source2", "source3")]).with_row_index("q_idx")
    s1 = pl.read_parquet(shared("test_source1_norm.parquet"), columns=["country", "name_core", "name_norm", "addr_norm", "addr_nums"]).with_row_index("s1_idx")
    if path.exists():
        log(f"reuse {path.name}")
        return pl.read_parquet(path), s1
    raw = pl.concat([pl.read_parquet(shared(f"test_{s}.parquet"), columns=["business_address"]) for s in ("source2", "source3")])["business_address"]
    fr = q["country"] == F["country"]
    idx = np.flatnonzero(fr.to_numpy())
    if SMOKE:
        idx = idx[:50_000]
    vocab = {}
    for n in s1.filter(pl.col("country") == F["country"])["name_core"].to_list():
        for t in n.split():
            vocab[t] = vocab.get(t, 0) + 1
    addr, nn, nc = q["addr_norm"].to_list(), q["name_norm"].to_list(), q["name_core"].to_list()
    raw_l = raw.to_list()
    n_addr = n_name = 0
    for i in idx:
        a = raw_l[i] or ""
        fa = fix_address(a)
        if fa != a:
            new = normalize_address(fa)[0]
            n_addr += new != addr[i]
            addr[i] = new
        x, y, ch = fix_name_tokens(nn[i], vocab)
        if ch:
            nn[i], nc[i] = x, y
            n_name += 1
    q = q.with_columns(pl.Series("addr_norm", addr), pl.Series("name_norm", nn), pl.Series("name_core", nc))
    q.write_parquet(path)
    stats = {"france_records": int(len(idx)), "addresses_changed": n_addr, "names_changed": n_name}
    json.dump(stats, open(RESULTS_DIR / tag("normalization_changes.json"), "w"), indent=2)
    log(f"France records {len(idx):,}: {n_addr:,} addresses and {n_name:,} names corrected")
    return q, s1


# --------------------------------------------------------------------------- step 2+3: pair features, stage-1 rescoring
def france_pair_features_and_stage1(q, s1):
    out_dir = mine(tag("france_feats"))
    out_dir.mkdir(exist_ok=True)
    stage1 = lgb.Booster(model_file=str(shared("lgbm_model.txt")))
    extra = s1_extra_table(s1.select("s1_idx", "country", "name_core"), shared("test_cands"))
    qsel = q.select(QCOLS + ["is_s3"])
    s1sel = s1.select("name_core", "name_norm", "addr_norm", "addr_nums")
    scored = []
    files = sorted(shared("test_cands").glob(f"{F['country']}_*.parquet"))
    for f in files[:1] if SMOKE else files:
        out = out_dir / f.name
        if not out.exists():
            c = pl.read_parquet(f).sort("q_idx", "rank")
            if SMOKE:
                c = c.head(200_000)
            qids = c["q_idx"].unique(maintain_order=True)
            parts = []
            for s in range(0, len(qids), 100_000):  # query-aligned chunks, as in build_features.py
                sub = c.filter(pl.col("q_idx").is_in(qids.slice(s, 100_000).implode()))
                parts.append(compute_features(sub, qsel, s1sel, extra))
            feats = pl.concat(parts)
            feats.write_parquet(out.with_suffix(".tmp"))
            out.with_suffix(".tmp").replace(out)
            log(f"{f.name}: {feats.height:,} France pairs re-featurized")
        d = pl.read_parquet(out)
        p = stage1.predict(d.select(FEATURES).to_numpy(), num_threads=16).astype(np.float32)
        scored.append(d.select("q_idx", "s1_idx").with_columns(pl.Series("p", p)))
    scored = pl.concat(scored)
    return scored, best_per_query(scored)


# --------------------------------------------------------------------------- step 4: stage 2
def france_stage2(best_fr, q, s1):
    # token odds exactly as 02's final step (learned on all eligible training records)
    _, _, _, best_tr = s2.train_rows()
    qn_tr, s1n_tr = s2.norm_tables("train")
    ref_size = s1n_tr.height
    te = s2.learn_te(best_tr, qn_tr, s1n_tr, pl.col("oos"))
    del qn_tr, s1n_tr, best_tr
    feats_dir = mine(tag("france_feats"))

    def pair_feats_fixed(split, keys):  # France pair features from step 2 instead of the shared cache
        keys = keys.select(pl.col("q_idx").cast(pl.UInt32), pl.col("s1_idx").cast(pl.UInt32)).unique()
        parts = [pl.read_parquet(f).join(keys, on=["q_idx", "s1_idx"], how="semi") for f in sorted(feats_dir.glob("*.parquet"))]
        return pl.concat(parts).rename({c: "f_" + c for c in s2.PAIR_FEATURES})

    s2.pair_feats = pair_feats_fixed
    qn = q.select("q_idx", "name_core", "addr_norm", "addr_nums")
    s1n = s1.select("s1_idx", "name_core", "addr_norm", "addr_nums")
    vocab = Vocab(s1n, ref_size)
    band = best_fr.filter(s2.in_band())
    X = s2.build("test", band.select("q_idx", "s1_idx", "p1", "p2"), qn, s1n, vocab, te, tag("france_test_rows"))
    model = lgb.Booster(model_file=str(APPROACH_DIR / F["stage2_model"]))
    pt = model.predict(X.select(s2.STAGE2_FEATURES).to_numpy().astype(np.float32), num_threads=16).astype(np.float32)
    return best_fr.join(X.select("q_idx", pl.Series("p_stage2", pt)), on="q_idx", how="left") \
                  .with_columns(pl.coalesce("p_stage2", "p1").alias("p_final"))


# --------------------------------------------------------------------------- step 5: merge + write
def main():
    log(f"03_FRANCE_NORMALIZATION {'(smoke)' if SMOKE else ''}")
    q, s1 = corrected_tables()
    scored_fr, best_fr = france_pair_features_and_stage1(q, s1)
    log(f"stage 1 re-scored {scored_fr.height:,} France pairs; {best_fr.height:,} France records")
    fr = france_stage2(best_fr, q, s1)
    fr.write_parquet(mine(tag("france_final_scored") + ".parquet"))
    t = C["threshold"]
    prev = pl.read_parquet(APPROACH_DIR / F["stage2_scored_other_countries"])
    is_fr = q.select("q_idx", (pl.col("country") == F["country"]).alias("fr"))
    prev = prev.join(is_fr, on="q_idx")
    old_fr = prev.filter("fr")
    merged = pl.concat([prev.filter(~pl.col("fr")).select("q_idx", "s1_idx", "p_final"),
                        fr.select("q_idx", "s1_idx", "p_final"),
                        old_fr.join(fr.select("q_idx"), on="q_idx", how="anti").select("q_idx", "s1_idx", "p_final")])  # smoke: untouched rows
    matches = merged.filter(pl.col("p_final") >= t).select("s1_idx", "q_idx")
    old_fr_acc = old_fr.join(fr.select("q_idx"), on="q_idx", how="semi").filter(pl.col("p_final") >= t)  # same records only
    new_fr_acc = fr.filter(pl.col("p_final") >= t)
    comp = old_fr_acc.select("q_idx", pl.col("s1_idx").alias("s_old")).join(new_fr_acc.select("q_idx", pl.col("s1_idx").alias("s_new")), on="q_idx", how="full", coalesce=True)
    summary = {
        "threshold": t, "smoke": SMOKE,
        "france_records_rescored": fr.height,
        "france_accepted_02": old_fr_acc.height, "france_accepted_03": new_fr_acc.height,
        "france_newly_accepted": int(comp.filter(pl.col("s_old").is_null()).height),
        "france_no_longer_accepted": int(comp.filter(pl.col("s_new").is_null()).height),
        "france_changed_s1": int(comp.filter(pl.col("s_old").is_not_null() & pl.col("s_new").is_not_null() & (pl.col("s_old") != pl.col("s_new"))).height),
        "france_stage1_acc_rate": float((fr["p1"] >= t).mean()), "france_final_acc_rate": float((fr["p_final"] >= t).mean()),
        "france_stage2_rejects_stage1_accept": float(((fr["p1"] >= t) & (fr["p_final"] < t)).mean()),
        "total_matched_records": matches.height,
    }
    log(json.dumps(summary, indent=1))
    s1_t, q_t = id_tables("test")
    out_dir = RESULTS_DIR / "smoke" if SMOKE else OUTPUT_DIR
    out_dir.mkdir(exist_ok=True)
    if SMOKE:
        matches = matches.join(fr.select("q_idx"), on="q_idx", how="semi").head(2000)
        cand = pl.read_parquet(shared("test_scored.parquet"), columns=["s1_idx", "q_idx"]).join(matches.select("q_idx"), on="q_idx", how="semi")
    else:
        cand = pl.read_parquet(shared("test_scored.parquet"), columns=["s1_idx", "q_idx"])
    write_grouped(s1_t["entity_id"], q_t["entity_id"], cand, "candidate_entity_ids", out_dir / "candidate_pairs.tsv")
    del cand
    n, n_m = write_grouped(s1_t["entity_id"], q_t["entity_id"], matches, "matched_entity_ids", out_dir / "matching_results.tsv")
    summary.update(s1_rows=n, s1_with_match=n_m, runtime_sec=round(time.time() - T0), output_dir=str(out_dir))
    json.dump(summary, open(RESULTS_DIR / tag("final_summary.json"), "w"), indent=2)
    log(f"wrote {out_dir / 'matching_results.tsv'}: {matches.height:,} matched records over {n_m:,}/{n:,} S1")


if __name__ == "__main__":
    json.dump(C, open(RESULTS_DIR / tag("config_used.json"), "w"), indent=2)
    main()
