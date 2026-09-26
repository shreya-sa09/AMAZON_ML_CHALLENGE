"""
Generator-aware "edit-type" features for a (S2/S3 record, S1 record) pair.

Measured on training data: a *noisy true copy* differs from its S1 record by OCR-style typos
(rn->m, cl->d, I->l, digits for letters) and by house-number edits such as a changed middle digit or
a prepended digit, while a *decoy* (a different business) swaps in another real word, transposes or
appends house-number digits, or adds a number the S1 record does not have. These features describe
*how* the two records differ, which the baseline similarity features do not.
"""
import re
from collections import Counter

import numpy as np
import polars as pl
from rapidfuzz import fuzz

_CONF = [("rn", "m"), ("cl", "d"), ("vv", "w"), ("ii", "u")]
_CHR = str.maketrans({"0": "o", "1": "l", "i": "l", "5": "s", "8": "b", "3": "e", "4": "a", "7": "t",
                      "9": "g", "6": "g", "2": "z", "|": "l"})
_NUM = re.compile(r"\d+")

# house-number edit codes (categorical)
HN_BOTH_MISSING, HN_SAME, HN_SUB_LAST, HN_SUB_MID, HN_TRANSPOSE, HN_MULTI, HN_APPEND1, HN_APPEND_MANY, \
    HN_PREPEND, HN_INSERT_MID, HN_OTHER_LEN, HN_S1_MISSING, HN_QUERY_MISSING = range(13)

EDIT_COLS = ["e_qo", "e_so", "e_ocr_hits", "e_all_ocr", "e_leftover_min_ratio", "e_qo_minfreq", "e_qo_maxfreq",
             "e_so_minfreq", "e_so_maxfreq", "e_ao", "e_aso", "e_ao_minfreq", "e_hn_edit", "e_hn_pos", "e_hn_rel",
             "e_hq_len", "e_hs_len",
             "e_qo_nonword", "e_so_nonword", "e_pair_ratio", "e_pair_lendiff", "e_pair_ocr", "e_pair_first",
             "e_te_q_min", "e_te_q_max", "e_te_s_min", "e_addr_pair_ratio", "e_addr_ocr_eq", "e_hn2_edit", "e_name_ocr_eq",
             "e_nq", "e_ns", "e_n_qonly", "e_n_sonly", "e_q_num_s_none", "e_q_first_num_new"]


def skel(t: str) -> str:
    """Visual-confusion skeleton: tokens that differ only by OCR-style confusions map to the same string."""
    t = t.translate(_CHR)
    for a, b in _CONF:
        t = t.replace(a, b)
    return t


def hn_edit(a: str, c: str):
    """(edit code, position / length difference, relative numeric difference) between two house numbers."""
    if not a or not c:
        return (HN_S1_MISSING if a else HN_QUERY_MISSING if c else HN_BOTH_MISSING), -1, -1.0
    if a == c:
        return HN_SAME, -1, 0.0
    rel = abs(int(a) - int(c)) / max(int(a), int(c), 1)
    if len(a) == len(c):
        d = [i for i in range(len(a)) if a[i] != c[i]]
        if len(d) == 1:
            return (HN_SUB_LAST if d[0] == len(a) - 1 else HN_SUB_MID), d[0], rel
        if len(d) == 2 and a[d[0]] == c[d[1]] and a[d[1]] == c[d[0]]:
            return HN_TRANSPOSE, d[0], rel
        return HN_MULTI, len(d), rel
    lo, hi = (a, c) if len(a) < len(c) else (c, a)
    if hi.startswith(lo):
        return (HN_APPEND1 if len(hi) - len(lo) == 1 else HN_APPEND_MANY), len(hi) - len(lo), rel
    if hi.endswith(lo):
        return HN_PREPEND, len(hi) - len(lo), rel
    if len(hi) - len(lo) == 1 and any(hi[:i] + hi[i + 1:] == lo for i in range(len(hi))):
        return HN_INSERT_MID, 1, rel
    return HN_OTHER_LEN, len(hi) - len(lo), rel


class Vocab:
    """S1-side token frequencies of one split, rescaled to the training S1 size so that train and test
    features live on the same scale (the reference corpus sizes differ)."""

    def __init__(self, s1_norm: pl.DataFrame, ref_size: int):
        scale = ref_size / max(s1_norm.height, 1)
        self.name = Counter(t for n in s1_norm["name_core"].to_list() for t in n.split())
        self.addr = Counter(t for n in s1_norm["addr_norm"].to_list() for t in n.split())
        self.scale = scale

    def lf_name(self, t):
        return float(np.log1p(self.name.get(t, 0) * self.scale))

    def lf_addr(self, t):
        return float(np.log1p(self.addr.get(t, 0) * self.scale))


def token_te(rows: pl.DataFrame, smoothing: float):
    """Smoothed P(true match) per query-only and per S1-only name token, learned from labelled rows
    (columns name_core, name_core_s, y). Returns two dicts."""
    src = rows.with_columns(pl.col("name_core").str.split(" ").alias("qt"), pl.col("name_core_s").str.split(" ").alias("st"))
    src = src.with_columns(pl.col("qt").list.set_difference("st").alias("qo"), pl.col("st").list.set_difference("qt").alias("so"))
    prior = src["y"].mean()
    out = []
    for c in ("qo", "so"):
        g = src.select("y", c).explode(c).drop_nulls().group_by(c).agg(pl.len().alias("n"), pl.col("y").sum().alias("k"))
        g = g.with_columns(((pl.col("k") + smoothing * prior) / (pl.col("n") + smoothing)).alias("te"))
        out.append(dict(zip(g[c].to_list(), g["te"].to_list())))
    return out[0], out[1]


def _first_num(x):
    m = _NUM.match(x or "")
    return m.group(0) if m else ""


def edit_features(df: pl.DataFrame, vocab: Vocab, te_q: dict, te_s: dict) -> pl.DataFrame:
    """df needs name_core, name_core_s, addr_norm, addr_norm_s, addr_nums, addr_nums_s (row-aligned output)."""
    qn, sn = df["name_core"].to_list(), df["name_core_s"].to_list()
    qa_, sa_ = df["addr_norm"].to_list(), df["addr_norm_s"].to_list()
    hq = [_first_num(x) for x in df["addr_nums"].to_list()]
    hs = [_first_num(x) for x in df["addr_nums_s"].to_list()]
    rows = []
    for a, b, aa, ab, h1, h2 in zip(qn, sn, qa_, sa_, hq, hs):
        qt, st = set(a.split()), set(b.split())
        qo, so = qt - st, st - qt
        sko = {skel(t) for t in so}
        ocr_hits = sum(skel(t) in sko for t in qo)
        best_r = [max((fuzz.ratio(t, u) for u in so), default=0.0) for t in qo]
        qo_f = [vocab.lf_name(t) for t in qo]
        so_f = [vocab.lf_name(t) for t in so]
        aq, as_ = set(aa.split()), set(ab.split())
        ao, aso = aq - as_, as_ - aq
        ao_f = [vocab.lf_addr(t) for t in ao]
        e, pos, rel = hn_edit(h1, h2)
        pr = max(((fuzz.ratio(t, u), t, u) for t in qo for u in so), default=(0.0, "", ""))
        tq = [te_q.get(t, -1.0) for t in qo]
        ts = [te_s.get(t, -1.0) for t in so]
        a_pr = max((fuzz.ratio(t, u) for t in ao for u in aso), default=100.0)
        nq, ns = _NUM.findall(aa), _NUM.findall(ab)
        e2 = hn_edit(nq[1] if len(nq) > 1 else "", ns[1] if len(ns) > 1 else "")[0]
        first = aa.split()[0] if aa else ""
        rows.append((
            len(qo), len(so), ocr_hits, float(ocr_hits == len(qo) and len(qo) == len(so) and len(qo) > 0),
            min(best_r) if best_r else 100.0, min(qo_f) if qo_f else -1.0, max(qo_f) if qo_f else -1.0,
            min(so_f) if so_f else -1.0, max(so_f) if so_f else -1.0, len(ao), len(aso), min(ao_f) if ao_f else -1.0,
            e, pos, rel, len(h1), len(h2),
            sum(vocab.name.get(t, 0) == 0 for t in qo), sum(vocab.name.get(t, 0) == 0 for t in so),
            pr[0], abs(len(pr[1]) - len(pr[2])), float(skel(pr[1]) == skel(pr[2]) and pr[1] != ""), float(pr[1][:1] == pr[2][:1]),
            min(tq) if tq else -2.0, max(tq) if tq else -2.0, min(ts) if ts else -2.0,
            a_pr, float(skel(aa) == skel(ab)), e2, float(skel(a) == skel(b)),
            len(nq), len(ns), len(set(nq) - set(ns)), len(set(ns) - set(nq)), float(len(ns) == 0 and len(nq) > 0),
            float(first.isdigit() and first not in ns),
        ))
    return pl.DataFrame(rows, schema=EDIT_COLS, orient="row").cast(pl.Float32)
