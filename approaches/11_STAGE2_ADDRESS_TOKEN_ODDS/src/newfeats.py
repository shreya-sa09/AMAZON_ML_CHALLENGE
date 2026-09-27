"""New Stage-2 feature groups. Each function: normalized pair rows (from common.norm_pairs) -> DataFrame, row-aligned.
Country-agnostic: only uses the normalized numeric tokens, so it applies unchanged to France."""
import re

import polars as pl

_NUM = re.compile(r"\d+")
MAXD = 10 ** 15

HN_COLS = ["hn_absd1", "hn_par1", "hn_absd2", "hn_par2", "hn_bestd", "hn_best_par"]


def _nums(s):
    return [int(x) for x in _NUM.findall(s or "") if len(x) <= 15]


def _pair(a, b):
    if a is None or b is None:
        return -1.0, -1.0
    d = abs(a - b)
    return float(d), float(d % 2 == 0)


def hn_features(df: pl.DataFrame) -> pl.DataFrame:
    """House-number absolute offset and parity (same side of street) for first / second numbers and for the
    query's first number against its closest S1 number. -1 = not available."""
    rows = []
    for qa, sa in zip(df["addr_nums"].to_list(), df["addr_nums_s"].to_list()):
        q, s = _nums(qa), _nums(sa)
        d1, p1 = _pair(q[0] if q else None, s[0] if s else None)
        d2, p2 = _pair(q[1] if len(q) > 1 else None, s[1] if len(s) > 1 else None)
        if q and s:
            b = min(s, key=lambda x: abs(x - q[0]))
            bd, bp = _pair(q[0], b)
        else:
            bd, bp = -1.0, -1.0
        rows.append((d1, p1, d2, p2, bd, bp))
    return pl.DataFrame(rows, schema=HN_COLS, orient="row").cast(pl.Float32)


GROUPS = {"hn": (HN_COLS, hn_features)}
