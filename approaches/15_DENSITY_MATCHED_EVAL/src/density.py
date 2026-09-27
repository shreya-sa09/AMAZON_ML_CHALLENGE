"""Density-matched validation (leaderboard proxy). Read-only.

Test has ~1.9x the decoys (records with no true S1) per S1 of train (Exp 04). Simulation (Exp 04 method): every
ACCEPTED decoy link on a fold-B S1 is counted k times; fractional k duplicates with probability k-1 (seeded).
mode 'same'  : duplicates land on the same S1;   mode 'spread': duplicates land on uniformly random fold-B S1.
Decision assembly identical to Experiments 08-14 (fold-B band rows: Stage-2 p; everything else: stage-1 p1).
"""
import numpy as np
import polars as pl

import common
from common import stage2
from metrics import macro_f05


class DensityEval:
    def __init__(self):
        s1, q, tp, best = stage2.train_rows()
        flags = pl.concat([pl.read_parquet(common.ROOT / f"cache/train_{s}_norm.parquet", columns=["addr_missing", "country"])
                           for s in ("source2", "source3")]).with_row_index("q_idx")
        self.tp = tp
        self.val_ids = s1.filter(pl.col("is_val"))["entity_id"]
        self.best = best.select("q_idx", "s1_idx", "p1", "true_s1", "q_is_val").join(flags, on="q_idx", how="left") \
            .join(s1.select("s1_idx", pl.col("entity_id").alias("s1_id")), on="s1_idx") \
            .join(q.select("q_idx", pl.col("entity_id").alias("m_id")), on="q_idx")

    def frame(self, scored):
        """scored: DataFrame(q_idx, p_stage2) for fold-B band rows -> best with p_final."""
        return self.best.join(scored.select("q_idx", "p_stage2"), on="q_idx", how="left") \
                        .with_columns(pl.coalesce("p_stage2", "p1").alias("p_final"))

    def evaluate(self, b, thr_expr, k=1.0, mode="same", seed=0):
        acc = b.filter(pl.col("p_final") >= thr_expr)
        pairs = acc.select("s1_id", "m_id")
        if k > 1.0:
            dec = acc.filter(pl.col("true_s1").is_null() & pl.col("s1_id").is_in(self.val_ids.implode())).select("s1_id", "m_id")
            rng = np.random.default_rng(seed)
            extra, left, j = [], k - 1.0, 0
            while left > 1e-9:
                prob = min(1.0, left)
                sel = dec.filter(pl.Series(rng.random(dec.height) < prob))
                if mode == "spread":
                    r = rng.integers(0, len(self.val_ids), sel.height)
                    sel = pl.DataFrame({"s1_id": self.val_ids.gather(r), "m_id": sel["m_id"]})
                extra.append(sel.with_columns((pl.col("m_id") + f"#dup{j}").alias("m_id")))
                left -= prob
                j += 1
            pairs = pl.concat([pairs] + extra)
        m = macro_f05(pairs, self.tp, self.val_ids)
        pe = m["per_entity"]
        return {"f05": m["macro_f05"], "precision": m["macro_precision"], "recall": m["macro_recall"],
                "false_singletons": pe.filter((pl.col("n_true") == 0) & (pl.col("n_pred") > 0)).height,
                "accepted": acc.height}
