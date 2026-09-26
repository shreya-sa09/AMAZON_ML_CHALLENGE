"""Supplement to audit.py: decoy-density runs for the 'excluded' and 'fixed_0.7' fold-A treatments."""
import sys
sys.argv = [sys.argv[0]]
import audit as A  # loads the same data (no grid/submissions: those run only under __main__)
import polars as pl

rows = []
for mode in ("excluded", "fixed_0.7"):
    for t in A.C["thresholds"]:
        ids = A.pairs_at(t, mode)
        for k in (2, 3):
            r, _ = A.metrics(A.with_decoys(ids, k, "same"))
            rows.append({"fold_a": mode, "decoys": f"x{k}", "threshold": t, **r})
        A.log(f"{mode} t={t}: " + " | ".join(f"{x['decoys']} {x['f05']:.6f}" for x in rows if x['fold_a'] == mode and x['threshold'] == t))
pl.DataFrame(rows).write_csv(A.RESULTS_DIR / "threshold_grid_extra_density.csv")
