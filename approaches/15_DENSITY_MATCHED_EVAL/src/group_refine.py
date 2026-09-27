"""Exp 15f: refine the lone-lenient / group-strict rule; two seeds; proxy spread k=2.4 (and k=1.9, plain). Read-only."""
import polars as pl
import common
from density import DensityEval
E11 = common.ROOT / "approaches/11_STAGE2_ADDRESS_TOKEN_ODDS/cache"
ev = DensityEval()
rows = []
for name, f in (("R2_s42", "val_scored_R2_hn_ate.parquet"), ("R2_s7", "val_scored_R2_hn_ate_s7.parquet")):
    base = ev.frame(pl.read_parquet(E11 / f))
    for conf in (0.8, 0.9, 0.95):
        b = base.with_columns(((pl.col("p_final") >= conf).cast(pl.Int32).sum().over("s1_idx") - (pl.col("p_final") >= conf).cast(pl.Int32)).alias("n"))
        grid = [(0.6, 0.85), (0.65, 0.85), (0.7, 0.85), (0.6, 0.9), (0.7, 0.9)] if conf == 0.9 else [(0.7, 0.85)]
        for ta, tg in grid:
            thr = pl.when(pl.col("n") > 0).then(pl.lit(tg)).otherwise(pl.lit(ta))
            p, q, r = ev.evaluate(b, thr, 2.4, "spread"), ev.evaluate(b, thr, 1.9, "spread"), ev.evaluate(b, thr)
            rows.append({"model": name, "conf": conf, "t_alone": ta, "t_group": tg, "lb_proxy": p["f05"] - 0.0019,
                         "lb_proxy_k1.9": q["f05"] - 0.0019, "plain": r["f05"], "P": p["precision"], "R": p["recall"], "FS": p["false_singletons"]})
            print(f"{name} conf={conf} alone={ta} group={tg}: LB-proxy {p['f05'] - 0.0019:.6f} (k1.9 {q['f05'] - 0.0019:.6f}, plain {r['f05']:.6f})", flush=True)
    u = ev.evaluate(base, pl.lit(0.8), 2.4, "spread")
    print(f"{name} uniform 0.80: LB-proxy {u['f05'] - 0.0019:.6f}", flush=True)
pl.DataFrame(rows).write_csv(common.ROOT / "approaches/15_DENSITY_MATCHED_EVAL/results/group_refine.csv")
