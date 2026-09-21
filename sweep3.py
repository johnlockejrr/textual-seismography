#!/usr/bin/env python3
"""Round three: joint use of the two signals, direction, and distribution shape."""
import itertools, sys
import numpy as np, pandas as pd
import evaluate_variants as ev

STEM = sys.argv[1]
npz, meta = ev.load(STEM); wc = ev.load_wc(STEM)
grid = []
for scale, win, spread, res, sk in itertools.product(
        ["log"], [21, "global"], ["median_mad", "mean_sd"], [True, False], [5, 10, 20]):
    grid.append(dict(scale=scale, window=win, spread=spread, combine="sum",
                     operator="hotelling", step_k=sk, residual=res))
    for op in ["step", "onesided", "mw"]:
        for comb in ["sum", "min", "diff", "geo", "gpt", "dicta"]:
            grid.append(dict(scale=scale, window=win, spread=spread, combine=comb,
                             operator=op, step_k=sk, residual=res))
print(f"{len(grid)} variants")
rows = []
for i, v in enumerate(grid):
    d = ev.evaluate(npz, meta, v, "plant", wc, control="random")
    mv = ev.evaluate(npz, meta, v, "move", wc, control="random")
    long = d[d.length >= 12]
    rows.append({**v, "auc": ev.auc(d), "auc_long": ev.auc(long), "auc_move": ev.auc(mv),
                 "top1_long": float((long.pct_spliced >= 99).mean())})
    if (i + 1) % 100 == 0: print(f"  {i+1}/{len(grid)}", flush=True)
df = pd.DataFrame(rows).sort_values("auc_long", ascending=False)
df.to_csv(STEM + "_sweep3.csv", index=False)
cols = ["operator","combine","step_k","window","spread","residual","auc","auc_long","auc_move","top1_long"]
pd.set_option("display.width", 200)
print("\nTOP 15 (blocks 12-24):")
print(df.head(15)[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\nbest per operator:")
for op in ["hotelling","step","onesided","mw"]:
    g=df[df.operator==op]
    if len(g): print(g.head(1)[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
