#!/usr/bin/env python3
"""The repaired search: length residualised out, conservative control, and the
no-model length baseline carried alongside every variant."""
import itertools, sys
import numpy as np, pandas as pd
import evaluate_variants as ev

STEM = sys.argv[1]
npz, meta = ev.load(STEM); wc = ev.load_wc(STEM)
assert wc is not None, "word counts missing"

grid = []
for scale, win, spread, comb, res in itertools.product(
        ["raw", "log"], [5, 9, 15, 21, "global"],
        ["mean_sd", "median_mad", "loo_mean_sd"],
        ["sum", "min", "geo", "max", "gpt", "dicta"], [True, False]):
    if win == "global" and spread == "loo_mean_sd":
        continue
    grid.append(dict(scale=scale, window=win, spread=spread, combine=comb,
                     operator="point", step_k=0, residual=res))
    for sk in [5, 10, 20]:
        grid.append(dict(scale=scale, window=win, spread=spread, combine=comb,
                         operator="step", step_k=sk, residual=res))

print(f"{len(grid)} variants (half of them with verse length regressed out)")
rows = []
for i, v in enumerate(grid):
    d = ev.evaluate(npz, meta, v, "plant", wc, control="random")
    mv = ev.evaluate(npz, meta, v, "move", wc, control="random")
    long = d[d.length >= 12]
    rows.append({**v, "auc": ev.auc(d), "auc_long": ev.auc(long),
                 "auc_move": ev.auc(mv),
                 "top1_long": float((long.pct_spliced >= 99).mean()),
                 "top1_long_ctrl": float((long.pct_untouched >= 99).mean())})
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(grid)}", flush=True)

df = pd.DataFrame(rows).sort_values("auc_long", ascending=False)
df.to_csv(STEM + "_sweep2.csv", index=False)

# the reference every variant has to beat: word counts, no model at all
for sk in [10, 20]:
    lv = dict(scale="log", window="global", spread="mean_sd", combine="sum",
              operator="step", step_k=sk, residual=False, signal="length")
    d = ev.evaluate(npz, meta, lv, "plant", wc, control="random")
    print(f"\nLENGTH ONLY, no model (step k={sk}): AUC {ev.auc(d):.3f}   "
          f"long blocks {ev.auc(d[d.length>=12]):.3f}")

pd.set_option("display.width", 220)
print("\nTOP 12 by AUC on blocks of 12-24 verses:")
cols = ["scale","window","spread","combine","operator","step_k","residual",
        "auc","auc_long","auc_move","top1_long","top1_long_ctrl"]
print(df.head(12)[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("\nBest variant WITH length residualised out:")
print(df[df.residual].head(5)[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
