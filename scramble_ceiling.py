#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scramble_ceiling.py — the ceiling check.

If the detector cannot flag a verse whose words have been shuffled into
ungrammatical Hebrew, then it is not insensitive to Jeremiah in particular;
it is insensitive to anything a single verse can do. One verse of Isaiah is
replaced in place by its own words in random order, so length, vocabulary and
position are held fixed and only the syntax is destroyed.

Both scores have to be recomputed here: DictaBERT reads the verse, and the
shuffled verse is a different string.
"""
import json, random, sys
import numpy as np, pandas as pd
from data_manager import DataManager
from seismograph_engine import SeismographEngine
from dicta_engine import DictaEngine

Z, K, CTX, TOL, MARGIN = 1.9, 5, 3, 1, 25
N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 30


def roll_z(v):
    s = pd.Series(v, dtype=float)
    m = s.rolling(K, min_periods=1, center=True).mean()
    sd = s.rolling(K, min_periods=1, center=True).std(ddof=0)
    return ((s - m) / sd).where(sd > 0, 0.0).to_numpy()


dm = DataManager()
V = [{"verse_id": v["verse_id"], "text": dm.clean_hebrew(v["text"])}
     for v in dm.parse_verses(dm.get_text("Isaiah"))]
texts = [v["text"] for v in V]
pub = lambda m: {r["verse_id"]: r["perplexity_score"]
                 for r in json.load(open(f"output/isaiah_{m}.scores.json"))["records"]}
base_gpt = [pub("gpt")[v["verse_id"]] for v in V]
base_pll = [pub("dicta")[v["verse_id"]] for v in V]

se, de = SeismographEngine(), DictaEngine()
rows = []
for t in range(N_TRIALS):
    rng = random.Random(f"scramble|{t}")
    i = rng.randrange(MARGIN, len(V) - MARGIN)
    words = texts[i].split()
    if len(words) < 4:
        continue
    shuffled = words[:]
    while " ".join(shuffled) == texts[i]:
        rng.shuffle(shuffled)
    new = " ".join(shuffled)

    tx = list(texts); tx[i] = new
    g = list(base_gpt); p = list(base_pll)
    for j in range(i, min(len(tx), i + CTX + 1)):          # the verse and what reads it
        g[j] = se._score_transition(tx, j, CTX)
    p[i] = de._compute_pll(new)

    zg, zd = roll_z(g), roll_z(p)
    win = [j for j in range(i - TOL, i + TOL + 1) if 0 <= j < len(tx)]
    rows.append({
        "verse": V[i]["verse_id"], "n_words": len(words),
        "ppl_gpt_before": base_gpt[i], "ppl_gpt_after": g[i],
        "pll_before": base_pll[i], "pll_after": p[i],
        "zg_after": max(zg[j] for j in win), "zd_after": max(zd[j] for j in win),
        "hit_shared": any(zg[j] >= Z and zd[j] >= Z for j in win),
        "hit_gpt": any(zg[j] >= Z for j in win),
        "hit_dicta": any(zd[j] >= Z for j in win),
    })
    print(f"  {t+1}/{N_TRIALS} {V[i]['verse_id']:<10} "
          f"ppl {base_gpt[i]:8.1f}->{g[i]:8.1f}   pll {base_pll[i]:10.1f}->{p[i]:10.1f}  "
          f"shared={rows[-1]['hit_shared']}", flush=True)

df = pd.DataFrame(rows)
df.to_csv("output/scramble_ceiling.csv", index=False)
print("\n" + "=" * 64)
print(f"n = {len(df)} verses, each replaced by its own words in random order")
print(f"  GPT-Neo perplexity   median x{np.median(df.ppl_gpt_after / df.ppl_gpt_before):.2f}")
print(f"  DictaBERT pseudo-ppl median x{np.median(df.pll_after / df.pll_before):.2f}")
print(f"  flagged by GPT-Neo   {df.hit_gpt.mean():.0%}")
print(f"  flagged by DictaBERT {df.hit_dicta.mean():.0%}")
print(f"  flagged by BOTH (the published rule) {df.hit_shared.mean():.0%}")
