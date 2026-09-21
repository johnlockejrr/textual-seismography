#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two controls the first design got wrong.

The move control took a block from at least a hundred verses away, inside a
book whose composite character is not in dispute — so it was very likely
carrying material across a real compositional boundary and calling the
detection a false alarm. Two tests separate the readings:

  1. split the existing long-distance moves by whether they cross one of the
     traditional divisions of the book (1-39 / 40-55 / 56-66);
  2. a short move, ten to twenty-five verses, wholly inside one division,
     where no recognised boundary is crossed. That is the control the design
     needed: if the detector is silent there, it is answering to composition
     and not to the act of cutting.

Verse vectors travel with their verses, so no model runs.
"""
import random
import numpy as np, pandas as pd
import evaluate_variants as ev
from embed_eval import rebuild2d, score, SEEDS
from data_manager import DataManager

K, MARGIN = 5, 25
SECT = [(1, 39), (40, 55), (56, 66)]


def sections(ids):
    ch = np.array([int(v.split(".")[1]) for v in ids])
    lab = np.full(len(ids), -1)
    for j, (lo, hi) in enumerate(SECT):
        lab[(ch >= lo) & (ch <= hi)] = j
    return lab


def main():
    dm = DataManager()
    ids = [v["verse_id"] for v in dm.parse_verses(dm.get_text("Isaiah"))]
    lab = sections(ids)
    H = np.load("output/emb_isaiah_berel.npy")
    base = score(H, "cosine", K)
    n = len(H)

    # ---- 1. the moves already run, split by whether they cross a division ----
    print("TEST 1 — long-distance moves, split by the traditional divisions\n")
    for tag in ("jeremiah", "habakkuk", "micah"):
        stem = f"output/planted_seams_{tag}_L1-3-6-12-24"
        _, meta = ev.load(stem)
        rec = []
        for m in meta[meta.condition == "move"].itertuples():
            rng = random.Random(f"{SEEDS[tag]}|{m.length}|{m.trial}")
            pos = rng.randrange(MARGIN, n - MARGIN)
            rng.randrange(0, 40)                     # the donor draw, skipped
            s0 = None
            for _ in range(200):
                c = rng.randrange(MARGIN, n - MARGIN - m.length)
                if abs(c - pos) >= 100:
                    s0 = c; break
            _, M = rebuild2d(H, H, SEEDS[tag], m.length, m.trial, n)
            s = score(M, "cosine", K)
            r = np.random.default_rng(abs(hash((m.length, m.trial))) % 2**32)
            q = int(r.integers(25, n - 25 - m.length))
            rec.append({"length": m.length,
                        "cross": bool(lab[s0] != lab[min(pos, n - 1)]),
                        "pct_spliced": ev.percentile_of_edge(s, ev.edge_positions(m.e_in, m.e_out, m.n)),
                        "pct_untouched": ev.percentile_of_edge(base, ev.edge_positions(q, q + m.length, n))})
        d = pd.DataFrame(rec); long = d[d.length >= 12]
        for cross, g in long.groupby("cross"):
            print(f"   {tag:<10} {'crosses a division' if cross else 'within one division':<20} "
                  f"n={len(g):>3}  AUC {ev.auc(g):.3f}  top1% {(g.pct_spliced>=99).mean():.0%}")
        print()

    # ---- 2. the short move, wholly inside one division ----
    print("TEST 2 — short move (10-25 verses) inside a single division\n")
    rec = []
    rng = random.Random("shortmove-2026")
    for L in (12, 24):
        made = 0
        while made < 60:
            j = rng.randrange(3)
            idx = np.where(lab == j)[0]
            lo, hi = idx.min(), idx.max()
            if hi - lo < 120:
                continue
            s0 = rng.randrange(lo + 5, hi - L - 30)
            gap = rng.randrange(10, 26)
            pos = s0 + L + gap
            if pos + 5 > hi or lab[pos] != j or lab[s0 + L] != j:
                continue
            keep = np.vstack([H[:s0], H[s0 + L:]])
            p2 = pos - L
            M = np.vstack([keep[:p2], H[s0:s0 + L], keep[p2:]])
            assert len(M) == n
            s = score(M, "cosine", K)
            e_in, e_out = p2, p2 + L
            q = rng.randrange(25, n - 25 - L)
            rec.append({"length": L, "gap": gap,
                        "pct_spliced": ev.percentile_of_edge(s, ev.edge_positions(e_in, e_out, n)),
                        "pct_untouched": ev.percentile_of_edge(base, ev.edge_positions(q, q + L, n))})
            made += 1
        g = pd.DataFrame(rec); g = g[g.length == L]
        print(f"   L={L}: n={len(g)}  AUC {ev.auc(g):.3f}  "
              f"top1% {(g.pct_spliced>=99).mean():.0%} (control {(g.pct_untouched>=99).mean():.0%})")
    d = pd.DataFrame(rec)
    print(f"\n   both lengths: AUC {ev.auc(d):.3f}")
    d.to_csv("output/short_move_control.csv", index=False)


if __name__ == "__main__":
    main()
