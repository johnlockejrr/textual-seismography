#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score the planting trials on verse vectors instead of two scalars.

A verse is now a point in 768 dimensions, and a boundary is a change of
direction rather than of level — which is the thing the scalar line could not
see. The trials are the ones already on disk: a vector travels with its verse,
so the spliced sequences are rebuilt exactly as the word counts were, and no
model runs again.

Two statistics at each boundary, over k verses either side:
  cosine   the angle between the two mean vectors
  hotelling  the same gap measured in units of the book's own covariance,
             after reducing to a few principal components
"""
import sys, random
import numpy as np, pandas as pd
import evaluate_variants as ev
from data_manager import DataManager

MARGIN, MIN_GAP = 25, 100
SEEDS = {"jeremiah": "planted-seams-2026", "leviticus": "planted-seams-2026",
         "habakkuk": "heldout-2026-b", "micah": "confirm-2026-c"}


def rebuild2d(H, D, seed, L, t, n):
    rng = random.Random(f"{seed}|{L}|{t}")
    pos = rng.randrange(MARGIN, n - MARGIN)
    d0 = rng.randrange(0, len(D) - L)
    s0 = None
    for _ in range(200):
        c = rng.randrange(MARGIN, n - MARGIN - L)
        if abs(c - pos) >= MIN_GAP:
            s0 = c; break
    plant = np.vstack([H[:pos], D[d0:d0 + L], H[pos:]])
    keep = np.vstack([H[:s0], H[s0 + L:]])
    p2 = pos - L if pos > s0 else pos
    move = np.vstack([keep[:p2], H[s0:s0 + L], keep[p2:]])
    return plant, move


def cosine_step(E, k):
    n = len(E)
    cs = np.vstack([np.zeros(E.shape[1]), np.cumsum(E, 0)])
    out = np.zeros(n)
    for i in range(n):
        a0, a1, b0, b1 = max(0, i - k), i, i, min(n, i + k)
        if a1 - a0 < 2 or b1 - b0 < 2:
            continue
        a = (cs[a1] - cs[a0]) / (a1 - a0); b = (cs[b1] - cs[b0]) / (b1 - b0)
        out[i] = 1.0 - float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return out


def pca(E, d=20):
    X = E - E.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:d].T


def score(E, how, k):
    if how == "cosine":
        s = cosine_step(E, k)
    else:
        s = ev.hotelling_step(pca(E, 20), k)
    return ev.standardise(s, "global", "mean_sd")


def run(tag, donor, ks=(5, 10, 20)):
    stem = f"output/planted_seams_{tag}_L1-3-6-12-24"
    _, meta = ev.load(stem)
    H = np.load("output/emb_isaiah_berel.npy")
    D = np.load(f"output/emb_{donor.lower()}_berel.npy")
    rows = []
    for how in ("cosine", "hotelling"):
        for k in ks:
            base = score(H, how, k)
            for cond in ("plant", "move"):
                rec = []
                for m in meta[meta.condition == cond].itertuples():
                    P, M = rebuild2d(H, D, SEEDS[tag], m.length, m.trial, len(H))
                    E = P if cond == "plant" else M
                    assert len(E) == m.n, f"rebuilt {len(E)} != recorded {m.n}"
                    s = score(E, how, k)
                    pos = ev.edge_positions(m.e_in, m.e_out, m.n)
                    r = np.random.default_rng(abs(hash((m.length, m.trial))) % 2**32)
                    q = int(r.integers(25, len(base) - 25 - m.length))
                    rec.append({"length": m.length,
                                "pct_spliced": ev.percentile_of_edge(s, pos),
                                "pct_untouched": ev.percentile_of_edge(
                                    base, ev.edge_positions(q, q + m.length, len(base)))})
                d = pd.DataFrame(rec); long = d[d.length >= 12]
                rows.append({"stat": how, "k": k, "cond": cond, "auc": ev.auc(d),
                             "auc_long": ev.auc(long),
                             "top1_long": float((long.pct_spliced >= 99).mean()),
                             "top1_ctrl": float((long.pct_untouched >= 99).mean())})
            print(f"  {how} k={k} done", flush=True)
    df = pd.DataFrame(rows)
    p = df.pivot_table(index=["stat", "k"], columns="cond",
                       values=["auc", "auc_long", "top1_long"])
    print(f"\n=== {tag.upper()} (donor {donor}) — BEREL verse vectors ===")
    print(p.round(3).to_string())
    return df


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
