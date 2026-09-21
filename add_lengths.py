#!/usr/bin/env python3
"""Reconstruct the word-count series for runs made before they were stored.

Verse length is the confound the method has to be cleared of, so it has to be
available for every trial. The splices are deterministic in the seed, so the
sequences can be rebuilt exactly; each one is checked against the length the
run itself recorded.
"""
import random, sys
import numpy as np, pandas as pd
from data_manager import DataManager
import evaluate_variants as ev

MARGIN, MIN_GAP = 25, 100


def rebuild(host_wc, donor_wc, seed, L, t, n):
    rng = random.Random(f"{seed}|{L}|{t}")
    pos = rng.randrange(MARGIN, n - MARGIN)
    d0 = rng.randrange(0, len(donor_wc) - L)
    s0 = None
    for _ in range(200):
        c = rng.randrange(MARGIN, n - MARGIN - L)
        if abs(c - pos) >= MIN_GAP:
            s0 = c
            break
    plant = np.concatenate([host_wc[:pos], donor_wc[d0:d0 + L], host_wc[pos:]])
    m = list(host_wc); blk = m[s0:s0 + L]; del m[s0:s0 + L]
    p2 = pos - L if pos > s0 else pos
    m[p2:p2] = blk
    return plant, np.asarray(m, dtype="float64")


def main(stem, donor_name, seed="planted-seams-2026", host_name="Isaiah"):
    dm = DataManager()
    hw = np.array([len(dm.clean_hebrew(v["text"]).split())
                   for v in dm.parse_verses(dm.get_text(host_name))], dtype="float64")
    dw = np.array([len(dm.clean_hebrew(v["text"]).split())
                   for v in dm.parse_verses(dm.get_text(donor_name))], dtype="float64")
    npz, meta = ev.load(stem)
    out = {"base|wc": hw}
    for m in meta.itertuples():
        if m.condition == "null":
            continue
        pl, mv = rebuild(hw, dw, seed, m.length, m.trial, len(hw))
        arr = pl if m.condition == "plant" else mv
        assert len(arr) == m.n, f"rebuilt length {len(arr)} != recorded {m.n}"
        out[f"{m.condition}|{m.length}|{m.trial}|wc"] = arr
    np.savez_compressed(stem + "_wc.npz", **out)
    print(f"{stem}: rebuilt {len(out)-1} word-count series, all lengths agree with the run")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
