#!/usr/bin/env python3
"""Per-trial series of the context-free score, in the spliced order.

The isolated perplexity belongs to the verse, so it travels with it; the
sequences are rebuilt exactly as the word counts are, and checked the same way.
"""
import json, sys
import numpy as np
from data_manager import DataManager
import evaluate_variants as ev
from add_lengths import rebuild

def main(stem, donor, seed, host="Isaiah"):
    dm = DataManager()
    def iso(book):
        d = json.load(open(f"output/{book.lower()}_gpt_isolated.scores.json"))
        return np.array([r["perplexity_score"] for r in d["records"]], dtype="float64")
    hi, di = iso(host), iso(donor)
    npz, meta = ev.load(stem)
    out = {"base|iso": hi}
    for m in meta.itertuples():
        if m.condition == "null":
            continue
        pl, mv = rebuild(hi, di, seed, m.length, m.trial, len(hi))
        arr = pl if m.condition == "plant" else mv
        assert len(arr) == m.n
        out[f"{m.condition}|{m.length}|{m.trial}|iso"] = arr
    np.savez_compressed(stem + "_iso.npz", **out)
    print(f"{stem}: {len(out)-1} isolated series")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
