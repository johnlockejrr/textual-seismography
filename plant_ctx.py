#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plant_ctx.py — the same planting experiment, with both models reading context.

The published masked score reads a verse alone, so it cannot witness a join.
Here DictaBERT reads the three preceding verses too, and the signal for both
models is the *context gain*: how much the preceding verses help.

    gain_gpt   = log ppl(verse alone)      - log ppl(verse after its context)
    gain_dicta = log pll(verse alone)      - log pll(verse after its context)

A verse's own difficulty appears in both terms of each difference and cancels,
which is what verse length, genre and rarity were riding on. What is left is
fit to what precedes — the only thing a join can break.

The same trial seeds as `plant_seams.py` are used, so a trial here is the same
splice at the same position as the trial of that name there, and the two can be
compared directly. Both contextual scores depend on a verse and the three
before it, so only the spliced verses and the three after them are recomputed.

  python plant_ctx.py --donor Micah --seed confirm-2026-c --lengths 12,24
"""
import argparse, json, math, os, random, time
import numpy as np
import pandas as pd

from data_manager import DataManager
from seismograph_engine import SeismographEngine
from contextual_pll import ContextualDicta

CONTEXT, MIN_GAP, MARGIN, OUT = 3, 100, 25, "output"


def load_scores(book, tag):
    p = os.path.join(OUT, f"{book.lower()}_{tag}.scores.json")
    return {r["verse_id"]: r["perplexity_score"] for r in json.load(open(p))["records"]}


class Pair:
    """The two contextual scorers, with a cache keyed on the verse and its context."""

    def __init__(self):
        self.gpt = SeismographEngine()
        self.dicta = ContextualDicta(left=CONTEXT)
        self.cache = {}
        self.calls = 0

    def score(self, texts, i):
        key = tuple(texts[max(0, i - CONTEXT): i + 1])
        if key not in self.cache:
            j = len(key) - 1
            self.cache[key] = (self.gpt._score_transition(list(key), j, CONTEXT),
                               self.dicta.pll(list(key), j))
            self.calls += 1
        return self.cache[key]


def splice(seqs, pos, blocks, cut=None):
    out = [list(s) for s in seqs]
    if cut is not None:
        lo, hi = cut
        for s in out:
            del s[lo:hi]
        if pos > lo:
            pos -= (hi - lo)
    for s, b in zip(out, blocks):
        s[pos:pos] = list(b)
    return out, pos, pos + len(blocks[0])


def run(a):
    dm = DataManager()
    host = [{"verse_id": v["verse_id"], "text": dm.clean_hebrew(v["text"])}
            for v in dm.parse_verses(dm.get_text(a.host))]
    donor = [{"verse_id": v["verse_id"], "text": dm.clean_hebrew(v["text"])}
             for v in dm.parse_verses(dm.get_text(a.donor))]
    ht = [v["text"] for v in host]; dt = [v["text"] for v in donor]
    hid = [v["verse_id"] for v in host]; did = [v["verse_id"] for v in donor]

    # context-free references: these travel with the verse
    h_iso_g = np.array([load_scores(a.host, "gpt_isolated")[i] for i in hid])
    h_free_d = np.array([load_scores(a.host, "dicta")[i] for i in hid])
    d_iso_g = np.array([load_scores(a.donor, "gpt_isolated")[i] for i in did])
    d_free_d = np.array([load_scores(a.donor, "dicta")[i] for i in did])

    P = Pair()
    base_ctx_g = np.array([load_scores(a.host, "gpt")[i] for i in hid])
    base_ctx_d = np.array([load_scores(a.host, "dicta_context")[i] for i in hid])
    # the cached base must be reproducible, or the incremental trick is unsound
    chk = [P.score(ht, i) for i in (200, 500, 900)]
    for j, i in enumerate((200, 500, 900)):
        assert abs(chk[j][0] - base_ctx_g[i]) / base_ctx_g[i] < 1e-3, "GPT base mismatch"
        assert abs(chk[j][1] - base_ctx_d[i]) / base_ctx_d[i] < 1e-3, "DictaBERT base mismatch"
    print("[check] the cached base reproduces on a sample of three verses", flush=True)

    n = len(host)
    series, meta = {"base|gain_g": np.log(h_iso_g) - np.log(base_ctx_g),
                    "base|gain_d": np.log(h_free_d) - np.log(base_ctx_d),
                    "base|wc": np.array([len(t.split()) for t in ht], dtype="float64")}, []
    t0 = time.time()
    for L in a.lengths:
        for t in range(a.trials):
            rng = random.Random(f"{a.seed}|{L}|{t}")
            pos = rng.randrange(MARGIN, n - MARGIN)
            d0 = rng.randrange(0, len(donor) - L)
            s0 = None
            for _ in range(200):
                c = rng.randrange(MARGIN, n - MARGIN - L)
                if abs(c - pos) >= MIN_GAP:
                    s0 = c; break
            for cond in ("plant", "move"):
                if cond == "plant":
                    blocks = (dt[d0:d0+L], d_iso_g[d0:d0+L], d_free_d[d0:d0+L])
                    cut = None
                else:
                    blocks = (ht[s0:s0+L], h_iso_g[s0:s0+L], h_free_d[s0:s0+L])
                    cut = (s0, s0 + L)
                (tx, iso_g, free_d), e_in, e_out = splice(
                    (ht, h_iso_g, h_free_d), pos, blocks, cut)
                changed = set(range(e_in, min(len(tx), e_out + CONTEXT)))
                if cut is not None:
                    c2 = cut[0] if cut[0] < pos else cut[0] + L
                    changed |= set(range(max(0, c2), min(len(tx), c2 + CONTEXT + 1)))
                kg, kd = list(base_ctx_g), list(base_ctx_d)
                if cut is not None:
                    del kg[cut[0]:cut[1]]; del kd[cut[0]:cut[1]]
                kg[e_in:e_in] = [np.nan]*L; kd[e_in:e_in] = [np.nan]*L
                cg, cd = [], []
                for i in range(len(tx)):
                    if i in changed:
                        g, d = P.score(tx, i)
                    else:
                        g, d = kg[i], kd[i]
                    cg.append(g); cd.append(d)
                assert not any(math.isnan(x) for x in cg + cd), "a spliced verse went unscored"
                key = f"{cond}|{L}|{t}"
                series[key + "|gain_g"] = np.log(np.asarray(iso_g)) - np.log(np.asarray(cg))
                series[key + "|gain_d"] = np.log(np.asarray(free_d)) - np.log(np.asarray(cd))
                series[key + "|wc"] = np.array([len(x.split()) for x in tx], dtype="float64")
                meta.append(dict(key=key, condition=cond, length=L, trial=t,
                                 e_in=e_in, e_out=e_out, n=len(tx)))
        print(f"  L={L} done, {P.calls} model calls, {time.time()-t0:.0f}s", flush=True)

    stem = os.path.join(OUT, f"ctx_{a.donor.lower()}_L{'-'.join(map(str, a.lengths))}")
    np.savez_compressed(stem + "_series.npz", **series)
    pd.DataFrame(meta).to_csv(stem + "_meta.csv", index=False)
    print(f"[saved] {stem}_series.npz  ({P.calls} paired model calls)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="Isaiah")
    ap.add_argument("--donor", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--lengths", default="12,24")
    x = ap.parse_args()
    x.lengths = [int(v) for v in x.lengths.split(",")]
    run(x)
