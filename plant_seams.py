#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plant_seams.py — can the two-witness detector find a seam that we put there?

The paper's claim is that a verse flagged by both models marks a join. This
script tests the claim the only way it can be tested: by making joins whose
position is known and asking whether the detector finds them.

Three conditions, identical in every respect except what is spliced in:

  plant   L consecutive verses of Jeremiah inserted at position p of Isaiah.
          Same genre, same register, neighbouring period — a hard test, not
          the easy one you get by splicing poetry into late prose.
  move    L consecutive verses of Isaiah taken from at least MIN_GAP verses
          away and inserted at p, REMOVED from where they came from. Nothing
          is duplicated. This is the negative control: a real seam in the
          book's own material, which the detector should mostly not flag.
  null    nothing is spliced. The same rule is evaluated at random positions
          of the untouched book, which gives the rate at which the rule fires
          on its own.

Detection is decided in advance, not read off a maximum. A trial counts as a
hit when a shared seam (both rolling Z >= 1.9) falls within EDGE_TOL verses of
either edge of the spliced block. Taking the global maximum of a score and
measuring its distance to the truth would report a hit whenever the maximum
happened to land nearby, which for a book this size happens often.

Scoring is the published pipeline, unchanged: GPT-Neo perplexity of a verse
given up to three preceding verses, DictaBERT pseudo-perplexity of the verse
alone, both turned into rolling Z over a centred window of five.

Only what changes is recomputed. DictaBERT reads one verse at a time, so a
verse's pseudo-perplexity travels with it; GPT-Neo reads three verses back, so
exactly the spliced verses and the three verses after them change. Everything
else is reused, and the reuse is checked against the published scores first.

  python plant_seams.py --trials 40 --lengths 1,3,6,12,24
"""
import argparse, json, math, os, random, time
import numpy as np
import pandas as pd

from data_manager import DataManager
from seismograph_engine import SeismographEngine

Z_THRESHOLD = 1.9      # the published shared-seam threshold
ROLL_WINDOW = 5        # the published rolling window
CONTEXT     = 3        # the published causal context, in verses
EDGE_TOL    = 1        # verses either side of an edge that still count as a hit
MIN_GAP     = 100      # how far a moved block must travel, in verses
MARGIN      = 25       # keep splices away from the ends of the book
OUT         = "output"

SECTIONS = [(1, 39, "1-39"), (40, 55, "40-55"), (56, 66, "56-66")]


def section_of(verse_id):
    ch = int(verse_id.split(".")[1])
    for lo, hi, name in SECTIONS:
        if lo <= ch <= hi:
            return name
    return "?"


# ---------------------------------------------------------------- scoring ---
def rolling_z(values):
    """The published normalisation: centred window, min_periods=1, ddof=0."""
    s = pd.Series(values, dtype="float64")
    mean = s.rolling(ROLL_WINDOW, min_periods=1, center=True).mean()
    std = s.rolling(ROLL_WINDOW, min_periods=1, center=True).std(ddof=0)
    z = (s - mean) / std
    return z.where(std > 0, 0.0).to_numpy()


class Scorer:
    """GPT-Neo perplexities for spliced sequences, with a cache.

    A verse's causal score depends only on itself and the CONTEXT verses in
    front of it, so the cache key is that tuple of texts. Nothing else about
    the sequence can change it.
    """

    def __init__(self, engine):
        self.engine = engine
        self.cache = {}
        self.calls = 0

    def score(self, texts, i):
        key = tuple(texts[max(0, i - CONTEXT): i + 1])
        if key not in self.cache:
            self.cache[key] = self.engine._score_transition(list(key), len(key) - 1, CONTEXT)
            self.calls += 1
        return self.cache[key]

    def score_all(self, texts, known=None, changed=None):
        """Perplexities for the whole sequence.

        `known` supplies values for positions whose context is untouched;
        `changed` is the set of positions that must be recomputed. When both
        are None every verse is scored.
        """
        out = []
        for i in range(len(texts)):
            if known is not None and changed is not None and i not in changed:
                out.append(known[i])
            else:
                out.append(self.score(texts, i))
        return out


# --------------------------------------------------------------- splicing ---
def splice(host_texts, host_pll, host_ids, block_texts, block_pll, block_ids,
           pos, cut=None):
    """Insert a block at `pos`; if `cut` is given, first remove it from there.

    Returns the new sequences and the two edge positions in the new index:
    the first verse of the block, and the first host verse after it.
    """
    texts, pll, ids = list(host_texts), list(host_pll), list(host_ids)
    if cut is not None:
        lo, hi = cut
        del texts[lo:hi]; del pll[lo:hi]; del ids[lo:hi]
        if pos > lo:                      # the removal shifted the target
            pos -= (hi - lo)
    texts[pos:pos] = list(block_texts)
    pll[pos:pos]   = list(block_pll)
    ids[pos:pos]   = list(block_ids)
    return texts, pll, ids, pos, pos + len(block_texts)


def edge_window(edges, n):
    idx = set()
    for e in edges:
        for d in range(-EDGE_TOL, EDGE_TOL + 1):
            if 0 <= e + d < n:
                idx.add(e + d)
    return sorted(idx)


# ------------------------------------------------------------------ trial ---
def evaluate(zg, zd, window):
    """The published rule, read off at the positions we decided in advance."""
    cfi = zg + zd
    shared = [i for i in window if zg[i] >= Z_THRESHOLD and zd[i] >= Z_THRESHOLD]
    return {
        "hit_shared": len(shared) > 0,
        "hit_gpt":    any(zg[i] >= Z_THRESHOLD for i in window),
        "hit_dicta":  any(zd[i] >= Z_THRESHOLD for i in window),
        "max_cfi":    float(max(cfi[i] for i in window)),
        "max_zg":     float(max(zg[i] for i in window)),
        "max_zd":     float(max(zd[i] for i in window)),
        "n_shared_book": int(((zg >= Z_THRESHOLD) & (zd >= Z_THRESHOLD)).sum()),
    }


def run(args):
    dm = DataManager()
    host = [{"verse_id": v["verse_id"], "text": dm.clean_hebrew(v["text"])}
            for v in dm.parse_verses(dm.get_text(args.host))]
    donor = [{"verse_id": v["verse_id"], "text": dm.clean_hebrew(v["text"])}
             for v in dm.parse_verses(dm.get_text(args.donor))]

    def load(book, model):
        p = os.path.join(OUT, f"{book.lower()}_{model}.scores.json")
        return {r["verse_id"]: r["perplexity_score"] for r in json.load(open(p))["records"]}

    host_gpt_pub = load(args.host, "gpt")
    host_pll = [load(args.host, "dicta")[v["verse_id"]] for v in host]
    donor_pll = [load(args.donor, "dicta")[v["verse_id"]] for v in donor]

    host_texts = [v["text"] for v in host]
    host_ids   = [v["verse_id"] for v in host]
    donor_texts = [v["text"] for v in donor]
    donor_ids   = [v["verse_id"] for v in donor]

    # Is a planted verse flagged where it came from? If Jeremiah's verses are
    # already anomalous at home, a hit after planting says nothing about the
    # join. This gives the intrinsic rate for exactly the verses we plant.
    donor_gpt = [load(args.donor, "gpt")[v["verse_id"]] for v in donor]
    donor_zg_home = rolling_z(donor_gpt)
    donor_zd_home = rolling_z(donor_pll)
    donor_shared_home = ((donor_zg_home >= Z_THRESHOLD) &
                         (donor_zd_home >= Z_THRESHOLD))
    home_idx = {v: i for i, v in enumerate(donor_ids)}

    def home_shared_at(ids):
        return bool(any(donor_shared_home[home_idx[i]] for i in ids))

    engine = SeismographEngine()
    sc = Scorer(engine)

    # The baseline: score the untouched book ourselves rather than trusting the
    # stored file, then check the two agree.
    t0 = time.time()
    base_gpt = sc.score_all(host_texts)
    published = np.array([host_gpt_pub[i] for i in host_ids])
    mine = np.array(base_gpt)
    rel = np.abs(mine - published) / np.maximum(published, 1e-9)
    print(f"[check] rescored {len(host)} verses in {time.time()-t0:.0f}s; "
          f"max relative difference from the published scores: {rel.max():.2e}")
    assert rel.max() < 1e-3, "rescoring does not reproduce the published GPT scores"

    base_zg = rolling_z(base_gpt)
    base_zd = rolling_z(host_pll)
    base_shared = int(((base_zg >= Z_THRESHOLD) & (base_zd >= Z_THRESHOLD)).sum())
    print(f"[check] shared seams in the untouched book: {base_shared}")

    if args.verify:
        # The incremental rescoring is the one thing that could silently
        # corrupt every number below. Check it the expensive way: build a
        # spliced text, score every verse of it from scratch, and compare.
        for cond, cut_it in (("plant", False), ("move", True)):
            rng = random.Random("verify|" + cond)
            pos = rng.randrange(MARGIN, len(host) - MARGIN)
            L = 12
            if cut_it:
                s0 = (pos + 400) % (len(host) - MARGIN - L - MARGIN) + MARGIN
                blk = (host_texts[s0:s0+L], host_pll[s0:s0+L], host_ids[s0:s0+L])
                cut = (s0, s0 + L)
            else:
                d0 = rng.randrange(0, len(donor) - L)
                blk = (donor_texts[d0:d0+L], donor_pll[d0:d0+L], donor_ids[d0:d0+L])
                cut = None
            texts, pll, ids, e_in, e_out = splice(
                host_texts, host_pll, host_ids, *blk, pos, cut)
            changed = set(range(e_in, min(len(texts), e_out + CONTEXT)))
            if cut is not None:
                c = cut[0] if cut[0] < pos else cut[0] + len(blk[0])
                changed |= set(range(max(0, c), min(len(texts), c + CONTEXT + 1)))
            known = list(base_gpt)
            if cut is not None:
                del known[cut[0]:cut[1]]
            known[e_in:e_in] = [float("nan")] * len(blk[0])
            fast = sc.score_all(texts, known=known, changed=changed)
            slow = [sc.score(texts, i) for i in range(len(texts))]
            d = np.abs(np.array(fast) - np.array(slow)) / np.maximum(np.array(slow), 1e-9)
            bad = int((d > 1e-6).sum())
            print(f"[verify] {cond}: {bad} of {len(texts)} verses differ between "
                  f"the incremental and the full rescoring (max {d.max():.2e})")
            assert bad == 0, "incremental rescoring is wrong"
        print("[verify] incremental rescoring reproduces a full rescoring exactly")

    n = len(host)
    rows = []
    # Every variant of the metric is arithmetic on these two series, so dump
    # them once and no model has to run again to evaluate a new variant.
    series = {"base|gpt": np.asarray(base_gpt, dtype="float64"),
              "base|pll": np.asarray(host_pll, dtype="float64"),
              "base|wc": np.asarray([len(x.split()) for x in host_texts],
                                    dtype="float64")}
    meta = []
    for L in args.lengths:
        for t in range(args.trials):
            # One generator per (condition, length, trial): a trial's draw does
            # not depend on what ran before it or on which flags were passed.
            # One generator per (length, trial). All three conditions then act
            # at the SAME position, so plant, move and null are paired rather
            # than three independent samples of the book.
            rng = random.Random(f"{args.seed}|{L}|{t}")
            pos = rng.randrange(MARGIN, n - MARGIN)
            d0 = rng.randrange(0, len(donor) - L)
            s0 = None
            for _ in range(200):
                cand = rng.randrange(MARGIN, n - MARGIN - L)
                if abs(cand - pos) >= MIN_GAP:
                    s0 = cand
                    break
            assert s0 is not None, "no donor site far enough from the target"

            for cond in ("plant", "move", "null"):
                if cond == "plant":
                    blk_t = donor_texts[d0:d0 + L]
                    blk_p = donor_pll[d0:d0 + L]
                    blk_i = donor_ids[d0:d0 + L]
                    cut = None
                elif cond == "move":
                    blk_t = host_texts[s0:s0 + L]
                    blk_p = host_pll[s0:s0 + L]
                    blk_i = host_ids[s0:s0 + L]
                    cut = (s0, s0 + L)
                else:
                    blk_t = blk_p = blk_i = []
                    cut = None

                if cond == "null":
                    texts, pll, ids = host_texts, host_pll, host_ids
                    gpt = base_gpt
                    e_in, e_out = pos, pos + L     # the same span, nothing spliced
                else:
                    texts, pll, ids, e_in, e_out = splice(
                        host_texts, host_pll, host_ids, blk_t, blk_p, blk_i, pos, cut)
                    # Only the spliced verses and the CONTEXT verses after the
                    # block can have a different causal context. With a cut,
                    # the same holds where the block was removed.
                    changed = set(range(e_in, min(len(texts), e_out + CONTEXT)))
                    if cut is not None:
                        c = cut[0] if cut[0] < pos else cut[0] + len(blk_t)
                        changed |= set(range(max(0, c), min(len(texts), c + CONTEXT + 1)))
                    known = list(base_gpt)
                    if cut is not None:
                        del known[cut[0]:cut[1]]
                    known[e_in:e_in] = [float("nan")] * len(blk_t)
                    gpt = sc.score_all(texts, known=known, changed=changed)
                    assert not any(math.isnan(x) for x in gpt), "a spliced verse went unscored"

                zg = rolling_z(gpt)
                zd = rolling_z(pll)
                win = edge_window([e_in, e_out], len(texts))
                r = evaluate(zg, zd, win)
                if cond != "null":
                    key = f"{cond}|{L}|{t}"
                    series[key + "|gpt"] = np.asarray(gpt, dtype="float64")
                    series[key + "|pll"] = np.asarray(pll, dtype="float64")
                    series[key + "|wc"] = np.asarray(
                        [len(x.split()) for x in texts], dtype="float64")
                meta.append({"key": f"{cond}|{L}|{t}", "condition": cond,
                             "length": L, "trial": t, "e_in": e_in, "e_out": e_out,
                             "n": len(texts)})
                r.update(home_flagged=(home_shared_at(blk_i) if cond == "plant" else None),
                         condition=cond, length=L, trial=t, pos=pos,
                         host_verse=host_ids[pos],
                         host_section=section_of(host_ids[pos]),
                         block_first=blk_i[0] if blk_i else "",
                         block_section=section_of(blk_i[0]) if (blk_i and cond == "move") else "",
                         n_verses=len(texts))
                rows.append(r)
        print(f"  L={L:>2} done  ({sc.calls} model calls so far)")

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    tag = args.donor.lower().replace(" ", "")
    out = os.path.join(OUT, f"planted_seams_{tag}_L{'-'.join(str(x) for x in args.lengths)}.csv")
    df.to_csv(out, index=False, encoding="utf-8")
    npz = out.replace(".csv", "_series.npz")
    np.savez_compressed(npz, **series)
    pd.DataFrame(meta).to_csv(out.replace(".csv", "_meta.csv"), index=False)
    print(f"[saved] {npz}  ({len(series)} series)")
    print(f"\n[saved] {out}   ({sc.calls} GPT-Neo calls in total)")
    report(df, base_shared)
    return df


# ----------------------------------------------------------------- report ---
def report(df, base_shared):
    print("\n" + "=" * 72)
    print("DETECTION RATE AT THE SPLICE  (shared seam within "
          f"+/-{EDGE_TOL} verses of an edge)")
    print("=" * 72)
    piv = (df.groupby(["length", "condition"])
             .agg(hit_shared=("hit_shared", "mean"),
                  hit_gpt=("hit_gpt", "mean"),
                  hit_dicta=("hit_dicta", "mean"),
                  max_cfi=("max_cfi", "mean"),
                  n=("hit_shared", "size"))
             .reset_index())
    print(piv.to_string(index=False,
                        formatters={"hit_shared": "{:.0%}".format,
                                    "hit_gpt": "{:.0%}".format,
                                    "hit_dicta": "{:.0%}".format,
                                    "max_cfi": "{:+.2f}".format}))

    print("\nplant vs null, per length (Fisher exact, one-sided):")
    try:
        from scipy import stats
        for L, g in df.groupby("length"):
            a = g[g.condition == "plant"].hit_shared
            b = g[g.condition == "null"].hit_shared
            tab = [[int(a.sum()), int((~a).sum())], [int(b.sum()), int((~b).sum())]]
            p = stats.fisher_exact(tab, alternative="greater")[1]
            print(f"  L={L:>2}  planted {a.mean():.0%} vs chance {b.mean():.0%}   p = {p:.3g}")
    except ImportError:
        print("  (scipy not installed — skipped)")

    pl = df[df.condition == "plant"]
    if len(pl):
        print("\nOf the planted blocks, the share already flagged in Jeremiah itself:")
        print(pl.groupby("length")
                .apply(lambda g: pd.Series({
                    "already flagged at home": g.home_flagged.mean(),
                    "hit here": g.hit_shared.mean(),
                    "hit here AND flagged at home": (g.hit_shared & g.home_flagged).mean()}),
                       include_groups=False)
                .to_string(float_format=lambda x: f"{x:.0%}"))

    print(f"\nShared seams in the untouched book: {base_shared}. "
          "Mean in the spliced runs:")
    print(df.groupby(["length", "condition"]).n_shared_book.mean()
            .unstack().round(1).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="Isaiah")
    ap.add_argument("--donor", default="Jeremiah")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--lengths", default="1,3,6,12,24")
    ap.add_argument("--seed", default="planted-seams-2026")
    ap.add_argument("--verify", action="store_true",
                    help="check the incremental rescoring against a full rescoring")
    a = ap.parse_args()
    a.lengths = [int(x) for x in a.lengths.split(",")]
    run(a)
