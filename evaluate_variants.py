#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_variants.py — compare metrics on a threshold-free scale.

`plant_seams.py` stores the two raw score series for every spliced text, so a
variant of the metric is arithmetic on arrays: no model has to run again.

What a variant is, in four independent choices:

  scale     raw perplexity, or its logarithm (= cross-entropy per token)
  baseline  what a verse is compared against: a centred window of k, or the
            whole book; centred on the mean or the median; scaled by the
            standard deviation, by the MAD, or by the standard deviation of
            the window with the verse itself left out
  operator  point — one verse against its neighbourhood (the published shape)
            step  — the mean of k verses before a boundary against the k after
  combine   how the two models are put together

How a variant is scored. Not "did it fire", which throws away everything the
trial knows and depends on an arbitrary threshold, but: **how high does the
true edge rank among all positions of that text**. Each trial gives a
percentile, and the same percentile is taken at the same positions of the
untouched book, so the comparison is paired. The summary is the paired AUC —
the probability that the planted edge outranks the same place when nothing was
planted. 0.5 is a useless metric; it needs no calibration and no threshold.

The published rule is included as a special case so the continuous harness can
be checked against the binary detection rates it must reproduce.
"""
import numpy as np
import pandas as pd

Z_PUB, K_PUB, TOL = 1.9, 5, 1


# ------------------------------------------------------------- windowing ---
def _windows(x, k):
    """(n, k) view of a centred window, NaN outside the series.

    Matches pandas' rolling(k, center=True, min_periods=1) at the edges.
    """
    pad = k // 2
    p = np.full(len(x) + 2 * pad, np.nan)
    p[pad:pad + len(x)] = x
    return np.lib.stride_tricks.sliding_window_view(p, k)


def standardise(x, k, spread):
    """Z-like score of every element against its own neighbourhood."""
    x = np.asarray(x, dtype="float64")
    if k == "global":
        if spread == "median_mad":
            c = np.median(x)
            s = 1.4826 * np.median(np.abs(x - c))
        else:
            c, s = x.mean(), x.std()
        return (x - c) / s if s > 0 else np.zeros_like(x)

    W = _windows(x, k)
    with np.errstate(invalid="ignore"):
        if spread == "mean_sd":
            c = np.nanmean(W, 1); s = np.nanstd(W, 1)
        elif spread == "median_mad":
            c = np.nanmedian(W, 1)
            s = 1.4826 * np.nanmedian(np.abs(W - c[:, None]), 1)
        elif spread == "loo_mean_sd":
            V = W.copy(); V[:, k // 2] = np.nan       # judge it by its neighbours only
            c = np.nanmean(V, 1); s = np.nanstd(V, 1)
        else:
            raise ValueError(spread)
    out = np.zeros_like(x)
    ok = s > 0
    out[ok] = (x[ok] - c[ok]) / s[ok]
    return out


def step_score(x, k):
    """|mean of the k verses before a boundary - mean of the k after|.

    Returned per verse index i, meaning the boundary in front of verse i.
    """
    x = np.asarray(x, dtype="float64")
    n = len(x)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(n)
    a0 = np.maximum(0, i - k); a1 = i
    b0 = i; b1 = np.minimum(n, i + k)
    na = a1 - a0; nb = b1 - b0
    before = np.where(na > 0, (cs[a1] - cs[a0]) / np.maximum(na, 1), 0.0)
    after = np.where(nb > 0, (cs[b1] - cs[b0]) / np.maximum(nb, 1), 0.0)
    out = np.abs(before - after)
    out[(na < 2) | (nb < 2)] = 0.0
    return out


def hotelling_step(F, k):
    """Two-sample Hotelling T^2 across each boundary, on the joint features.

    Combining two separately standardised scalars with a fixed rule throws away
    the direction in which the two models differ together. T^2 measures the gap
    between the mean vectors of the k verses before and the k after in units of
    the book's own covariance, so the informative direction does not have to be
    guessed — and no labels are needed for it.
    """
    F = np.asarray(F, dtype="float64")
    n, p = F.shape
    S = np.cov(F.T) + 1e-9 * np.eye(p)
    Si = np.linalg.inv(S)
    cs = np.vstack([np.zeros(p), np.cumsum(F, axis=0)])
    out = np.zeros(n)
    for i in range(n):
        a0, a1 = max(0, i - k), i
        b0, b1 = i, min(n, i + k)
        na, nb = a1 - a0, b1 - b0
        if na < 2 or nb < 2:
            continue
        d = (cs[a1] - cs[a0]) / na - (cs[b1] - cs[b0]) / nb
        out[i] = (na * nb / (na + nb)) * float(d @ Si @ d)
    return out


def onesided_step(x, k):
    """The rise just after a boundary, relative to the level just before it.

    A join is asymmetric: what follows is poorly predicted from what precedes,
    not the other way round. The absolute difference mixes both directions.
    """
    x = np.asarray(x, dtype="float64")
    n = len(x)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(n)
    a0, a1 = np.maximum(0, i - k), i
    b0, b1 = i, np.minimum(n, i + k)
    na, nb = a1 - a0, b1 - b0
    before = np.where(na > 0, (cs[a1] - cs[a0]) / np.maximum(na, 1), 0.0)
    after = np.where(nb > 0, (cs[b1] - cs[b0]) / np.maximum(nb, 1), 0.0)
    out = after - before
    out[(na < 2) | (nb < 2)] = 0.0
    return out


def mannwhitney_step(x, k):
    """Rank-based version: the k values before against the k after.

    Catches a change in the shape of the distribution when the means agree.
    """
    x = np.asarray(x, dtype="float64")
    n = len(x)
    out = np.zeros(n)
    for i in range(n):
        a, b = x[max(0, i - k):i], x[i:i + k]
        if len(a) < 3 or len(b) < 3:
            continue
        u = (a[:, None] < b[None, :]).sum() + 0.5 * (a[:, None] == b[None, :]).sum()
        out[i] = abs(u / (len(a) * len(b)) - 0.5) * 2
    return out


# ------------------------------------------------------------- combining ---
def combine(zg, zd, how):
    if how == "sum":      return zg + zd                      # the CFI of the paper
    if how == "min":      return np.minimum(zg, zd)           # a soft two-witness rule
    if how == "max":      return np.maximum(zg, zd)
    if how == "geo":
        a, b = np.clip(zg, 0, None), np.clip(zd, 0, None)
        return np.sqrt(a * b)
    if how == "diff":     return zg - zd   # context misfit with intrinsic difficulty taken out
    if how == "gpt":      return zg
    if how == "dicta":    return zd
    raise ValueError(how)


def residualise(y, wc):
    """Take out of `y` everything a straight line in log verse length explains.

    Verse length is the confound: a longer verse is easier or harder to predict
    for reasons that have nothing to do with where it came from. After this,
    a detection cannot be attributed to length, by construction.
    """
    x = np.log(np.maximum(np.asarray(wc, dtype="float64"), 1.0))
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def score_series(gpt, pll, v, wc=None, iso=None):
    """The full per-position score of one spliced text under variant `v`."""
    if v.get("signal") == "length":          # the no-model reference
        g = d = np.log(np.maximum(np.asarray(wc, dtype="float64"), 1.0))
    elif v.get("signal") == "delta":
        # How much the preceding verses help. The verse's own difficulty is in
        # both terms and cancels, so what is left is fit to the context alone —
        # which is the only thing a join can break.
        if iso is None:
            raise ValueError("the context-gain signal needs the isolated scores")
        g = np.log(np.asarray(iso, dtype="float64")) - np.log(np.asarray(gpt, dtype="float64"))
        d = np.log(np.asarray(pll, dtype="float64"))
    else:
        g = np.log(gpt) if v["scale"] == "log" else np.asarray(gpt, dtype="float64")
        d = np.log(pll) if v["scale"] == "log" else np.asarray(pll, dtype="float64")
    if v.get("signal") == "delta_only":
        g = d = np.log(np.asarray(iso, dtype="float64")) - np.log(np.asarray(gpt, dtype="float64"))
    if v.get("residual"):
        if wc is None:
            raise ValueError("residualising needs the word counts")
        g, d = residualise(g, wc), residualise(d, wc)
    if v["operator"] == "hotelling":
        t2 = hotelling_step(np.column_stack([g, d]), v["step_k"])
        return standardise(t2, v["window"], v["spread"])
    fn = {"point": None, "step": step_score,
          "onesided": onesided_step, "mw": mannwhitney_step}[v["operator"]]
    if fn is None:
        zg = standardise(g, v["window"], v["spread"])
        zd = standardise(d, v["window"], v["spread"])
    else:
        zg = standardise(fn(g, v["step_k"]), v["window"], v["spread"])
        zd = standardise(fn(d, v["step_k"]), v["window"], v["spread"])
    return combine(zg, zd, v["combine"])


# --------------------------------------------------------------- scoring ---
def edge_positions(e_in, e_out, n, tol=TOL):
    idx = set()
    for e in (e_in, e_out):
        for dd in range(-tol, tol + 1):
            if 0 <= e + dd < n:
                idx.add(e + dd)
    return np.fromiter(sorted(idx), dtype=int)


def percentile_of_edge(score, pos):
    """Where the best position at the seam sits among every position."""
    best = np.nanmax(score[pos])
    return float((score <= best).mean() * 100.0)


def evaluate(npz, meta, variant, condition="plant", wcnpz=None, control="random", isonpz=None):
    """Percentile of the true edge, against a control place in the untouched book.

    control="random" draws an unrelated position, which is the conservative
    choice; control="same" reuses the trial's own position, which is paired and
    therefore less noisy but easier to mistake for an effect.
    """
    base_g, base_p = npz["base|gpt"], npz["base|pll"]
    base_wc = wcnpz["base|wc"] if wcnpz is not None else None
    base_iso = isonpz["base|iso"] if isonpz is not None else None
    base_score = score_series(base_g, base_p, variant, base_wc, base_iso)
    rows = []
    for m in meta[meta.condition == condition].itertuples():
        key = f"{m.condition}|{m.length}|{m.trial}"
        g, p = npz[key + "|gpt"], npz[key + "|pll"]
        wc = wcnpz[key + "|wc"] if wcnpz is not None else None
        it = isonpz[key + "|iso"] if isonpz is not None else None
        s = score_series(g, p, variant, wc, it)
        pos = edge_positions(m.e_in, m.e_out, m.n)
        if control == "same":
            ctrl = edge_positions(m.e_in, m.e_out, len(base_score))
        else:
            r = np.random.default_rng(abs(hash((m.length, m.trial))) % 2**32)
            q = int(r.integers(25, len(base_score) - 25 - m.length))
            ctrl = edge_positions(q, q + m.length, len(base_score))
        rows.append({"length": m.length, "trial": m.trial,
                     "pct_spliced": percentile_of_edge(s, pos),
                     "pct_untouched": percentile_of_edge(base_score, ctrl)})
    return pd.DataFrame(rows)


def auc(df):
    """P(the seam outranks the same place when nothing was planted)."""
    a, b = df.pct_spliced.to_numpy(), df.pct_untouched.to_numpy()
    return float((a > b).mean() + 0.5 * (a == b).mean())


def summarise(df):
    out = {"n": len(df), "auc": auc(df),
           "mean_pct_spliced": df.pct_spliced.mean(),
           "mean_pct_untouched": df.pct_untouched.mean()}
    try:
        from scipy import stats
        d = df.pct_spliced - df.pct_untouched
        out["wilcoxon_p"] = float(stats.wilcoxon(d).pvalue) if d.abs().sum() > 0 else 1.0
    except Exception:
        out["wilcoxon_p"] = np.nan
    return out


# ------------------------------------------------- the published rule ------
def published_hit_rate(npz, meta, condition):
    """The binary rule of the paper, for checking this file against the runs."""
    base_g, base_p = npz["base|gpt"], npz["base|pll"]
    hits = []
    for m in meta[meta.condition == condition].itertuples():
        if condition == "null":
            g, p = base_g, base_p
        else:
            key = f"{m.condition}|{m.length}|{m.trial}"
            g, p = npz[key + "|gpt"], npz[key + "|pll"]
        zg = standardise(g, K_PUB, "mean_sd")
        zd = standardise(p, K_PUB, "mean_sd")
        pos = edge_positions(m.e_in, m.e_out, len(g))
        hits.append(bool(np.any((zg[pos] >= Z_PUB) & (zd[pos] >= Z_PUB))))
    return pd.DataFrame({"length": meta[meta.condition == condition].length.to_numpy(),
                         "hit": hits}).groupby("length").hit.mean()


def load_iso(stem):
    import os
    p = stem + "_iso.npz"
    return np.load(p) if os.path.isfile(p) else None


def load_wc(stem):
    """Word counts: the run's own if it stored them, else the rebuilt file.

    A rebuild depends on the seed, so a run that recorded its own series must
    always win — a rebuild with the wrong seed still has the right lengths and
    would pass unnoticed.
    """
    import os
    main = np.load(stem + "_series.npz")
    if "base|wc" in main.files:
        return main
    p = stem + "_wc.npz"
    return np.load(p) if os.path.isfile(p) else None


def load(stem):
    # keep_default_na=False: the condition named "null" is a string, and pandas
    # turns it into NaN by default, which silently empties that whole group.
    return (np.load(stem + "_series.npz"),
            pd.read_csv(stem + "_meta.csv", keep_default_na=False, na_values=[""]))
