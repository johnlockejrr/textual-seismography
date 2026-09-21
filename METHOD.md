# What this detector measures, and what it cannot

Two instruments live in this repository. The first is the one the paper
reports. The second was built after the first was measured, and the measuring
is the part worth reading.

## v1 — the published metric

GPT-Neo scores each verse against the three before it, DictaBERT scores each
verse on its own, both are turned into a rolling Z over a centred window of
five, and a verse flagged by both at Z ≥ 1.9 is called a shared seam. Isaiah
yields 22 of them, and they agree far more often than chance: 22 against 7.5
expected under circular rotation, p < 5e-5.

**What it detects is an isolated verse, not a join.** Three findings, two of
them arithmetic:

The rolling Z of a window of *k* cannot exceed √(k−1). With k = 5 the ceiling
is exactly 2.0, and the largest values in Isaiah are 1.9996 (GPT-Neo) and
1.99996 (DictaBERT). A threshold of 1.9 therefore selects verses at 95% of the
mathematical maximum. To reach it a verse must exceed the largest of its four
neighbours by a median factor of 2.9, with those four similar to each other.

A planted block is invisible by construction, because its verses are each
other's baseline. Blocks of 1 to 24 verses spliced into Isaiah were detected at
5–28% against a chance rate of 5–15%; no length significant, for a donor of the
same genre or of another.

The consensus rule, not the models, is what loses the signal. A verse whose
words are shuffled into ungrammatical Hebrew raises GPT-Neo perplexity by a
median factor of 6.2 and DictaBERT by 9.0. GPT-Neo flags 63% of them,
DictaBERT 27%, and the rule that requires both flags 13%.

None of this touches the 22 seams or the permutation result. What it removes is
the step from "both models flag this verse" to "there is a join here".

## v2 — a statistic shaped like a join

Log scores (cross-entropy per token) with verse length regressed out, the
context gain of the causal model — its score for a verse alone minus its score
for the same verse read after its neighbours — paired with DictaBERT's
context-free score, compared across each boundary by a rank test over ten
verses on each side, standardised over a window of 21 by median and MAD, the
two channels combined geometrically.

Every element answers a measured failure of v1: the log scale for the heavy
tail, the residual for verse length, the context gain to cancel a verse's own
difficulty, the rank test for a change of distribution without a change of
mean, the two-sided window because a join is a step and not a spike.

## What v2 does, on four donors

Blocks spliced into Isaiah at random positions, 40 trials per length. Detection
is the percentile of the true edge among all positions of that text, against an
unrelated position in the untouched book; the summary is AUC, where 0.5 is a
statistic that ignores the splice. The variant was fixed in advance of the last
two columns.

| donor | genre | AUC, blocks 12–24 | verse length alone | same-book move |
|---|---|---|---|---|
| Leviticus | legal prose | **0.875** | 0.562 | 0.592 |
| Jeremiah | prophetic poetry | 0.700 | 0.562 | 0.600 |
| Habakkuk | prophetic poetry | 0.725 | 0.475 | 0.610 |
| Micah | prophetic poetry | 0.613 | 0.625 | 0.605 |

On legal prose it works: the seam lands in the top 1% of positions in a third
of trials, against 4% for the control. On prophetic poetry — the case the
redaction hypothesis for Isaiah actually requires — it ranges from 0.61 to
0.73, and on Micah word counts alone do better than the two transformers.
Moving a block of Isaiah's own material registers at 0.59–0.61, so part of what
is being detected is the act of splicing rather than the material.

## Why, and where the limit is

The detector performs at roughly the level the signal permits. Asking a simple
classifier how well the two per-verse scores separate a donor's verses from
Isaiah's:

| donor | two model scores | verse length alone | detector on long blocks |
|---|---|---|---|
| Leviticus | 0.842 | 0.539 | 0.875 |
| Habakkuk | 0.663 | 0.571 | 0.725 |
| Jeremiah | 0.637 | 0.645 | 0.700 |
| Micah | 0.540 | 0.430 | 0.613 |

The two columns track each other. A verse of Micah is very nearly
indistinguishable from a verse of Isaiah to these two models, so no arrangement
of their output can find a Micah insertion, and for Jeremiah the scores add
nothing to what verse length already says.

**The limit is the signal, not the statistic.** Two scalar perplexities from a
modern-Hebrew causal model and a modern-Hebrew masked model separate genres
well and authors within a genre barely. A detector for the seams the field
argues about needs a richer per-verse representation — contextual
pseudo-likelihood, or embeddings — not a better rule over these two numbers.


## The context gain, and why it does not rescue the method

The obvious next signal was to make the masked model read context too — give
DictaBERT the three preceding verses, mask the target verse's tokens in turn,
and take the difference from its context-free score. Both models then measure
the same thing, *how much the past helps*, with the verse's own difficulty
cancelling in the difference. The causal model's version of this was already
tested above; it is what `signal = delta` computes.

Contextual pseudo-perplexity for the whole of Isaiah takes 38 minutes at about
1.7 seconds a verse, so it is affordable. What it measures, before any detector
is built on it, is not:

| the verse is read after… | DictaBERT, median log change | got worse | GPT-Neo | got worse |
|---|---|---|---|---|
| three verses of Micah instead of its own | +0.187 | 68% | +0.074 | 68% |
| its own verses instead of Micah's (reverse) | +0.224 | 76% | +0.108 | 76% |
| **another passage of Isaiah itself** | **+0.195** | **68%** | **+0.384** | **84%** |

The control is as large as the effect, and for the causal model three times
larger. Putting a verse after a different passage of the same book damages its
predictability as much as putting it after a different book. What these models
draw from context is local lexical and topical continuity, not authorial
register — so the context gain cannot tell a redactional seam from any
discontinuity, and a detector built on it would be measuring the wrong thing
however it were shaped.

This closes the scalar line of attack. The remaining option is a richer
per-verse representation — embeddings, where a verse is a vector and a boundary
is a change of direction rather than of level.


## v3 — verse vectors, and what finally settles the question

The last signal left was to stop reducing a verse to one number. BEREL 3.0
turns each verse into a 768-dimensional vector (mean-pooled last layer, special
tokens excluded); a boundary is then a change of direction rather than of
level, scored as the angle between the mean vectors of the five verses before
and the five after. A vector belongs to its verse, so the planting trials
already on disk were re-scored without running a model again.

AUC on blocks of 12–24 verses, against an unrelated position in the untouched
book, with the same trials as everything above:

| donor | genre | planted block | **moved block of Isaiah's own material** |
|---|---|---|---|
| Leviticus | legal prose | **1.000** | 0.762 |
| Jeremiah | prophetic poetry | 0.788 | 0.738 |
| Habakkuk | prophetic poetry | 0.700 | 0.712 |
| Micah | prophetic poetry | 0.612 | 0.675 |

For legal prose this is a solved problem: every planted block of twelve verses
or more lands in the top 1% of all positions in the book, forty trials out of
forty. No scalar variant came near that.

For prophetic poetry the second column is the finding. A block of Isaiah moved
from one part of the book to another is detected **as well as or better than** a
block of Micah or Habakkuk inserted from outside. The statistic is not
measuring where the material came from; it is measuring how much the subject
matter changes across the boundary — and Micah, a contemporary treating the
same themes in the same forms, disturbs Isaiah less than Isaiah's own chapters
disturb each other.


## The move control, corrected

The first design treated a moved block of Isaiah as a false alarm. That was
wrong, and the objection is philological: ancient composition was
conventional, and a block taken from a hundred verses away in a book whose
composite character is not disputed is very likely carrying material across a
real boundary. Detecting it is a second detection, not an error. Two controls
separate the readings, both on verse vectors, both free of model runs.

**A short move, inside one division.** Ten to twenty-five verses, no
recognised boundary crossed — the control the design actually needed:

| what was moved | AUC, blocks of 12 verses |
|---|---|
| 10–25 verses, inside one division | **0.500** |
| 100+ verses, inside one division | 0.675 |
| 100+ verses, across a division | 0.713 |

At a short distance the detector is **silent**: exactly chance, the seam in the
top 1% of positions in 5% of trials against 7% for the control. It does not
respond to the act of cutting. What it responds to is how far the material
came from, and the response keeps growing with that distance — across four
bands of distance the AUC runs 0.656, 0.627, 0.750, 0.783, with the seam in the
top 1% of the book in 10%, 7%, 13% and 28% of trials.

**Split by the traditional divisions.** Pooled over three independent sets of
trials, long moves crossing 1-39 / 40-55 / 56-66 score 0.724 against 0.670 for
long moves inside one division, with the seam in the top 1% twice as often (18%
against 8%). The direction is the expected one; the difference on the
continuous measure is not significant (p = 0.42), so the divisions are not
where the heterogeneity stops.

**What follows.** A detected break is not a verdict but a magnitude: these two
neighbourhoods are unusually far apart in the book's own space. And the
comparison that matters is this one — a block of Isaiah moved from far away in
the book is detected **more** readily (0.783 in the farthest band) than a block
of Micah inserted from outside it (0.613). Isaiah's internal heterogeneity
exceeds its distance from a contemporary prophet writing in the same forms.
That is an argument for a composite text, not against the instrument.

## What the whole investigation shows

Three different signals, three different failures, one cause:

| signal | what it turns out to measure | evidence |
|---|---|---|
| perplexity level | how predictable the register is | prose 0.875, same-genre poetry 0.61–0.73; verse length alone matches it on two donors |
| context gain | local lexical continuity | moving a verse inside Isaiah costs as much as moving it after Micah (+0.195 against +0.187) |
| verse vectors | distance in the book's own space | a short move inside one division is not detected at all; a long one is, and the further the material travels the more so |

What they measure is register and subject matter. A redactional seam is
visible to them exactly in so far as it coincides with a change in one of
those — which is a claim about when the instrument may be believed, not about
what it fails to be.

What remains true and useful: a validated detector of register discontinuity,
perfect on prose inserted into poetry at twelve verses and up; a verse-level
anomaly detector (v1) whose 22 agreements beat chance and whose meaning is
"isolated local outlier", not "join"; and a test bed on which any future
proposal — a Hebrew model trained on the right register, a fine-tuned
authorship representation — can be measured in an afternoon instead of argued
about.

## How the variant was chosen, and what did not confirm

The search was over a thousand variants of scale, baseline, window, operator
and combination, run on a development set (donor Jeremiah) with the choice of
winner and the success criterion written down before any reserve was looked at.
Two reserve sets followed: Leviticus, and Habakkuk with fresh positions. The
candidate above was reached after those reserves had been consulted more than
once, so a fourth donor, Micah, was scored from scratch for a single
confirming measurement. **It did not confirm**: 0.613 against a pre-registered
bar of 0.65, with the no-model baseline at 0.625.

The honest status is therefore: v2 is a real improvement on v1 and a working
detector of register discontinuity, and it is not a validated detector of
same-genre insertion.

## Caveats that apply to every figure above

Planted seams are unsmoothed — a block is inserted without any rewriting at the
join, while a redactor rewrites — so these are an upper bound on performance
against real redaction, not an estimate of it. Habakkuk has 56 verses, so at
block length 24 its forty trials draw from twenty start points and overlap by
56%. And nothing here establishes anything about the boundaries of the real
book: tested against a list specified by someone else before the run
(Isa 24:1, 36:1, 40:1, 56:1), v2's mean percentile is 42.0 where 50 is no
signal, and the top of its ranking moves substantially with the window width.

## Reproducing

```bash
python score_book.py Micah                       # any donor, from scratch
python isolated_scores.py                        # the context-free pass
python plant_seams.py --donor Micah --trials 40 --lengths 1,3,6,12,24 \
       --seed confirm-2026-c
python add_isolated.py output/planted_seams_micah_L1-3-6-12-24 Micah confirm-2026-c
python sweep3.py output/planted_seams_<donor>_<lengths>
```
