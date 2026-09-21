# Textual Seismography

**Detecting redactional seams in the Book of Isaiah with dual-model transformer perplexity**

Agnieszka Blanka Ziemińska · Pontifical University of John Paul II, Kraków · Centre for Digital Humanities, IBL PAN, Warsaw

**→ [Interactive seismograph for Isaiah](https://agnieszkachr.github.io/textual-seismography/)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code: MIT](https://img.shields.io/badge/code-MIT-yellow.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-green.svg)](LICENSE-DATA.md)

---

## What this is

Source criticism finds the joins in a composite ancient text by reading: a word that appears
nowhere else nearby, a shift in the picture of God, a change in the rhythm of the sentences.
Those are powerful tools, but they deliver a judgement, and a judgement cannot be measured.

This project asks a different question. Not *who wrote this*, but *where does the surface break* —
and it asks two neural language models, of deliberately different architectures, to answer
independently.

- **GPT-Neo** (`Norod78/hebrew-gpt_neo-small`), a causal model, reads forwards only. Given a sliding
  window of three preceding verses, it reports how **surprised** it is by the next one. The loss is
  masked so that only the target verse's tokens count. This measures narrative disruption.
- **DictaBERT** (`dicta-il/dictabert`), a masked model, sees the whole verse at once. Each token is
  hidden in turn and the model is asked to restore it; the mean log-probability gives a
  pseudo-perplexity. This measures how **strange** the verse is in itself.

Both signals are converted to rolling Z-scores over a centred window of five verses, so each verse
is judged against its own immediate neighbourhood rather than against the book as a whole. A verse
is flagged as a **shared seam** only when *both* models independently reach Z ≥ 1.9 on it. If only
one objects, the reading is discarded as an artefact of that architecture.

The method is unsupervised and needs no labels, no dates and no commentaries.

Everything it produces has been tested against seams whose position was known in advance, by
planting passages from other books into Isaiah and asking whether the detector recovers them. That
test is what the second half of this README is about, and what
**[METHOD.md](METHOD.md)** documents in full. It is worth reading before quoting any figure from
here, because it says which figures may be read as evidence of a join and which may not.

## Results for Isaiah

The whole book as preserved in the Westminster Leningrad Codex: 66 chapters, 1,291 verses.

| | |
|---|---|
| Verses scored | 1,291 |
| GPT-Neo peaks (Z ≥ 1.9) | 77 |
| DictaBERT peaks (Z ≥ 1.9) | 126 |
| **Shared seams (both models)** | **22** |
| Pearson *r* between the two signals | 0.37 |
| Spearman *ρ* | 0.36 |

The two signals agree only weakly overall — which is the point. Where two instruments this
different stumble on exactly the same verse, something is there, and the permutation tests below
show the agreement is not accidental: 22 against 7.5 expected.

What a shared seam is, exactly: a verse that stands out from its four nearest neighbours by a wide
margin, in two independent measurements. Among them are Isa 53:8, in the Fourth Servant Song;
Isa 28:12 and 28:17; the anti-idol material at Isa 40:19 and 46:5; and Isa 63:13. That is a
verse-level claim and it stands on its own.

It is not, on the evidence of the planting experiments, a claim that a join is located there. A
rolling Z-score over five verses is bounded above by √4 = 2.0, so a threshold of 1.9 selects verses
sitting at 95% of the arithmetical maximum — isolated spikes with quiet neighbours — and a block of
inserted text is invisible to it, because its own verses become the baseline it is judged against.
Planted blocks of 1 to 24 verses were recovered at 5–28% against a chance rate of 5–15%.

**The boundary profile** on the dashboard is the second instrument, and a different one. Each verse
becomes a vector (BEREL 3.0, mean-pooled), and the height at a boundary is the angle between the
five verses before it and the five after. On the untouched book it finds both edges of the prose
narrative of chapters 36–39 — the parallel to 2 Kings 18–20, sitting inside the poetry:

| boundary | rank of 1,291 | p (rotation) | at window 10 | at window 20 |
|---|---|---|---|---|
| Isa 36:1, where the prose begins | **6** | 0.004 | 4 | 2 |
| Isa 40:1, where the prose ends | **32** | 0.024 | 34 | 39 |

Both were named before the run and both hold as the window changes, which little else in the
ranking does. The highest scores in the book fall in **Isa 3:16–4:1**, the catalogue of women's
finery — a prose list inside a poetic oracle, long treated as an insertion, and independently the
passage where BEREL's own perplexity is most extreme.

Chapter 40 therefore appears twice in these results, and in two different roles. On the verse-level
metric it is unremarkable: Isa 40:1 is in the first percentile of predictability in the whole book,
five words of doubled imperative, and the step between chapters 39 and 40 is the 13th largest of 65
with no shift of level across it (chapters 30–39 average +0.00 against +0.01 for 40–49,
Mann-Whitney p = 0.95). On the boundary profile it returns — as the end of the prose block, not as
the seam the tradition places there.

Against a list of four boundaries specified by a third party before the run (Isa 24:1, 36:1, 40:1,
56:1), the mean percentile is 75.3 where 50 is no signal, p = 0.042. The two carrying it are the
prose edges; Isa 24:1 sits at the 34th percentile and Isa 56:1 at the 70th. The instrument finds
the boundary that is a change of kind, and does not find the boundaries that are arguments.

## Where the numbers come from

**Read this before quoting any figure.**

The scores behind the published analysis are `output/isaiah_gpt.scores.json` and
`output/isaiah_dicta.scores.json`, produced on 23 February 2026. Each record already carries
`perplexity_score` and `z_score`; the file header records the parameters of the run
(`context: 3`, `window: 5`). Adding the two `z_score` columns gives the Combined Fracture Index
directly. `output/isaiah_abstract_metric.csv` is a convenience join of the two.

`output/isaiah_global_seismograph.csv` contains **two** perplexity pairs and they are not
interchangeable:

- `raw_ppl_gpt` / `raw_ppl_dicta` — identical to the February scores above.
- `disc_ppl_gpt` / `disc_ppl_dicta` — a *discounted* variant introduced by a later revision of the
  pipeline (see the `rare_ratio` column). These differ on 1,177 of the 1,291 verses. Rolling Z over
  them also yields 22 shared seams, but a **different set of verses**.

That later revision also computes global rather than rolling Z-scores, a vector-magnitude CFI,
chi-square p-values with FDR correction and PELT changepoint detection. It is a stricter, separate
analysis — not a correction of the abstract — and under it only Isa 22:6 survives as a strict
shared seam. The dashboard in `docs/` uses the February metric.

Chapter-level aggregates appear in some of these tables and carry very little. A rolling Z is a
local residual, so averaging it over a whole chapter mostly cancels: the standard deviation of the
chapter means is 0.15 against 1.66 per verse, and no chapter reaches significance on any of the
three formulations tried (mean, difference of adjacent means, density of flagged verses). They are
descriptive only.

Only the Isaiah dashboard is published. Verse-level scores for the other books the pipeline has been
run over are in `output/` as `<book>_gpt.scores.json` and `<book>_dicta.scores.json`, so any of them can
be re-scored or re-rendered from source.

## Robustness: swapping the masked model

Neither of the two models speaks Isaiah's Hebrew, so the honest test is which results survive changing
one. `output/isaiah_berel_control.csv` re-runs the pipeline with **BEREL 3.0** (`dicta-il/BEREL_3.0`), a
model trained on Rabbinic rather than Modern Hebrew, holding everything else fixed: the same GPT-Neo
scores, the same PLL code, the same rolling Z, the same threshold.

| | DictaBERT | BEREL 3.0 |
|---|---|---|
| peaks at Z ≥ 1.9 | 126 | 149 |
| shared seams | 22 | 23 |
| median pseudo-perplexity on Isaiah | 6,002 | 102 |

**11 of the 22 shared seams survive**: Isa 2:18, 13:22, 18:5, 28:12, 28:17, 30:28, 37:38, 38:6, 40:19,
46:5, 63:13. Half the signal is therefore model-dependent, and the eleven that survive both a change of
architecture and a change of training register are the ones worth defending.

The two masked models correlate at r = 0.53, while GPT-Neo and DictaBERT correlate at only r = 0.37 —
changing the training register moves the signal less than changing the architecture does.

Note that BEREL carries a confound of its own in the opposite direction. DictaBERT and GPT-Neo are
trained on web text, news and subtitles; BEREL is trained on the entirety of the Sefaria and Dicta
libraries, which contain the Tanakh and quote it throughout. On Isaiah, 99 of its verses (7.7%) score a
pseudo-perplexity below 2 and its minimum is exactly 1.00 — it has memorised part of the book, unevenly,
concentrated in the liturgically prominent chapters. Uniform memorisation would cancel in a local
baseline; uneven memorisation does not. One model does not know the language; the other knows the
answers. That argues for triangulating across models rather than substituting one for another.

Swapping the register does not remove the limits described under *validation* below: the boundary
profile there runs on BEREL, the register-appropriate model, and behaves the same way.

## Does the agreement beat chance?

The two-witness rule only means something if the two models agree more often than they would by
accident. `validate_agreement.py` measures that directly, against three nulls:

```bash
python validate_agreement.py --metric output/isaiah_abstract_metric.csv \
    --extra output/isaiah_berel_control.csv:z_berel --json output/isaiah_validation.json
```

| masked model | shared seams | chance, rotation | chance, length-matched | p |
|---|---|---|---|---|
| DictaBERT | 22 | 7.5 | 10.2 | < 5 × 10⁻⁵ |
| BEREL 3.0 | 23 | 8.9 | 11.0 | 1.5 × 10⁻⁴ |

*rotation* circularly shifts one model's series against the other, which keeps the autocorrelation the
rolling window induces and breaks only the pairing. *length-matched* permutes within verse-length
deciles, so the length-to-score relationship survives in the null and cannot account for any excess.
Neither null reached the observed count in 20,000 draws for DictaBERT; three draws in 20,000 did for
BEREL.

The confound is real but partial. Score and verse length correlate at rho = −0.20 (GPT-Neo) and −0.24
(DictaBERT), and the shared seams average 10.0 words against 12.6 elsewhere (Mann-Whitney p = 0.003).
Controlling for it raises the chance rate from 7.5 to 10.2 and leaves the excess standing.

This tests one thing only: that the agreement is not accidental. Whether what the two models agree
about is redaction is a separate question, and the next section is the experiment that asks it.

## Validation: seams whose position is known

`plant_seams.py` splices blocks from another book into Isaiah at random positions and asks whether
the detector recovers them. Three conditions act at the same position in every trial: a **planted**
block from a donor book, a **moved** block of Isaiah's own material — removed from where it came
from, so nothing is duplicated — and **nothing at all**, which gives the rate at which the rule
fires on its own. Detection is fixed in advance rather than read off a maximum, and the outcome is
threshold-free: the rank of the true edge among all positions of that text.

```bash
python plant_seams.py --donor Jeremiah --trials 40 --lengths 1,3,6,12,24
python embed_eval.py jeremiah Jeremiah
python move_controls.py
```

**The published rule does not find them.** Blocks of 1 to 24 verses, detected at 5–28% against a
chance rate of 5–15%, no length significant, for a donor of the same genre (Jeremiah) or of another
(Leviticus). Scrambling a verse's word order raises GPT-Neo's perplexity by a median factor of 6.2
and DictaBERT's by 9.0 — both models see it clearly — yet the rule that requires both to cross 1.9
flags only 13% of them. The loss is in the rule, not in the models.

**The boundary profile does, within a measured range.** On verse vectors, AUC on blocks of 12–24
verses, against an unrelated position in the untouched book:

| donor | genre | planted block | moved block of Isaiah's own material |
|---|---|---|---|
| Leviticus | legal prose | **1.000** | 0.762 |
| Jeremiah | prophetic poetry | 0.788 | 0.738 |
| Habakkuk | prophetic poetry | 0.700 | 0.712 |
| Micah | prophetic poetry | 0.612 | 0.675 |

For prose inserted into poetry this is solved: every planted block of twelve verses or more landed
in the top 1% of positions in the book, forty trials out of forty. For a donor of the same genre it
is weak, and the second column says why — a block of Isaiah moved from far away in the book scores
as high as a block of Micah inserted from outside it. What is being measured is distance in the
book's own space, not provenance.

The control that separates the readings is a **short move**: ten to twenty-five verses, wholly
inside one traditional division, crossing no recognised boundary.

| what was moved | AUC, blocks of 12 verses |
|---|---|
| 10–25 verses, inside one division | **0.500** |
| 100+ verses, inside one division | 0.675 |
| 100+ verses, across a division | 0.713 |

At short range the detector is silent — exactly chance. It does not respond to the act of cutting;
it responds to how far the material came from, and keeps responding more the further it travels.
Isaiah's internal heterogeneity, on this measure, exceeds its distance from a contemporary prophet
writing in the same forms. That is an argument for a composite text rather than against the
instrument, but it also means a detected break is a magnitude and not a verdict.

Three signals were tried and their limits measured: the level of perplexity, which tracks how
predictable the register is; the context gain, which turns out to track local lexical continuity
(moving a verse inside Isaiah costs as much as moving it after Micah); and verse vectors, which
track subject matter. [METHOD.md](METHOD.md) has the full comparison, the pre-registered selection
rule, and the confirmation that failed.

## Has the model read the text already?

A low perplexity can mean two things: the model understands the grammar, or it remembers the words.
`extraction_probe.py` separates them. Part of a verse is hidden and the model is asked to put it back;
the score is the share of hidden words returned **verbatim**.

```bash
python extraction_probe.py --metric output/isaiah_abstract_metric.csv \
    --berel-pll output/berel_pll.json --out output/extraction_probe.csv
```

Recovery over words that both tokenisers keep whole, 15 verses per set, identical masks for both models:

| model | verses | 1 word hidden | 25% hidden | 50% hidden |
|---|---|---|---|---|
| BEREL 3.0 | its own lowest-perplexity | 100% | 100% | **73.9%** |
| BEREL 3.0 | ordinary (both models at median) | 50.0% | 53.3% | 28.7% |
| DictaBERT | its own lowest-perplexity | 60.0% | 54.3% | **18.9%** |
| DictaBERT | ordinary (both models at median) | 41.7% | 8.9% | 1.2% |

BEREL returns three words in four with half the verse hidden. That is recall, not prediction: a model
working from grammar alone does not reconstruct the exact wording of a text it has not seen. On the same
verses and the same masks DictaBERT manages 11.6%.

DictaBERT is not innocent either. On the verses **it** scores as near-certain it recovers 18.9% against
1.2% on ordinary verses — a much fainter trace than BEREL's, but not nothing. Neither model is a clean
instrument; they differ in degree.

Three caveats, since the design decides the answer. The number of mask tokens tells the model how many
sub-word pieces the hidden word has, so the task is easier than open-ended generation; this applies
equally to both models. The tokenisers do not split Hebrew at the same rate (1.15 against 1.25 pieces
per word), which is why the table is restricted to words both keep whole — scoring over all words moves
BEREL's headline figure by about a point. And hiding half a verse **as one block at the end** collapses
recovery to 3–10% for every model and set: masked models need anchors on both sides, so a contiguous
gap measures the architecture rather than the memory. All four conditions are in
`output/extraction_probe.csv`.

## Reproducing

```bash
git clone https://github.com/Agnieszkachr/textual-seismography.git
cd textual-seismography
pip install -r requirements.txt
python run_analysis.py
```

The Hebrew source texts are **not** stored here. `data_manager.py` downloads them on first run from
the [OpenScriptures morphhb](https://github.com/openscriptures/morphhb) project and caches them in
`src/`.

Version pins in `requirements.txt` are strict, and the models are locked to specific revisions, so
that Z-scores and FDR-corrected p-values stay identical across future ecosystem updates:

- GPT-Neo `d9ee75ea9eb03f817c5c97a4e4d7b2e300be365b`
- DictaBERT `8884c6db002aba4002ee638fe4070c92e9ffbbf1`

Scoring 1,291 verses takes roughly an hour on CPU and a few minutes on a GPU. The engines detect
CUDA automatically. The planting experiments are cheaper than they look: a verse's masked score and
its vector travel with the verse, and only the spliced verses and the three after them change for
the causal model, so a trial costs about fifteen model calls rather than thirteen hundred.

### Offline reports

By default the generated dashboard links plotly.js from a CDN, which keeps the file small but means
it draws no charts without a network connection. Pass `--offline` to embed the library instead:

```bash
python run_analysis.py --book Isaiah --offline
```

The report grows from about 440 KB to about 4 MB and then opens from a USB stick or an air-gapped
machine. The library is taken from the installed `plotly` package; `requirements.txt` pins
plotly 5.17.0 because it bundles plotly.js 2.26.0, the version this template targets. To use a
different copy, point at it explicitly:

```bash
python run_analysis.py --book Isaiah --offline --plotly-js path/to/plotly.min.js
```

If the library cannot be found the report falls back to the CDN and says so, rather than producing
a page with no charts. Embedding a major version other than 2.x prints a warning, since the chart
code was written against 2.26.0.

## Layout

```
data_manager.py          fetch, cache and parse WLC texts; write results
seismograph_engine.py    GPT-Neo: conditional perplexity per verse
dicta_engine.py          DictaBERT: pseudo-log-likelihood per verse
contextual_pll.py        DictaBERT reading a verse inside its context
run_analysis.py          the pipeline: scoring, statistics, report
report_builder.py        builds the interactive HTML dashboard
score_book.py            score any book from scratch, both models
isolated_scores.py       causal perplexity with no context, per verse
ctx_base.py              contextual pseudo-perplexity for a whole book
embed_book.py            mean-pooled verse vectors, cached per book

validate_agreement.py    do the two models agree more than chance?
extraction_probe.py      has the model memorised the text?
plant_seams.py           splice known seams in and try to recover them
plant_ctx.py             the same, with both models reading context
embed_eval.py            the planting trials scored on verse vectors
move_controls.py         short and long moves; the negative controls
scramble_ceiling.py      the ceiling check: a verse with its words shuffled
evaluate_variants.py     one metric variant = arithmetic on stored series
sweep2.py, sweep3.py     the variant searches, scored by AUC
add_lengths.py           rebuild the word-count series of past trials
add_isolated.py          rebuild the context-free series of past trials

eval_boundaries.py       agreement with scholarly boundary hypotheses
run_sensitivity.py       parameter sweep over window and threshold
run_diagnostics.py       diagnostic runs

output/                  per-verse scores, verse vectors, derived tables
docs/                    the published site (GitHub Pages)
                         isaiah.html is self-contained: Plotly.js is bundled in,
                         so it opens from a USB stick with no network
METHOD.md                what each instrument measures, and the validation
```

## What the method cannot do

Both models were trained on **Modern Hebrew** — a revived language, not the Hebrew of Isaiah — so they
may be detecting the history of the language rather than the work of an editor. Because both share that
training, the two-model rule does not cover this bias: it guards against architectural bias, not against
one the models hold in common. The rolling local baseline is the mitigation, since the question is never
*how modern is this verse* but *how surprising is this verse, here*, and a uniform register mismatch is a
constant that a local Z-score cancels. The robustness check above shows how far that holds; the
boundary profile answers it more directly, since it runs on a model trained on the right register and
behaves the same way.

The causal model reads the canonical order, which is itself the last editor's achievement, so what is
measured is how well the final redactor smoothed his joins — not when his sources were written.
**The instrument dates nothing.** It locates boundaries in the text as we now have it; every
historical conclusion beyond that point is the philologist's work.

Three limits are measured rather than argued, and they bound what may be claimed:

- A shared seam marks a verse that is an isolated local outlier to two independent models. It is not
  evidence that a join lies there; planted joins are not recovered above chance by that rule.
- The boundary profile recovers an inserted block reliably when the insertion differs in register —
  perfectly, for prose in poetry, at twelve verses and up — and unreliably when it does not.
- A detected break is a magnitude, not a verdict. The tallest peak in Isaiah is not significant
  against a rotation null of the maximum (p = 0.35), and ranks below the top move when the window
  width changes, so no claim should rest on a boundary that does not survive that change.

The planting experiments themselves are an upper bound, not an estimate: a block is spliced in
without any rewriting at the join, and a redactor rewrites.

## Citing

See `CITATION.cff`. Code is MIT; data and results are CC BY 4.0 — see `LICENSE-DATA.md`, which also
covers the third-party Hebrew text, English renderings and model weights.
