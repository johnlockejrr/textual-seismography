#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pseudo-perplexity of a verse read inside its context.

The published masked score reads one verse alone, so the same verse gets the
same number wherever it stands: it cannot witness a join, only an odd verse.
Here the model is given the preceding verses as well, each token of the target
verse is masked in turn, and only the target's tokens are scored. The verse's
own difficulty is still in the number, so the quantity of interest is the
difference from the context-free score — how much the preceding verses help.

Only the left context is used. A join breaks the fit with what precedes; the
verses after are not what a redactor's seam disturbs, and one-sided context
keeps the sequences short enough to run.
"""
import math, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

MODEL = "dicta-il/dictabert"
REVISION = "8884c6db002aba4002ee638fe4070c92e9ffbbf1"


class ContextualDicta:
    def __init__(self, left=3, max_masks=None):
        self.tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        self.model = AutoModelForMaskedLM.from_pretrained(MODEL, revision=REVISION)
        self.model.eval()
        self.left = left
        self.max_masks = max_masks
        self.calls = 0

    @torch.no_grad()
    def pll(self, texts, i):
        """Pseudo-perplexity of texts[i] given up to `left` verses before it."""
        ctx = " ".join(texts[max(0, i - self.left):i])
        tgt = texts[i]
        if not tgt.strip():
            return float("nan")
        cls, sep = self.tok.cls_token_id, self.tok.sep_token_id
        c_ids = self.tok.encode(ctx, add_special_tokens=False) if ctx else []
        t_ids = self.tok.encode(tgt, add_special_tokens=False)
        if not t_ids:
            return float("nan")
        ids = [cls] + c_ids + t_ids + [sep]
        start = 1 + len(c_ids)
        pos = list(range(start, start + len(t_ids)))
        if self.max_masks and len(pos) > self.max_masks:      # deterministic thinning
            step = len(pos) / self.max_masks
            pos = [pos[int(j * step)] for j in range(self.max_masks)]
        base = torch.tensor([ids])
        batch = base.repeat(len(pos), 1)
        targets = [batch[j, p].item() for j, p in enumerate(pos)]
        for j, p in enumerate(pos):
            batch[j, p] = self.tok.mask_token_id
        logits = self.model(input_ids=batch).logits
        lp = 0.0
        for j, p in enumerate(pos):
            probs = torch.softmax(logits[j, p], dim=-1)
            lp += math.log(probs[targets[j]].item() + 1e-12)
        self.calls += 1
        return math.exp(-lp / len(pos))
