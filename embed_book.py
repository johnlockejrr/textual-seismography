#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mean-pooled verse vectors, cached per book and model.

A verse's vector does not depend on where the verse stands, so it travels with
the verse into any splice: the planting experiments already on disk can be
re-scored on vectors without running a model again.

Special tokens are excluded from the pooling. Including [CLS] and [SEP] adds a
constant to every vector and shrinks the angles between them, which is the
quantity the boundary statistic is built on.
"""
import os, sys
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
from data_manager import DataManager

MODELS = {"berel": ("dicta-il/BEREL_3.0", None),
          "dicta": ("dicta-il/dictabert", "8884c6db002aba4002ee638fe4070c92e9ffbbf1"),
          "msbert": ("dicta-il/MsBERT", None)}


@torch.no_grad()
def embed(texts, tag, batch=16):
    name, rev = MODELS[tag]
    tok = AutoTokenizer.from_pretrained(name, revision=rev)
    model = AutoModel.from_pretrained(name, revision=rev).eval()
    special = {tok.cls_token_id, tok.sep_token_id, tok.pad_token_id}
    out = []
    for s in range(0, len(texts), batch):
        enc = tok(texts[s:s+batch], padding=True, truncation=True,
                  max_length=256, return_tensors="pt")
        h = model(**enc).last_hidden_state
        keep = enc["attention_mask"].clone()
        for sp in special:
            if sp is not None:
                keep[enc["input_ids"] == sp] = 0
        m = keep.unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).numpy())
    return np.vstack(out)


if __name__ == "__main__":
    tag = sys.argv[1]
    dm = DataManager()
    for book in sys.argv[2:]:
        p = f"output/emb_{book.lower()}_{tag}.npy"
        if os.path.isfile(p):
            print(f"{book} {tag}: cached"); continue
        T = [dm.clean_hebrew(v["text"]) for v in dm.parse_verses(dm.get_text(book))]
        E = embed(T, tag)
        np.save(p, E)
        print(f"{book} {tag}: {E.shape[0]} verses, dim {E.shape[1]}", flush=True)
