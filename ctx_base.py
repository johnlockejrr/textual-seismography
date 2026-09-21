#!/usr/bin/env python3
"""Contextual pseudo-perplexity for a whole book, cached once."""
import json, sys, time
from data_manager import DataManager
from contextual_pll import ContextualDicta
book = sys.argv[1] if len(sys.argv) > 1 else "Isaiah"
dm = DataManager()
V = [{"verse_id": v["verse_id"], "text": dm.clean_hebrew(v["text"])}
     for v in dm.parse_verses(dm.get_text(book))]
T = [v["text"] for v in V]
c = ContextualDicta(left=3)
t0 = time.time(); rec = []
for i, v in enumerate(V):
    rec.append({"verse_id": v["verse_id"], "perplexity_score": c.pll(T, i)})
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(V)}  {(time.time()-t0)/(i+1):.2f}s/verse", flush=True)
json.dump({"book": book, "model": "dictabert", "context": 3, "records": rec},
          open(f"output/{book.lower()}_dicta_context.scores.json", "w"))
print(f"{book}: {len(rec)} verses in {time.time()-t0:.0f}s")
