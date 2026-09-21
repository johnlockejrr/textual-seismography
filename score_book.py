#!/usr/bin/env python3
"""Score a book the way the pipeline does, plus the context-free pass."""
import json, sys, time
from data_manager import DataManager
from seismograph_engine import SeismographEngine
from dicta_engine import DictaEngine

book = sys.argv[1]
dm = DataManager()
V = [{"verse_id": v["verse_id"], "text": dm.clean_hebrew(v["text"])}
     for v in dm.parse_verses(dm.get_text(book))]
T = [v["text"] for v in V]
se = SeismographEngine()
for tag, ctx in [("gpt", 3), ("gpt_isolated", 0)]:
    t0 = time.time()
    rec = [{"verse_id": V[i]["verse_id"],
            "perplexity_score": se._score_transition(T if ctx else [T[i]], i if ctx else 0, 3)}
           for i in range(len(V))]
    json.dump({"book": book, "model": "gpt_neo", "context": ctx, "records": rec},
              open(f"output/{book.lower()}_{tag}.scores.json", "w"))
    print(f"{book} {tag}: {len(rec)} verses, {time.time()-t0:.0f}s", flush=True)
de = DictaEngine(); t0 = time.time()
rec = [{"verse_id": v["verse_id"], "perplexity_score": de._compute_pll(v["text"])} for v in V]
json.dump({"book": book, "model": "dictabert", "records": rec},
          open(f"output/{book.lower()}_dicta.scores.json", "w"))
print(f"{book} dicta: {len(rec)} verses, {time.time()-t0:.0f}s", flush=True)
