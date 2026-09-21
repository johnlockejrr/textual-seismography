#!/usr/bin/env python3
"""Perplexity of every verse with NO preceding context.

The contextual score already computed mixes two things: how hard the verse is
in itself, and how well it follows what comes before. Scoring the same verse
alone isolates the first, so the difference isolates the second — which is the
only part a join can change. The isolated score is a property of the verse, so
it is computed once per book and travels with the verse wherever it is spliced.
"""
import json, sys, time
from data_manager import DataManager
from seismograph_engine import SeismographEngine

books = sys.argv[1:] or ["Isaiah", "Jeremiah", "Leviticus", "Habakkuk"]
dm, se = DataManager(), SeismographEngine()
for b in books:
    verses = [{"verse_id": v["verse_id"], "text": dm.clean_hebrew(v["text"])}
              for v in dm.parse_verses(dm.get_text(b))]
    t0 = time.time(); rec = []
    for i, v in enumerate(verses):
        rec.append({"verse_id": v["verse_id"],
                    "perplexity_score": se._score_transition([v["text"]], 0, 3)})
        if (i + 1) % 300 == 0:
            print(f"  {b} {i+1}/{len(verses)}", flush=True)
    json.dump({"book": b, "model": "gpt_neo", "context": 0, "records": rec},
              open(f"output/{b.lower()}_gpt_isolated.scores.json", "w"))
    print(f"{b}: {len(rec)} verses in {time.time()-t0:.0f}s", flush=True)
