#!/usr/bin/env python3
"""Emit the gold set in run format, so it can be scored like any model.

This is the bench's calibration. Selection precision and recall come out at
100% by construction, which is the point — it proves the matcher pairs entries
correctly. What is *not* 100% is the citation-form column, and that number is
the honest ceiling: it is what a frontier model with the full README actually
achieved, so a local model is measured against a real target rather than a
perfect one.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

tasks = [json.loads(l) for l in
         open(os.path.join(HERE, "..", "goldset", "tasks.jsonl"), encoding="utf-8")]

out = os.path.join(HERE, "..", "runs", "GOLD-frontier.jsonl")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    for t in tasks:
        fh.write(json.dumps({
            "id": t["id"], "book": t["book"], "chapter": t["chapter"],
            "seconds": 0.0, "usage": {}, "parse": "clean", "error": None,
            "raw": "", "entries": t["gold"],
        }, ensure_ascii=False) + "\n")
print("wrote %s (%d tasks)" % (os.path.abspath(out), len(tasks)))
