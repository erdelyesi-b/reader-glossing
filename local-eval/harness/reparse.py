#!/usr/bin/env python3
"""Re-parse a recorded run's raw text with the current extractor.

Runs store the raw response precisely so parsing can be revisited without
paying for generation again — a 26B streaming off SSD costs minutes per task.

    python3 reparse.py ../runs/turbo-gemma4-26b.jsonl
"""

import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_model import extract_json  # noqa: E402

for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    before = collections.Counter(r.get("parse") for r in rows)
    for r in rows:
        r["entries"], r["parse"] = extract_json(r.get("raw") or "")
    after = collections.Counter(r.get("parse") for r in rows)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("%s\n  before %s\n  after  %s" % (
        os.path.basename(path), dict(before), dict(after)))
