#!/usr/bin/env python3
"""What would sub-task B score if the lexicon wrote the citation form?

The bench measures models end-to-end, but the design question is narrower: is
it better to let the model write the citation form, or to have it emit a bare
lemma and let the lexicon inflate that into the term string?

This answers it without running a model, by replaying the gold set: take each
gold entry's headword, ask the lexicon for a citation form, and run the result
through the same form_check score.py uses. Comparing that against the gold
entries' own B score isolates the lexicon's contribution from any model's.

    python3 hybrid_projection.py
"""

import collections
import json
import os
import sqlite3
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from score import head, classify, form_check, load_db  # noqa: E402
from lexicon_coverage import lookup                    # noqa: E402


def main():
    lex = sqlite3.connect(os.path.abspath(
        os.path.join(HERE, "..", "models", "lexicon", "lexicon.db")))
    genders, _ = load_db(os.path.join(ROOT, "glossary.db"))
    tasks = [json.loads(l) for l in
             open(os.path.join(HERE, "..", "goldset", "tasks.jsonl"), encoding="utf-8")]

    s = collections.Counter()
    gold_bad = collections.Counter()
    hyb_bad = collections.Counter()

    for t in tasks:
        for entries in t["gold"].values():
            for e in entries:
                gold_term = unicodedata.normalize("NFC", e[0])
                kind = classify(gold_term)
                s["n"] += 1

                gb = form_check(gold_term, genders)
                if not gb:
                    s["gold_clean"] += 1
                for b in gb:
                    gold_bad[b] += 1

                # The hybrid: model supplies only the headword, lexicon inflates.
                row = lookup(lex, head(gold_term), kind) if kind in ("noun", "verb") else None
                hyb_term = row[1] if row else gold_term   # fall back to the model
                if row:
                    s["lexicon_supplied"] += 1
                else:
                    s["model_fallback"] += 1
                hb = form_check(unicodedata.normalize("NFC", hyb_term), genders)
                if not hb:
                    s["hybrid_clean"] += 1
                for b in hb:
                    hyb_bad[b] += 1

    n = s["n"] or 1
    print("gold entries replayed: %d" % n)
    print("  citation form supplied by lexicon: %d (%.0f%%), model fallback %d" % (
        s["lexicon_supplied"], 100.0 * s["lexicon_supplied"] / n, s["model_fallback"]))
    print()
    print("  B:form, model writes the term  : %5.1f%%" % (100.0 * s["gold_clean"] / n))
    print("  B:form, lexicon writes the term: %5.1f%%" % (100.0 * s["hybrid_clean"] / n))
    print()
    print("  model-written failures : %s" % dict(gold_bad.most_common(5)))
    print("  lexicon-written failures: %s" % dict(hyb_bad.most_common(5)))
    lex.close()


if __name__ == "__main__":
    main()
