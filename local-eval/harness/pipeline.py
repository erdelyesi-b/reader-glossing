#!/usr/bin/env python3
"""The staged pipeline: how much of an entry needs a model at all?

The monolithic run failed because one model was asked to do word selection,
German morphology, a German definition, a Hungarian translation and JSON
syntax at once — and it broke on every axis, most damagingly by inventing
Hungarian words 31.6% of the time.

Each field has a different best source, so each gets its own stage:

    term  <- dictionary.sqlite gender + Wiktionary plural / principal parts
    de    <- Wiktionary gloss, else the model
    hu    <- dictionary.sqlite senses, else the model
    JSON  <- assembled in Python, never generated

Everything a lookup can answer is answered by a lookup. What reaches the model
is only what genuinely needs judgement, and the Hungarian it does write is
gated by hunspell, so an invented word is caught rather than shipped.

    python3 pipeline.py            # coverage of the deterministic layer
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
from score import head, classify, hu_tokens, form_check, load_db  # noqa: E402
from stage_form import Former                                     # noqa: E402
from stage_hu import Dictionary                                   # noqa: E402

LEX = os.path.join(HERE, "..", "models", "lexicon", "lexicon.db")


class Pipeline:
    def __init__(self, use_corpus=True):
        self.use_corpus = use_corpus
        self.former = Former()
        self.dict = Dictionary()
        self.lex = sqlite3.connect(os.path.abspath(LEX))
        # The corpus itself is the best source of all: 50k entries already in
        # the house style, and make_batches.py has usually resolved them for
        # free before the model ever sees the batch.
        self.corpus = {}
        cdb = sqlite3.connect(os.path.join(ROOT, "glossary.db"))
        for lemma, term, de, hu in cdb.execute(
                "SELECT lemma, term, de, hu FROM entries ORDER BY freq ASC"):
            key = lemma.split(":", 1)[1] if ":" in lemma else lemma
            self.corpus[key] = (term, de, hu)
        cdb.close()

    def build(self, word, kind=None):
        """Return (entry, sources) with None for any field a lookup can't fill."""
        w = head(word)
        src = {}

        got = self.corpus.get(w) if self.use_corpus else None
        if got:
            return list(got), {"term": "corpus", "de": "corpus", "hu": "corpus"}

        # term: gender from the de-hu dictionary, plural/parts from Wiktionary
        term = None
        f = self.former.form(w, kind)
        if f:
            term, src["term"] = f[0], f[1]
        else:
            look = self.dict.look(w)
            if look and look[0]:
                term, src["term"] = "%s %s" % (look[0], w.capitalize()), "dict-gender"

        # de: Wiktionary's first gloss. Encyclopedic rather than learner-style,
        # so it is a fallback the model should usually rewrite, not a finished
        # field — recorded separately for that reason.
        de = None
        row = self.lex.execute(
            "SELECT de FROM lemma WHERE word = ? AND de <> '' LIMIT 1", (w,)).fetchone()
        if row:
            de, src["de"] = row[0], "wiktionary"

        # hu: dictionary senses, first one until a model ranks them
        hu = None
        look = self.dict.look(w)
        if look and look[1]:
            hu, src["hu"] = ", ".join(look[1][:2]), look[2]

        return [term, de, hu], src


def main():
    # make_batches.py already resolves everything glossary.db knows before the
    # model is involved, so the pipeline only ever meets words the corpus does
    # NOT have. Measuring with the corpus on would just re-find the gold.
    use_corpus = "--with-corpus" in sys.argv
    p = Pipeline(use_corpus=use_corpus)
    print("corpus lookup: %s\n" % ("on (circular)" if use_corpus else "OFF"))
    genders, _ = load_db(os.path.join(ROOT, "glossary.db"))
    tasks = [json.loads(l) for l in
             open(os.path.join(HERE, "..", "goldset", "tasks.jsonl"), encoding="utf-8")]

    stat = collections.Counter()
    srcs = collections.Counter()
    need = collections.Counter()

    for t in tasks:
        for entries in t["gold"].values():
            for e in entries:
                gold = unicodedata.normalize("NFC", e[0])
                stat["n"] += 1
                ent, src = p.build(gold, classify(gold))
                for field, i in (("term", 0), ("de", 1), ("hu", 2)):
                    if ent[i]:
                        stat["have_" + field] += 1
                        srcs["%s:%s" % (field, src.get(field, "?"))] += 1
                    else:
                        need["model_" + field] += 1
                if all(ent):
                    stat["complete"] += 1
                if ent[0] and not form_check(unicodedata.normalize("NFC", ent[0]),
                                             genders):
                    stat["term_clean"] += 1

    n = stat["n"] or 1
    print("gold entries: %d\n" % n)
    print("  field filled with NO model:")
    for f in ("term", "de", "hu"):
        print("    %-5s %4d  %5.1f%%" % (f, stat["have_" + f],
                                         100.0 * stat["have_" + f] / n))
    print("\n  all three fields, no model: %d  (%.1f%%)" % (
        stat["complete"], 100.0 * stat["complete"] / n))
    print("  term structurally clean   : %d  (%.1f%% of all entries)" % (
        stat["term_clean"], 100.0 * stat["term_clean"] / n))
    print("\n  sources: %s" % dict(srcs.most_common()))
    print("  model still needed for: %s" % dict(need))


if __name__ == "__main__":
    main()
