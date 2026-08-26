#!/usr/bin/env python3
"""Does picking the Wiktionary sense by context beat always taking the first?

The pipeline composes two independent sources, so it can cross senses: 'tagen'
came out as ["tagen, tagte, hat getagt", "Tag werden", "ülésezik"] — Wiktionary's
first sense (to become day) next to the dictionary's first sense (to hold a
session). Both correct, different words, one entry.

build_lexicon.py keeps only the first gloss, so this streams the source dump
again for the words in question and keeps all of them, then scores three ways
of choosing against what the corpus actually wrote.

    python3 stage_sense.py
"""

import collections
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from score import head          # noqa: E402
from lemma import Lemmatizer    # noqa: E402

SRC = os.path.join(HERE, "..", "models", "lexicon", "dewikt.jsonl.gz")
WORD = re.compile(r"[A-Za-zÄÖÜäöüß]+")
STOP = {"der", "die", "das", "ein", "eine", "und", "oder", "von", "mit", "auf",
        "für", "sich", "man", "etwas", "jemand", "jemanden", "jemandem", "den",
        "dem", "des", "zu", "in", "im", "an", "am", "ist", "wird", "werden"}


def toks(s):
    return {w.lower() for w in WORD.findall(s or "")
            if len(w) > 3 and w.lower() not in STOP}


def main():
    lem = Lemmatizer()
    tasks = [json.loads(l) for l in
             open(os.path.join(HERE, "..", "goldset", "tasks.jsonl"), encoding="utf-8")]

    # word -> [(gold_de, sentence)]
    want = collections.defaultdict(list)
    for t in tasks:
        sents = {str(s["i"]): s["text"] for s in t["sentences"]}
        for k, entries in t["gold"].items():
            for e in entries:
                want[lem.of(head(e[0]))].append((e[1], sents.get(k, "")))

    senses = collections.defaultdict(list)
    with gzip.open(SRC, "rt", encoding="utf-8") as fh:
        for line in fh:
            if '"lang_code": "de"' not in line and '"lang_code":"de"' not in line:
                continue
            d = json.loads(line)
            w = (d.get("word") or "").lower()
            if d.get("lang_code") != "de" or w not in want:
                continue
            for s in d.get("senses") or []:
                for g in s.get("glosses") or []:
                    if g and g not in senses[w]:
                        senses[w].append(g)

    multi = {w: v for w, v in senses.items() if len(v) > 1}
    print("words wanted %d, found in dump %d, with >1 sense %d\n" % (
        len(want), len(senses), len(multi)))

    first = ctx = best = n = 0
    rows = []
    for w, cases in want.items():
        opts = senses.get(w)
        if not opts:
            continue
        for gold_de, sentence in cases:
            n += 1
            g = toks(gold_de)
            # 1. always the first sense
            first += bool(toks(opts[0]) & g)
            # 2. the sense sharing most words with the sentence it appears in
            st = toks(sentence)
            pick = max(opts, key=lambda o: len(toks(o) & st))
            ctx += bool(toks(pick) & g)
            # 3. the ceiling: the best sense available
            hit = any(toks(o) & g for o in opts)
            best += hit
            if len(opts) > 1 and hit:
                rows.append((w, gold_de, opts[0], pick))

    print("agreement with the corpus's German definition (%d cases):" % n)
    print("  always sense 1        %4d  %5.1f%%" % (first, 100.0 * first / (n or 1)))
    print("  sense picked by context %3d  %5.1f%%" % (ctx, 100.0 * ctx / (n or 1)))
    print("  best available (ceiling) %3d %5.1f%%" % (best, 100.0 * best / (n or 1)))
    print("\n  where context changed the pick:")
    shown = 0
    for w, gold, s1, pk in rows:
        if s1 != pk and shown < 8:
            print("    %-16s gold %-28s s1 %-30s ctx %s" % (
                w[:16], gold[:28], s1[:30], pk[:34]))
            shown += 1


if __name__ == "__main__":
    main()
