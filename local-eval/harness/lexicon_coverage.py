#!/usr/bin/env python3
"""How much of sub-task B can a lookup actually do?

Takes every gold entry, throws away its citation form, keeps only the bare
headword, and asks the lexicon to rebuild the form. Then compares against what
the frontier model wrote. This is the whole "take B away from the model"
hypothesis, measured rather than asserted.

    python3 lexicon_coverage.py
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
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from score import head, classify        # noqa: E402
from german import base_forms           # noqa: E402

LEX = os.path.join(HERE, "..", "models", "lexicon", "lexicon.db")


def lookup(db, word, want_pos):
    """Best lexicon row for a headword, preferring the right part of speech."""
    w = word.lower()
    rows = db.execute(
        "SELECT pos, term, de, hu FROM lemma WHERE word = ?", (w,)).fetchall()
    if not rows:                                   # try the surface-form index
        hit = db.execute(
            "SELECT word FROM forms WHERE surface = ? LIMIT 1", (w,)).fetchone()
        if hit:
            rows = db.execute(
                "SELECT pos, term, de, hu FROM lemma WHERE word = ?",
                (hit[0],)).fetchall()
    if not rows:                                   # last resort: cheap morphology
        for f in base_forms(w)[:6]:
            rows = db.execute(
                "SELECT pos, term, de, hu FROM lemma WHERE word = ?", (f,)).fetchall()
            if rows:
                break
    if not rows:
        return None
    for r in rows:
        if r[0] == want_pos:
            return r
    return rows[0]


def main():
    db = sqlite3.connect(os.path.abspath(LEX))
    tasks = [json.loads(l) for l in
             open(os.path.join(HERE, "..", "goldset", "tasks.jsonl"), encoding="utf-8")]

    stat = collections.Counter()
    misses, wins, losses = [], [], []

    for t in tasks:
        for entries in t["gold"].values():
            for e in entries:
                gold_term = unicodedata.normalize("NFC", e[0])
                kind = classify(gold_term)
                stat["total"] += 1
                stat["kind_" + kind] += 1
                if kind not in ("noun", "verb"):
                    continue                      # nothing to reconstruct
                stat["reconstructable"] += 1

                row = lookup(db, head(gold_term), kind)
                if not row:
                    stat["not_found"] += 1
                    misses.append(gold_term)
                    continue
                stat["found"] += 1
                got = unicodedata.normalize("NFC", row[1])
                if got == gold_term:
                    stat["exact"] += 1
                elif head(got) == head(gold_term):
                    stat["same_word_diff_form"] += 1
                    (wins if len(got) > len(gold_term) else losses).append(
                        (gold_term, got))
                else:
                    stat["wrong_word"] += 1
                    losses.append((gold_term, got))
                if row[3]:
                    stat["has_hu"] += 1
                if row[2]:
                    stat["has_de"] += 1

    n = stat["reconstructable"] or 1
    print("gold entries: %d   (noun %d, verb %d, phrase %d, other %d)" % (
        stat["total"], stat["kind_noun"], stat["kind_verb"],
        stat["kind_phrase"], stat["kind_other"]))
    print("\nnouns + verbs, where a citation form can be reconstructed: %d" % n)
    print("  found in lexicon        %5d  %5.1f%%" % (stat["found"], 100 * stat["found"] / n))
    print("    exact match to gold   %5d  %5.1f%%" % (stat["exact"], 100 * stat["exact"] / n))
    print("    same word, other form %5d  %5.1f%%" % (
        stat["same_word_diff_form"], 100 * stat["same_word_diff_form"] / n))
    print("    resolved to wrong word%5d  %5.1f%%" % (
        stat["wrong_word"], 100 * stat["wrong_word"] / n))
    print("  not found               %5d  %5.1f%%" % (
        stat["not_found"], 100 * stat["not_found"] / n))
    print("\n  of those found: German gloss %.0f%%,  Hungarian %.0f%%" % (
        100 * stat["has_de"] / (stat["found"] or 1),
        100 * stat["has_hu"] / (stat["found"] or 1)))

    print("\nnot found (sample):  %s" % ", ".join(misses[:12]))
    print("\ndisagreements (gold -> lexicon):")
    for a, b in (losses + wins)[:12]:
        print("  %-42s -> %s" % (a, b))
    db.close()


if __name__ == "__main__":
    main()
