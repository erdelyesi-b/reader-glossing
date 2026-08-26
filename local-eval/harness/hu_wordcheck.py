#!/usr/bin/env python3
"""Are the Hungarian glosses actually Hungarian words?

The C:hu measure in score.py asks whether a gloss overlaps what the corpus
already uses for that lemma. It cannot see the failure that matters most: a
model that invents a plausible-looking Hungarian word. 'megjámul' and
'házasmertek' are not words, and a learner reading them has no way to know.

This runs every Hungarian field through the LibreOffice hu_HU hunspell
dictionary and reports the non-word rate, with the corpus's own rate as the
control — the corpus contains compounds and proper nouns that hunspell will
also reject, so only the difference between the two is meaningful.

    python3 hu_wordcheck.py ../runs/turbo-gemma4-26b.jsonl
"""

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DICT = os.path.join(HERE, "..", "models", "hu-dict", "hu_HU")
SPLIT = re.compile(r"[,;/()\[\]…·]+")


def words(field):
    out = []
    for chunk in SPLIT.split(field or ""):
        for w in chunk.split():
            w = w.strip(".!?\"'’-–—:")
            # Skip the placeholders the corpus uses for government, e.g. vmit.
            if len(w) > 2 and not w.lower().startswith(("vki", "vmi", "vala")):
                out.append(w)
    return out


def check(all_words):
    """Return the set hunspell rejects."""
    if not all_words:
        return set()
    uniq = sorted(set(all_words))
    p = subprocess.run(["hunspell", "-d", os.path.abspath(DICT), "-l"],
                       input="\n".join(uniq), capture_output=True, text=True)
    return {w for w in p.stdout.split("\n") if w.strip()}


def collect(path, key="entries"):
    out = []
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        ents = row.get(key)
        if not isinstance(ents, dict):
            continue
        for v in ents.values():
            if not isinstance(v, list):
                continue
            for e in v:
                if isinstance(e, list) and len(e) >= 3 and isinstance(e[2], str):
                    out.append(e[2])
    return out


def report(label, fields):
    ws = [w for f in fields for w in words(f)]
    bad = check(ws)
    n_bad = sum(1 for w in ws if w in bad)
    entries_bad = sum(1 for f in fields if any(w in bad for w in words(f)))
    print("%-26s %4d entries  %5d words  non-words %4d (%5.1f%%)  "
          "entries touched %4d (%5.1f%%)" % (
              label, len(fields), len(ws), n_bad,
              100.0 * n_bad / (len(ws) or 1),
              entries_bad, 100.0 * entries_bad / (len(fields) or 1)))
    return sorted(bad)


def main():
    print("Hungarian fields checked against LibreOffice hu_HU\n")
    gold = os.path.join(HERE, "..", "runs", "GOLD-frontier.jsonl")
    if os.path.exists(gold):
        g = report("GOLD (frontier)", collect(gold))
        print("   corpus rejects (control): %s\n" % ", ".join(g[:14]))
    for p in sys.argv[1:]:
        b = report(os.path.basename(p)[:-6], collect(p))
        print("   rejects: %s\n" % ", ".join(b[:24]))


if __name__ == "__main__":
    main()
