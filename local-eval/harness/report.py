#!/usr/bin/env python3
"""Collect every run into one scorecard, plus the two non-model results.

score.py grades one run at a time; this is the thing to read at the end. It
adds the results that are not model runs — what the lexicon can reconstruct on
its own, and what a lexicon-plus-model split would score — because the answer
to "which solution" turned out to be a split rather than a single model.

    python3 report.py
"""

import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, *args):
    return subprocess.run([sys.executable, os.path.join(HERE, script)] + list(args),
                          capture_output=True, text=True).stdout


def main():
    runs = sorted(glob.glob(os.path.join(HERE, "..", "runs", "*.jsonl")))
    runs = [r for r in runs if os.path.getsize(r) > 0]
    # Baseline first, then the rest.
    runs.sort(key=lambda p: (0 if "GOLD" in p else 1, p))

    print("=" * 100)
    print("MODEL RUNS  (all on the same tasks, same prompt, temperature 0)")
    print("=" * 100)
    print(run("score.py", *runs, "--detail"))

    print("=" * 100)
    print("LEXICON — can a lookup write the citation form?")
    print("=" * 100)
    print(run("lexicon_coverage.py"))

    print("=" * 100)
    print("HYBRID — what B scores if the lexicon writes the term")
    print("=" * 100)
    print(run("hybrid_projection.py"))

    tg = os.path.join(HERE, "..", "results", "translategemma-hu.json")
    if os.path.exists(tg):
        d = json.load(open(tg, encoding="utf-8"))
        print("=" * 100)
        print("TRANSLATEGEMMA-4B — Hungarian only (it has no chat mode, so it "
              "cannot do selection or definitions)")
        print("=" * 100)
        print("  %d terms, %d share a token with the corpus gloss (%.1f%%)" % (
            d["n"], d["hit"], 100.0 * d["hit"] / d["n"]))


if __name__ == "__main__":
    main()
