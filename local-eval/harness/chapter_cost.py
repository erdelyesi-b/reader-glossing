#!/usr/bin/env python3
"""What would glossing a real chapter locally actually cost in wall-clock time?

The bench runs 10-sentence tasks; a chapter is 150-700 sentences. This projects
one onto the other using measured throughput, and shows what dropping the
candidate-free sentences saves — the single biggest lever, because prefill is
most of the cost on a model that streams its weights.

Rates default to what TurboFieldfare measured here on the 8 GB M2.

    python3 chapter_cost.py --prefill 500 --decode 2.0
"""

import argparse
import glob
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CHARS_PER_TOKEN = 3.5          # German, Gemma tokenizer, measured on these batches
TOKENS_PER_ENTRY = 30          # README §8's budget, and what the bench saw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", type=float, default=500.0,
                    help="prompt tokens/second")
    ap.add_argument("--decode", type=float, default=2.0,
                    help="generated tokens/second")
    ap.add_argument("--work", default=os.path.join(ROOT, "work"))
    args = ap.parse_args()

    chapters = {}
    for bp in sorted(glob.glob(os.path.join(args.work, "*", "ch*", "batch_*.json"))):
        key = os.sep.join(bp.split(os.sep)[-3:-1])
        batch = json.load(open(bp, encoding="utf-8"))
        c = chapters.setdefault(key, {"sent": 0, "live": 0, "all_ch": 0,
                                      "live_ch": 0, "cands": 0})
        for s in batch["sentences"]:
            c["sent"] += 1
            c["all_ch"] += len(s["text"])
            if s["new"]:
                c["live"] += 1
                c["live_ch"] += len(s["text"])
                c["cands"] += len(s["new"])

    def minutes(chars, entries):
        prompt = chars / CHARS_PER_TOKEN
        out = entries * TOKENS_PER_ENTRY
        return (prompt / args.prefill + out / args.decode) / 60.0

    rows = []
    for k, c in chapters.items():
        # Roughly one entry per two candidates, which is what the corpus shows.
        entries = c["cands"] * 0.55
        rows.append((k, c, minutes(c["all_ch"], entries),
                     minutes(c["live_ch"], entries)))

    full = [r[2] for r in rows]
    trim = [r[3] for r in rows]
    print("throughput assumed: %.0f prompt tok/s, %.1f gen tok/s" % (
        args.prefill, args.decode))
    print("chapters: %d\n" % len(rows))
    print("  per chapter, all sentences in the prompt : median %5.1f min   max %5.1f" % (
        statistics.median(full), max(full)))
    print("  per chapter, candidate-free dropped      : median %5.1f min   max %5.1f" % (
        statistics.median(trim), max(trim)))
    print("  saving: %.0f%%" % (100 * (1 - sum(trim) / sum(full))))

    print("\n  a 37-chapter book: %.1f h  ->  %.1f h" % (
        statistics.median(full) * 37 / 60, statistics.median(trim) * 37 / 60))

    rows.sort(key=lambda r: -r[3])
    print("\n  worst chapters (trimmed):")
    for k, c, f, t in rows[:5]:
        print("    %-52s %4d sent (%3d live) %4d cand  %5.1f min" % (
            k[-52:], c["sent"], c["live"], c["cands"], t))


if __name__ == "__main__":
    main()
