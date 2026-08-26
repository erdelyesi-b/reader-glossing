#!/usr/bin/env python3
"""Stage 4b — pick which dictionary sense fits this sentence.

The monolithic run's worst failure was inventing Hungarian words. Ranking makes
that impossible by construction: the model chooses among senses the dictionary
supplied, so the worst case is the wrong real word instead of a fabricated one.

It is also a much smaller job than writing an entry, which matters because the
only models that fit in 8 GB are the ones that failed the big job. This asks
whether a 2B that cannot write a glossary entry can still choose from a list —
answering it decides whether the pipeline needs the 26B at all.

Baseline to beat: taking the dictionary's first sense with no model.

    python3 stage_rank.py --base http://127.0.0.1:8080/v1 --model <path>
"""

import argparse
import collections
import json
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from score import head, hu_tokens          # noqa: E402
from stage_hu import Dictionary            # noqa: E402
from run_model import call                 # noqa: E402

SYSTEM = ("Du wählst die passende ungarische Bedeutung. Antworte NUR mit der "
          "Nummer der besten Option, sonst nichts.")
USER = """Satz: {sentence}
Deutsches Wort: {word}

Welche ungarische Übersetzung passt hier?
{options}

Antwort (nur die Nummer):"""


def build_cases(dictionary, limit):
    """Gold entries where the dictionary offers a real choice."""
    tasks = [json.loads(l) for l in
             open(os.path.join(HERE, "..", "goldset", "tasks.jsonl"), encoding="utf-8")]
    cases = []
    for t in tasks:
        sents = {str(s["i"]): s["text"] for s in t["sentences"]}
        for k, entries in t["gold"].items():
            for e in entries:
                term = unicodedata.normalize("NFC", e[0])
                got = dictionary.look(head(term))
                if not got or len(got[1]) < 2:
                    continue                    # no choice to make
                senses = got[1][:5]
                gtok = hu_tokens(e[2])
                right = [i for i, s in enumerate(senses) if hu_tokens(s) & gtok]
                if not right:
                    continue                    # gold isn't on the menu; unscoreable
                cases.append({"sentence": sents.get(k, ""), "word": head(term),
                              "senses": senses, "right": set(right), "gold": e[2]})
    return cases[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="rank")
    ap.add_argument("--limit", type=int, default=80)
    args = ap.parse_args()

    d = Dictionary()
    cases = build_cases(d, args.limit)
    print("scoreable cases (dictionary offers >1 sense and gold is among them): %d\n"
          % len(cases))

    first = sum(1 for c in cases if 0 in c["right"])
    print("  baseline, always take sense 1: %d/%d  (%.1f%%)" % (
        first, len(cases), 100.0 * first / (len(cases) or 1)))

    hit = bad = 0
    rows = []
    for c in cases:
        opts = "\n".join("%d. %s" % (i + 1, s) for i, s in enumerate(c["senses"]))
        try:
            text, _ = call(args.base, args.model, SYSTEM,
                           USER.format(sentence=c["sentence"], word=c["word"],
                                       options=opts),
                           "x", 0.0, 512, 900)
        except Exception as e:                     # noqa: BLE001 - report, don't die
            text = ""
            bad += 1
        m = re.search(r"(?s)(?:Antwort|answer)\D{0,12}(\d)", text or "") or re.search(r"\b(\d)\b(?!.*\b\d\b)", (text or "")[-80:])
        pick = int(m.group()) - 1 if m else -1
        ok = pick in c["right"]
        hit += ok
        rows.append({"word": c["word"], "pick": pick, "senses": c["senses"],
                     "gold": c["gold"], "ok": ok, "raw": (text or "")[:20]})

    n = len(cases) or 1
    print("  %-22s %d/%d  (%.1f%%)   unusable replies: %d" % (
        args.label, hit, len(cases), 100.0 * hit / n, bad))
    print("\n  sample:")
    for r in rows[:12]:
        chose = r["senses"][r["pick"]] if 0 <= r["pick"] < len(r["senses"]) else "?"
        print("    %-22s chose %-18s gold %-18s %s" % (
            r["word"][:22], chose[:18], r["gold"][:18], "ok" if r["ok"] else "MISS"))

    out = os.path.join(HERE, "..", "results", "rank-%s.json" % args.label)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"n": len(cases), "hit": hit, "baseline_first": first, "rows": rows},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
