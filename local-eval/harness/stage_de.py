#!/usr/bin/env python3
"""Stage 3 — the German definition: German in, shorter German out.

Wiktionary has a gloss for 64% of entries but averages 61 characters against
the corpus's 26, and cutting at the first clause fixes the length while losing
the meaning too often ('das Zubehör' -> 'Gegenstände'). Shortening is therefore
the one stage where a chat model genuinely earns its place.

It is also the easiest thing in the pipeline to ask for: one language, no
morphology, no Hungarian, no JSON, and a wrong answer is a clumsy definition
rather than a corrupted lemma or an invented word. If a model that fits in RAM
can do anything here, it is this.

    python3 stage_de.py --model models/gemma-4-e2b-it-4bit --limit 40
"""

import argparse
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_model import call  # noqa: E402

# Same trim the deterministic baseline uses, kept here so the model is always
# compared against it on identical inputs.
CUT = re.compile(r"\s*[;,]\s*|\s+(?:der|die|das|dessen|deren|welche[rs]?|"
                 r"mit dem|bei dem|in den)\s+")
LEAD = re.compile(r"^(ein|eine|etwas|jemand|jemanden|jemandem)\s+")

SYSTEM = ("Du kürzt Wörterbuch-Erklärungen für Deutschlerner (Niveau B1). "
          "Gib eine sehr kurze Umschreibung, höchstens 30 Zeichen, ohne Artikel, "
          "kein ganzer Satz, ohne Punkt. Das Stichwort selbst darf NICHT "
          "vorkommen. Antworte nur mit der Umschreibung.")
USER = "Stichwort: {word}\nLange Erklärung: {gloss}\n\nKurze Umschreibung:"


def trim(gloss):
    return LEAD.sub("", CUT.split(gloss)[0].strip()).strip(" .,;")


def clean(text, word):
    """Take the first usable line and strip the things models add anyway.

    Gemma 4's small variants are reasoning models. Given too small a budget
    they return only their chain of thought, which is English prose about the
    task — never a definition — so it is rejected outright rather than trimmed
    into something that looks like an answer.
    """
    t = (text or "").split("<end_of_turn>")[0].strip()
    if t.lower().startswith(("thinking process", "**analyz", "1.  **")):
        return ""
    for line in t.splitlines():
        line = line.strip().strip('"').strip()
        line = re.sub(r"^(Kurze\s+)?Umschreibung\s*:\s*", "", line, flags=re.I)
        line = line.strip(' "\'.,;:')
        if 3 < len(line) < 80 and not line.lower().startswith(("hier", "ok", "gerne")):
            return line
    return ""


def shorten(base, model, word, gloss, timeout=600):
    try:
        text, _ = call(base, model, SYSTEM,
                       USER.format(word=word, gloss=gloss), "x", 0.0, 1200, timeout)
    except Exception:                      # noqa: BLE001 — fall back, don't die
        return None
    return clean(text, word) or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", default="de")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    import sqlite3
    from score import head
    lex = sqlite3.connect(os.path.abspath(
        os.path.join(HERE, "..", "models", "lexicon", "lexicon.db")))
    tasks = [json.loads(l) for l in
             open(os.path.join(HERE, "..", "goldset", "tasks.jsonl"), encoding="utf-8")]

    cases = []
    for t in tasks:
        for entries in t["gold"].values():
            for e in entries:
                r = lex.execute(
                    "SELECT de FROM lemma WHERE word = ? AND de <> '' LIMIT 1",
                    (head(e[0]),)).fetchone()
                if r and len(r[0]) > 30:            # only the ones needing work
                    cases.append((e[0], e[1], r[0]))
    cases = cases[:args.limit]
    print("cases (Wiktionary gloss longer than 30 chars): %d\n" % len(cases))

    rows = []
    for term, gold, gloss in cases:
        got = shorten(args.base, args.model, head(term), gloss)
        rows.append({"term": term, "gold": gold, "wikt": gloss,
                     "trim": trim(gloss), "model": got})

    def stats(key):
        vals = [len(r[key]) for r in rows if r.get(key)]
        n_ok = sum(1 for r in rows if r.get(key) and 4 <= len(r[key]) <= 52
                   and head(r["term"]) not in r[key].lower())
        return (statistics.mean(vals) if vals else 0, len(vals), n_ok)

    print("                    mean len   produced   within spec")
    for key, label in (("gold", "gold (frontier)"), ("trim", "first-clause trim"),
                       ("model", args.label)):
        m, produced, ok = stats(key)
        print("  %-18s %6.1f     %4d/%d      %4d  (%.0f%%)" % (
            label, m, produced, len(rows), ok, 100.0 * ok / (len(rows) or 1)))

    print("\n  sample:")
    for r in rows[:12]:
        print("    %-26s gold %-30s trim %-26s model %s" % (
            r["term"][:26], r["gold"][:30], r["trim"][:26], (r["model"] or "-")[:30]))

    out = os.path.join(HERE, "..", "results", "de-%s.json" % args.label)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
