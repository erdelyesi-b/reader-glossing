#!/usr/bin/env python3
"""How good is a local translation model at the Hungarian field, on its own?

TranslateGemma is not a chat model — its template accepts only
{type, source_lang_code, target_lang_code, text} — so it can never do sub-task
A or write a German definition. The only role it could play is the `hu` field.
This measures whether it is good enough for that one job.

Scored against the corpus: for each term, does the model's Hungarian share a
token with any Hungarian the corpus already uses for that lemma? That is the
same C:hu measure score.py applies to the general models, so the numbers are
comparable.

    python3 test_translategemma.py --n 150
"""

import argparse
import collections
import json
import os
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from score import head, hu_tokens  # noqa: E402


def clean_hu(text):
    """Normalise the model's answer before scoring it.

    The corpus writes bare citation forms, so the Hungarian definite article
    and a leading capital are formatting, not errors — counting them as wrong
    would understate the model. Genuine mistranslations still miss.
    """
    t = (text or "").split("<end_of_turn>")[0].strip()
    t = t.splitlines()[0].strip() if t else ""
    t = re.sub(r"^(a|az)\s+", "", t, flags=re.I)
    return t.strip(" .!\"'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "..", "models",
                                                    "translategemma-4b-it-4bit"))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results",
                                                  "translategemma-hu.json"))
    args = ap.parse_args()

    from mlx_lm import load, generate

    tasks = [json.loads(l) for l in
             open(os.path.join(HERE, "..", "goldset", "tasks.jsonl"), encoding="utf-8")]
    items = []
    for t in tasks:
        for entries in t["gold"].values():
            for e in entries:
                items.append((e[0], e[2]))       # term, gold hungarian
    items = items[:args.n]

    model, tok = load(os.path.abspath(args.model))
    rows, hit = [], 0
    for term, gold_hu in items:
        # Feed the bare headword, not the citation form: "das Sockenpaar, -e"
        # is a dictionary convention, and an MT model reads the ", -e" as text.
        # Keep 'sich' and particles, which do change the meaning.
        src = term.split(",")[0].strip() if head(term) else term
        msg = [{"role": "user", "content": [{
            "type": "text", "source_lang_code": "de",
            "target_lang_code": "hu", "text": src}]}]
        prompt = tok.apply_chat_template(msg, add_generation_prompt=True)
        out = generate(model, tok, prompt=prompt, max_tokens=24, verbose=False)
        out = clean_hu(out)
        ok = bool(hu_tokens(out) & hu_tokens(gold_hu))
        hit += ok
        rows.append({"term": term, "gold": gold_hu, "got": out, "match": ok})

    n = len(rows) or 1
    print("translategemma-4b, German term -> Hungarian, %d terms" % n)
    print("  shares a token with the corpus gloss: %d  (%.1f%%)" % (hit, 100.0 * hit / n))
    print("\n  sample:")
    for r in rows[:20]:
        print("    %-44s gold %-22s got %-22s %s" % (
            r["term"][:44], r["gold"][:22], r["got"][:22], "ok" if r["match"] else "MISS"))

    out_p = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    json.dump({"n": n, "hit": hit, "rows": rows}, open(out_p, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n  wrote %s" % out_p)


if __name__ == "__main__":
    main()
