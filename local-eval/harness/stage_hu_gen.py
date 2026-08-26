#!/usr/bin/env python3
"""Stage 4c — generate Hungarian for the words no dictionary covers.

204 of 683 selected words are dropped for having no Hungarian at all, and that
single gap is most of the difference between 320 entries and the corpus's 424.
Generation is the only way to close it, and generation is exactly what put
'megjámul' and 'házasmertek' into the monolithic run.

What makes it safe now is that the gate exists. Every generated word goes
through hunspell before it can ship, so the failure mode changes from "invents
a word the reader cannot check" to "produces nothing for this candidate" —
which the pipeline already handles by dropping the entry.

Measured three ways, because they are different questions:
  fill      how often a usable Hungarian comes out at all
  clean     how often it survives the spellchecker
  agrees    how often it matches what the corpus chose, where the corpus has it

    python3 stage_hu_gen.py --limit 120
"""

import argparse
import collections
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from score import head, hu_tokens        # noqa: E402
from stage_hu import Dictionary          # noqa: E402
from lemma import Lemmatizer             # noqa: E402

HUDICT = os.path.join(HERE, "..", "models", "hu-dict", "hu_HU")
STRIP_ART = re.compile(r"^(a|az)\s+", re.I)


def clean_hu(text):
    t = (text or "").split("<end_of_turn>")[0].strip()
    t = t.splitlines()[0].strip() if t else ""
    t = STRIP_ART.sub("", t)
    return t.strip(" .!\"'()")


def spell_bad(words):
    words = sorted({w for w in words if len(w) > 2})
    if not words or not os.path.exists(os.path.abspath(HUDICT) + ".dic"):
        return set()
    p = subprocess.run(["hunspell", "-d", os.path.abspath(HUDICT), "-l"],
                       input="\n".join(words), capture_output=True, text=True)
    return {w for w in p.stdout.split("\n") if w.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(
        HERE, "..", "models", "translategemma-4b-it-4bit"))
    ap.add_argument("--limit", type=int, default=120)
    args = ap.parse_args()

    from mlx_lm import load, generate

    d, lem = Dictionary(), Lemmatizer()
    tasks = [json.loads(l) for l in
             open(os.path.join(HERE, "..", "goldset", "tasks.jsonl"), encoding="utf-8")]

    # The gap: gold entries whose word the dictionary cannot translate.
    gaps = []
    for t in tasks:
        for entries in t["gold"].values():
            for e in entries:
                w = head(e[0])
                if not d.look(w):
                    gaps.append((w, e[2], e[0]))
    gaps = gaps[:args.limit]
    print("gap words (no dictionary Hungarian): %d\n" % len(gaps))

    model, tok = load(os.path.abspath(args.model))
    rows = []
    for w, gold_hu, term in gaps:
        # Feed the lemma, not the citation form: an MT model reads ', -e' as
        # text. The lemma is also what the dictionary would have been keyed on.
        src = lem.of(w)
        msg = [{"role": "user", "content": [{
            "type": "text", "source_lang_code": "de",
            "target_lang_code": "hu", "text": src}]}]
        out = generate(model, tok, prompt=tok.apply_chat_template(
            msg, add_generation_prompt=True), max_tokens=24, verbose=False)
        rows.append({"word": w, "src": src, "gold": gold_hu,
                     "got": clean_hu(out), "term": term})

    bad = spell_bad([x for r in rows for x in re.split(r"[,;/()\s]+", r["got"])])
    n = len(rows) or 1
    fill = sum(1 for r in rows if r["got"])
    clean = 0
    agree = 0
    for r in rows:
        toks = [x for x in re.split(r"[,;/()\s]+", r["got"]) if len(x) > 2]
        r["clean"] = bool(r["got"]) and not any(x in bad for x in toks)
        clean += r["clean"]
        r["agrees"] = bool(hu_tokens(r["got"]) & hu_tokens(r["gold"]))
        agree += r["agrees"] and r["clean"]

    print("  produced something      %4d  %5.1f%%" % (fill, 100.0 * fill / n))
    print("  survives hunspell       %4d  %5.1f%%   <- what would actually ship" % (
        clean, 100.0 * clean / n))
    print("  agrees with the corpus  %4d  %5.1f%%   (of those that ship: %.0f%%)" % (
        agree, 100.0 * agree / n, 100.0 * agree / (clean or 1)))
    print("\n  shipped sample:")
    for r in [x for x in rows if x["clean"]][:14]:
        print("    %-24s -> %-24s gold %-22s %s" % (
            r["src"][:24], r["got"][:24], r["gold"][:22],
            "ok" if r["agrees"] else "differs"))
    print("\n  rejected by hunspell (would be dropped, not shipped):")
    for r in [x for x in rows if x["got"] and not x["clean"]][:10]:
        print("    %-24s -> %s" % (r["src"][:24], r["got"][:34]))

    out_p = os.path.join(HERE, "..", "results", "hu-gen.json")
    os.makedirs(os.path.dirname(out_p), exist_ok=True)
    json.dump(rows, open(out_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
