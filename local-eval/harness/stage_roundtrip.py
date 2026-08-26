#!/usr/bin/env python3
"""Generate Hungarian, then verify it by translating back to German.

The earlier gate was hunspell alone, which only asks "is this a Hungarian
word". It passed 76.7% of generated translations while just 5% were correct,
because a wrong translation is usually made of perfectly real words:
'kopfüber aufhängen' came back as 'Szöveges zaklatás' (textual harassment).

Round-tripping asks the question that actually matters. Translate the German
word to Hungarian, translate that Hungarian back to German, and keep the result
only if it lands near where it started. A wrong translation does not survive
the return trip — 'Szöveges zaklatás' goes back to something about text and
harassment, nowhere near 'kopfüber aufhängen'.

This costs a second model call per word and throws away everything it cannot
confirm, which is the right trade: a dropped candidate is invisible, a wrong
gloss is read and believed.

    python3 stage_roundtrip.py --limit 120
"""

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from score import head, hu_tokens        # noqa: E402
from lemma import Lemmatizer             # noqa: E402
from german import base_forms            # noqa: E402

STRIP_ART = re.compile(r"^(a|az|der|die|das|ein|eine)\s+", re.I)
WORD = re.compile(r"[A-Za-zÄÖÜäöüß]+")


def clean(text):
    t = (text or "").split("<end_of_turn>")[0].strip()
    t = t.splitlines()[0].strip() if t else ""
    return STRIP_ART.sub("", t).strip(" .!\"'()")


def stems(word, lem):
    """Comparable keys for a German word, so 'aufhängen' meets 'hängt auf'."""
    out = set()
    for w in WORD.findall(word or ""):
        w = w.lower()
        if len(w) < 4:
            continue
        out.add(w)
        out.add(lem.of(w))
        out |= {b for b in base_forms(w)[:4] if len(b) > 3}
        out.add(w[:6])                 # separable prefixes shuffle the tail
    return out


def round_trips(source_de, hungarian, back_de, lem):
    """Landed back where it started, and actually left in the first place.

    An untranslated echo round-trips perfectly and means nothing:
    'niederkauern' -> 'Niederkauern' -> 'Niederkauern'. Requiring the Hungarian
    to differ from the German closes that loophole, which otherwise lets the
    model pass by refusing to translate.
    """
    if stems(source_de, lem) & stems(hungarian, lem):
        return False
    a, b = stems(source_de, lem), stems(back_de, lem)
    return bool(a & b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(
        HERE, "..", "models", "translategemma-4b-it-4bit"))
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--gen", default=os.path.join(HERE, "..", "results", "hu-gen.json"))
    args = ap.parse_args()

    from mlx_lm import load, generate

    rows = json.load(open(args.gen, encoding="utf-8"))[:args.limit]
    lem = Lemmatizer()
    model, tok = load(os.path.abspath(args.model))

    def translate(text, src, tgt, n=24):
        msg = [{"role": "user", "content": [{
            "type": "text", "source_lang_code": src,
            "target_lang_code": tgt, "text": text}]}]
        return clean(generate(model, tok, prompt=tok.apply_chat_template(
            msg, add_generation_prompt=True), max_tokens=n, verbose=False))

    kept = agree_kept = dropped = agree_dropped = 0
    out = []
    for r in rows:
        if not r.get("got"):
            continue
        back = translate(r["got"], "hu", "de")
        ok = round_trips(r["src"], r["got"], back, lem)
        agrees = bool(hu_tokens(r["got"]) & hu_tokens(r["gold"]))
        if ok:
            kept += 1
            agree_kept += agrees
        else:
            dropped += 1
            agree_dropped += agrees
        out.append({**r, "back": back, "roundtrip": ok, "agrees": agrees})

    n = kept + dropped
    print("generated Hungarian for %d gap words\n" % n)
    print("  survived the round trip   %4d  %5.1f%%  <- what would ship" % (
        kept, 100.0 * kept / (n or 1)))
    print("    of those, correct       %4d  %5.1f%%  <- PRECISION" % (
        agree_kept, 100.0 * agree_kept / (kept or 1)))
    print("  rejected                  %4d  %5.1f%%" % (
        dropped, 100.0 * dropped / (n or 1)))
    print("    of those, actually right%4d  %5.1f%%  <- lost by being strict" % (
        agree_dropped, 100.0 * agree_dropped / (dropped or 1)))
    print("\n  baseline without the round trip: %.1f%% correct" % (
        100.0 * (agree_kept + agree_dropped) / (n or 1)))

    print("\n  KEPT:")
    for r in [x for x in out if x["roundtrip"]][:12]:
        print("    %-22s -> %-22s back %-22s gold %-18s %s" % (
            r["src"][:22], r["got"][:22], r["back"][:22], r["gold"][:18],
            "ok" if r["agrees"] else "differs"))
    print("\n  REJECTED:")
    for r in [x for x in out if not x["roundtrip"]][:8]:
        print("    %-22s -> %-22s back %s" % (
            r["src"][:22], r["got"][:22], r["back"][:30]))

    json.dump(out, open(os.path.join(HERE, "..", "results", "roundtrip.json"),
                        "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
