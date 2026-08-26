#!/usr/bin/env python3
"""TranslateGemma as the working model — used the way an MT model wants to be.

The first TranslateGemma test fed it bare citation forms and scored 13.3%. That
was the wrong test: an MT model is trained on sentences, and a word with no
context is its worst case. It also has no chat mode, so it cannot be asked to
choose, explain, or emit JSON.

But it does not need to. Translate the whole sentence, then look for which of
the dictionary's senses actually turns up in that translation. The MT model
supplies context — its real strength — while the dictionary supplies the
vocabulary, so an invented word remains impossible. It is the same ranking
stage as stage_rank.py, with translation standing in for judgement.

Hungarian is agglutinative, so 'behúzta' has to match the sense 'behúz'.
hunspell -s gives the stems; a prefix comparison catches the rest.

Any German→Hungarian translator can be scored here, not just TranslateGemma —
`translators.py` hides whether it is MLX, a seq2seq MT model or a GGUF behind
llama-server. The matcher and the case set stay fixed, so the numbers from
different backends sit in one table.

    python3 stage_mt_rank.py --limit 60
    python3 stage_mt_rank.py --backend nllb --model ../models/nllb-1.3b \
        --label nllb-1.3b
"""

import argparse
import collections
import json
import os
import subprocess
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from score import head, hu_tokens          # noqa: E402
from stage_hu import Dictionary            # noqa: E402
from translators import build              # noqa: E402

HUDICT = os.path.join(HERE, "..", "models", "hu-dict", "hu_HU")
STEM_LEN = 5


def stems(words):
    """word -> {stems}, via hunspell's morphological analyser."""
    out = collections.defaultdict(set)
    uniq = sorted({w for w in words if w})
    if not uniq:
        return out
    p = subprocess.run(["hunspell", "-d", os.path.abspath(HUDICT), "-s"],
                       input="\n".join(uniq), capture_output=True, text=True)
    cur = None
    for line in p.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        cur = parts[0]
        out[cur] |= set(parts)
    for w in uniq:
        out[w].add(w)
    return out


def matches(sense, translation_words, st):
    """Does this dictionary sense appear in the translated sentence?"""
    for token in hu_tokens(sense):
        cands = st.get(token, {token}) | {token}
        for tw in translation_words:
            tws = st.get(tw, {tw}) | {tw}
            if cands & tws:
                return True
            # Agglutination the stemmer missed: compare truncated stems.
            if len(token) >= STEM_LEN and len(tw) >= STEM_LEN and \
                    token[:STEM_LEN] == tw[:STEM_LEN]:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(
        HERE, "..", "models", "translategemma-4b-it-4bit"))
    ap.add_argument("--backend", default="translategemma",
                    choices=["translategemma", "mlx-chat", "nllb", "openai"])
    ap.add_argument("--base", default=None,
                    help="OpenAI-compatible endpoint, for --backend openai")
    ap.add_argument("--label", default="mt",
                    help="names the run file: results/rank-<label>.json")
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()

    d = Dictionary()
    tasks = [json.loads(l) for l in
             open(os.path.join(HERE, "..", "goldset", "tasks.jsonl"), encoding="utf-8")]

    # Same case set as stage_rank.py, so the numbers are directly comparable.
    cases = []
    for t in tasks:
        sents = {str(s["i"]): s["text"] for s in t["sentences"]}
        for k, entries in t["gold"].items():
            for e in entries:
                got = d.look(head(unicodedata.normalize("NFC", e[0])))
                if not got or len(got[1]) < 2:
                    continue
                senses = got[1][:5]
                gtok = hu_tokens(e[2])
                right = {i for i, s in enumerate(senses) if hu_tokens(s) & gtok}
                if not right:
                    continue
                cases.append({"sentence": sents.get(k, ""), "word": head(e[0]),
                              "senses": senses, "right": right, "gold": e[2]})
    cases = cases[:args.limit]
    print("cases: %d\n" % len(cases))
    base = sum(1 for c in cases if 0 in c["right"])
    print("  baseline, always sense 1: %d/%d (%.1f%%)" % (
        base, len(cases), 100.0 * base / (len(cases) or 1)))

    model_ref = args.model if args.backend == "openai" \
        else os.path.abspath(args.model)
    translate = build(args.backend, model_ref, base=args.base)
    translations = translate([c["sentence"] for c in cases])

    st = stems([w for tr in translations for w in hu_tokens(tr)] +
               [w for c in cases for s in c["senses"] for w in hu_tokens(s)])

    hit = found = 0
    rows = []
    for c, tr in zip(cases, translations):
        tw = hu_tokens(tr)
        picked = [i for i, s in enumerate(c["senses"]) if matches(s, tw, st)]
        if picked:
            found += 1
            ok = bool(set(picked) & c["right"])
        else:
            ok = 0 in c["right"]          # fall back to sense 1
        hit += ok
        rows.append({"word": c["word"], "picked": picked, "ok": bool(ok),
                     "senses": c["senses"], "gold": c["gold"], "tr": tr[:70]})

    n = len(cases) or 1
    print("  a sense was located in the translation: %d/%d (%.1f%%)" % (
        found, n, 100.0 * found / n))
    print("  correct (falling back to sense 1 when not located): %d/%d (%.1f%%)" % (
        hit, n, 100.0 * hit / n))
    print("\n  sample:")
    for r in rows[:10]:
        chose = ", ".join(r["senses"][i] for i in r["picked"]) or "(none found)"
        print("    %-18s -> %-24s gold %-20s %s" % (
            r["word"][:18], chose[:24], r["gold"][:20], "ok" if r["ok"] else "MISS"))

    out = os.path.join(HERE, "..", "results", "rank-%s.json" % args.label)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"n": n, "hit": hit, "found": found, "baseline": base, "rows": rows},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
