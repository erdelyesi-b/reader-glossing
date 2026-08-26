#!/usr/bin/env python3
"""The whole pipeline, end to end, emitting a run file score.py can grade.

Four stages, each handled by whatever is actually good at it:

  1 SELECT   which candidates deserve an entry        rules over the lexicon
  2 FORM     der/die/das + plural, or the three parts  Wiktionary + CharSplit
  3 HU       Hungarian                                 dictionary.sqlite,
                                                       ranked by TranslateGemma
  4 DE       learner-German definition                 Wiktionary, shortened by
                                                       a small chat model
  5 CHECK    hunspell, charset, length, shape          rules
  6 ASSEMBLE the JSON                                  Python

The models run in separate phases because only one fits in 8 GB at a time, so
each phase caches its output and the next picks it up. That is a constraint of
the machine, but it is also why the design works: no single model is being
asked to hold the whole job in its head.

    python3 run_pipeline.py --phase mt   --model models/translategemma-4b-it-4bit
    python3 run_pipeline.py --phase de   --model models/gemma-4-e2b-it-4bit
    python3 run_pipeline.py --phase assemble
"""

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from score import head, hu_tokens, form_check, non_latin, load_db  # noqa: E402
from stage_form import Former                                      # noqa: E402
from stage_hu import Dictionary                                    # noqa: E402
from stage_mt_rank import stems, matches                           # noqa: E402
from stage_de import trim, shorten                                 # noqa: E402
from lemma import Lemmatizer                                       # noqa: E402

CACHE = os.path.join(HERE, "..", "runs", "pipeline-cache.json")
TASKS = os.path.join(HERE, "..", "goldset", "tasks.jsonl")
MAX_PER_SENTENCE = 3       # README §5: the band is a density, not a maximum


def load_tasks():
    return [json.loads(l) for l in open(TASKS, encoding="utf-8")]


def cache_load():
    if os.path.exists(CACHE):
        return json.load(open(CACHE, encoding="utf-8"))
    return {"select": {}, "mt": {}, "de": {}}


def cache_save(c):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(c, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)


# ------------------------------------------------------------------ 1. SELECT

def select(tasks, former, dictionary):
    """Pick candidates worth glossing, with no model.

    README §4 says to skip proper nouns, fragments and anything sub-A2. Two of
    those are decidable here: a capitalised word no lexicon knows is almost
    always a name, and a token under three letters is a fragment. What is left
    is ranked by how obscure the word is, which is what rarest-first already
    does elsewhere in this repo.
    """
    out = {}
    for t in tasks:
        picks = {}
        seen = set()
        for s in t["sentences"]:
            chosen = []
            for w in s["new"]:
                lw = w.lower()
                if len(w) < 3 or lw in seen:
                    continue
                known = former.form(lw, None) or dictionary.look(lw)
                if not known:
                    continue                      # unknown to every source
                if w[:1].isupper() and not former.form(lw, "noun"):
                    continue                      # capitalised and unattested: a name
                chosen.append(w)
            if chosen:
                picks[str(s["i"])] = chosen[:MAX_PER_SENTENCE]
                seen.update(c.lower() for c in chosen[:MAX_PER_SENTENCE])
        out[t["id"]] = picks
    return out


# ---------------------------------------------------------------------- phases

def phase_mt(args, tasks, sel):
    """Translate each selected sentence once; cache it for the ranking step."""
    from mlx_lm import load, generate
    model, tok = load(os.path.abspath(args.model))
    cache = cache_load()
    mt = cache.get("mt", {})
    todo = [(t["id"], str(s["i"]), s["text"])
            for t in tasks for s in t["sentences"]
            if str(s["i"]) in sel.get(t["id"], {})]
    print("translating %d sentences" % len(todo))
    for n, (tid, si, text) in enumerate(todo, 1):
        key = "%s:%s" % (tid, si)
        if key in mt:
            continue
        msg = [{"role": "user", "content": [{
            "type": "text", "source_lang_code": "de",
            "target_lang_code": "hu", "text": text}]}]
        out = generate(model, tok, prompt=tok.apply_chat_template(
            msg, add_generation_prompt=True), max_tokens=180, verbose=False)
        mt[key] = (out or "").split("<end_of_turn>")[0].strip()
        if n % 25 == 0:
            cache["mt"] = mt
            cache_save(cache)
            print("  %d/%d" % (n, len(todo)), flush=True)
    cache["mt"] = mt
    cache_save(cache)
    print("cached %d translations" % len(mt))


def phase_de(args, tasks, sel, former):
    """Shorten each needed Wiktionary gloss once."""
    import sqlite3
    lex = sqlite3.connect(os.path.abspath(
        os.path.join(HERE, "..", "models", "lexicon", "lexicon.db")))
    cache = cache_load()
    de = cache.get("de", {})
    words = sorted({w.lower() for t in tasks
                    for ws in sel.get(t["id"], {}).values() for w in ws})
    print("shortening definitions for %d words" % len(words))
    for n, w in enumerate(words, 1):
        if w in de:
            continue
        row = lex.execute(
            "SELECT de FROM lemma WHERE word = ? AND de <> '' LIMIT 1", (w,)).fetchone()
        if not row:
            de[w] = None
            continue
        gloss = row[0]
        # The first-clause trim already lands 90% within spec at zero cost, so
        # the model is called only where the trim comes out unusable — too
        # long, or collapsed to something uselessly generic. On the gold set
        # that is a minority of words, and it is the difference between ~110
        # minutes of definitions per chapter and ~15.
        if len(gloss) <= 34:
            de[w] = gloss
        else:
            t = trim(gloss)
            de[w] = (t if 8 <= len(t) <= 46
                     else (shorten(args.base, args.model, w, gloss) or t))
        if n % 25 == 0:
            cache["de"] = de
            cache_save(cache)
            print("  %d/%d" % (n, len(words)), flush=True)
    cache["de"] = de
    cache_save(cache)
    print("cached %d definitions" % len(de))


# ---------------------------------------------------------------- 5+6 ASSEMBLE

def recase(text, lexicon):
    """Restore capitals on any word Wiktionary knows only as a noun."""
    out = []
    for w in text.split():
        bare = w.strip(".,;:()")
        if bare and bare[:1].islower() and len(bare) > 3:
            row = lexicon.execute(
                "SELECT pos FROM lemma WHERE word = ? AND pos = 'noun' LIMIT 1",
                (bare.lower(),)).fetchone()
            if row:
                w = w.replace(bare, bare[0].upper() + bare[1:], 1)
        out.append(w)
    return " ".join(out)


def hunspell_bad(fields):
    """Hungarian words the hu_HU dictionary rejects.

    The source de-hu dictionary has its own typos — 'probláma' for 'probléma',
    'aláterí' for 'aláterít' — which no amount of pipeline care can prevent.
    A spellchecker catches them for free, and because the dictionary usually
    offers several senses the bad one can be dropped rather than the entry.
    """
    words = sorted({w for f in fields for w in re.split(r"[,;/()\s]+", f or "")
                    if len(w) > 2})
    if not words:
        return set()
    dic = os.path.abspath(os.path.join(HERE, "..", "models", "hu-dict", "hu_HU"))
    if not os.path.exists(dic + ".dic"):
        return set()
    p = subprocess.run(["hunspell", "-d", dic, "-l"], input="\n".join(words),
                       capture_output=True, text=True)
    return {w for w in p.stdout.split("\n") if w.strip()}


def phase_assemble(tasks, sel, former, dictionary):
    import sqlite3
    lexicon = sqlite3.connect(os.path.abspath(os.path.join(
        HERE, "..", "models", "lexicon", "lexicon.db")))
    lemmatizer = Lemmatizer()
    cache = cache_load()
    mt, de_cache = cache.get("mt", {}), cache.get("de", {})
    genders, _ = load_db(os.path.join(ROOT, "glossary.db"))

    all_hu = [s for t in tasks for ws in sel.get(t["id"], {}).values()
              for w in ws for s in (dictionary.look(w.lower()) or (None, [], ""))[1]]
    st = stems([w for tr in mt.values() for w in hu_tokens(tr)] +
               [w for s in all_hu for w in hu_tokens(s)])

    bad_words = hunspell_bad(
        [x for t in tasks for ws in sel.get(t["id"], {}).values() for w in ws
         for x in (dictionary.look(w.lower()) or (None, [], ""))[1]])
    rows, drop = [], collections.Counter()
    for t in tasks:
        entries = {}
        for si, words in sel.get(t["id"], {}).items():
            got = []
            for w in words:
                lw = w.lower()
                f = former.form(lw, None)
                look = dictionary.look(lw)

                term = f[0] if f else (
                    "%s %s" % (look[0], w.capitalize())
                    if look and look[0] else None)
                if not term:
                    drop["no-term"] += 1
                    continue

                # 3. HU — dictionary senses, ranked against the translation.
                hu = None
                if look and look[1]:
                    senses = look[1][:5]
                    tr = hu_tokens(mt.get("%s:%s" % (t["id"], si), ""))
                    picked = [s for s in senses if matches(s, tr, st)]
                    ordered = picked or senses
                    # Drop senses the spellchecker rejects, keeping the rest.
                    good = [x for x in ordered
                            if not any(w in bad_words
                                       for w in re.split(r"[,;/()\s]+", x) if len(w) > 2)]
                    hu = ", ".join((good or [])[:2])
                    if not hu:
                        drop["hu-misspelled"] += 1
                if not hu:
                    drop["no-hu"] += 1
                    continue

                # Definitions were cached under the surface form; resolve the
                # lemma too, and fall back to trimming Wiktionary directly so a
                # word missing from the cache is not lost for want of a lookup.
                de = de_cache.get(lw) or de_cache.get(lemmatizer.of(lw))
                if not de:
                    row = lexicon.execute(
                        "SELECT de FROM lemma WHERE word = ? AND de <> '' LIMIT 1",
                        (lemmatizer.of(lw),)).fetchone()
                    if row:
                        short = trim(row[0])
                        de = short if 8 <= len(short) <= 46 else None
                if not de:
                    drop["no-de"] += 1
                    continue

                # 5. CHECK — every guard the merge would apply, before shipping.
                entry = [term, de, hu]
                if any(non_latin(x) for x in entry):
                    drop["charset"] += 1
                    continue
                if form_check(unicodedata.normalize("NFC", term), genders):
                    drop["bad-form"] += 1
                    continue
                if len(de) > 52 or len(hu) > 30:
                    drop["too-long"] += 1
                    continue
                # German capitalises nouns. E2B lowercases them often enough
                # ('tiefe zuneigung', 'mächtige weibliche figur') that a
                # learner would read it as wrong German, and nothing else here
                # would notice.
                de = recase(de, lexicon)
                got.append(entry)
            if got:
                entries[si] = got
        rows.append({"id": t["id"], "book": t["book"], "chapter": t["chapter"],
                     "seconds": 0.0, "usage": {}, "parse": "clean", "error": None,
                     "raw": "", "entries": entries})

    out = os.path.join(HERE, "..", "runs", "PIPELINE.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = sum(len(v) for r in rows for v in r["entries"].values())
    print("wrote %s\n  %d entries across %d tasks" % (out, n, len(rows)))
    print("  rejected by the checks: %s" % dict(drop))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=("select", "mt", "de", "assemble"))
    ap.add_argument("--base", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="")
    args = ap.parse_args()

    tasks = load_tasks()
    former, dictionary = Former(), Dictionary()
    cache = cache_load()
    if not cache.get("select"):
        cache["select"] = select(tasks, former, dictionary)
        cache_save(cache)
        picked = sum(len(w) for v in cache["select"].values() for w in v.values())
        print("selected %d entries to build" % picked)
    sel = cache["select"]

    if args.phase == "select":
        return
    if args.phase == "mt":
        phase_mt(args, tasks, sel)
    elif args.phase == "de":
        phase_de(args, tasks, sel, former)
    else:
        phase_assemble(tasks, sel, former, dictionary)


if __name__ == "__main__":
    main()
