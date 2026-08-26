#!/usr/bin/env python3
"""One sentence at a time, with every check Balázs specified.

The earlier attempts checked generated Hungarian one way at a time — spelling
only (76.7% passed, 5% correct), then a round trip (17.5% passed, 14.3%
correct). Both fail for the same reason: a single check has a single blind
spot, and a wrong translation made of real words walks through it.

This runs two *independent* translation paths for the same word and requires
them to agree:

    path A   the word, in a list of words, translated on its own
    path B   the whole sentence translated, then the word looked for in it

A wrong translation has to be wrong the same way twice to survive, and the two
paths see completely different context. That is a far stronger test than
asking one path to justify itself.

The full order, per sentence:

  1 candidates   tokens the glossary could not resolve
  2 real word    is this German at all, or dialect/OCR noise/a fragment
  3 A2+          above the level the corpus bothers to gloss
  4 lemma        surface form -> lemma
  5 form         lemma -> citation form (article + plural, or the three parts)
  6 path A       translate the lemma list
  7 path B       translate the sentence
  8 agree        the word's Hungarian must show up in the sentence Hungarian
  9 leakage      Hungarian, and not German or English, and spelled correctly
 10 grammar      the citation form is structurally sound
 11 assemble     Python writes the JSON

    python3 sentence_pipeline.py --limit 40
"""

import argparse
import collections
import json
import os
import re
import sqlite3
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
from lemma import Lemmatizer                                       # noqa: E402
from stage_mt_rank import stems as hu_stems                        # noqa: E402

LEX = os.path.join(HERE, "..", "models", "lexicon", "lexicon.db")
HUDICT = os.path.join(HERE, "..", "models", "hu-dict", "hu_HU")
STRIP = re.compile(r"^(a|az)\s+", re.I)
SPLIT = re.compile(r"[,;/()\s]+")


def clean(text):
    t = (text or "").split("<end_of_turn>")[0].strip()
    t = t.splitlines()[0].strip() if t else ""
    return STRIP.sub("", t).strip(" .!\"'()[]")


class SentencePipeline:
    def __init__(self, model_path, stop_rank=1200, use_dict=True):
        self.use_dict = use_dict
        from mlx_lm import load
        self.model, self.tok = load(os.path.abspath(model_path))
        self.lex = sqlite3.connect(os.path.abspath(LEX))
        self.former = Former()
        self.dict = Dictionary()
        self.lem = Lemmatizer()
        self.genders, _ = load_db(os.path.join(ROOT, "glossary.db"))
        db = sqlite3.connect(os.path.join(ROOT, "glossary.db"))
        # Step 3's A2+ bar is the corpus's own: the stoplist is every form the
        # first three books declined to gloss even once.
        self.stop = {r[0] for r in db.execute(
            "SELECT word FROM stoplist WHERE rank <= ?", (stop_rank,))}
        db.close()
        self.stats = collections.Counter()

    # -- step 2 -------------------------------------------------------------
    def is_german_word(self, w):
        """Real German, not dialect, OCR noise or a clipped fragment.

        Dropped-h dialect ("'Ermine", "ätten") and transcribed accents produce tokens
        that are not words, and the corpus notes they simply have to be
        skipped. Wiktionary membership decides it without a model.
        """
        if len(w) < 3 or not w.isalpha():
            return False
        for cand in (w.lower(), self.lem.of(w)):
            if self.lex.execute("SELECT 1 FROM lemma WHERE word=? LIMIT 1",
                                (cand,)).fetchone():
                return True
            if self.lex.execute("SELECT 1 FROM forms WHERE surface=? LIMIT 1",
                                (cand,)).fetchone():
                return True
        # Wiktionary's tables do not hold every inflected form, and they hold
        # no novel compound at all. Membership alone therefore threw away
        # 'unterhielten', 'hinabspähte' and 'Sockenpaar' as if they were noise.
        # Two cheap recoveries keep those while still rejecting 'Freds' and
        # 'Durmstrangs': a stem that lands on a known lemma, and a compound
        # whose head is a known noun.
        if self.former.form(w.lower(), None):
            return True
        if w[:1].isupper():
            from stage_form import _charsplit
            for score, mod, tail in _charsplit(w[0].upper() + w[1:]):
                if score >= 0 and self.lex.execute(
                        "SELECT 1 FROM lemma WHERE word=? AND pos='noun' LIMIT 1",
                        (tail.lower(),)).fetchone():
                    return True
        return False

    def translate(self, text, src, tgt, n=64):
        msg = [{"role": "user", "content": [{
            "type": "text", "source_lang_code": src,
            "target_lang_code": tgt, "text": text}]}]
        from mlx_lm import generate
        return clean(generate(self.model, self.tok,
                              prompt=self.tok.apply_chat_template(
                                  msg, add_generation_prompt=True),
                              max_tokens=n, verbose=False))

    # -- step 9 -------------------------------------------------------------
    def leaks_german(self, hu):
        """The 'Hungarian' is really German — the model declined to translate."""
        for w in SPLIT.split(hu):
            if len(w) < 4:
                continue
            if self.lex.execute("SELECT 1 FROM lemma WHERE word=? LIMIT 1",
                                (w.lower(),)).fetchone():
                return True
        return False

    def run_sentence(self, text, candidates):
        s = self.stats
        # 1-5: everything decidable without a model
        prepared = []
        for w in candidates:
            s["candidates"] += 1
            if not self.is_german_word(w):
                s["drop_not_a_word"] += 1
                continue
            if w.lower() in self.stop:
                s["drop_below_a2"] += 1
                continue
            lemma = self.lem.of(w)
            form = self.former.form(lemma, None)
            if not form:
                s["drop_no_form"] += 1
                continue
            prepared.append({"surface": w, "lemma": lemma, "term": form[0]})
        if not prepared:
            return []

        # 6a: the dictionary first. It cannot invent a meaning, so where it
        # answers there is nothing for a model to improve on — generation is
        # only for the 45% of candidates it has never heard of.
        for p in prepared:
            look = self.dict.look(p["lemma"]) if self.use_dict else None
            p["dict"] = list(look[1][:3]) if look and look[1] else []

        # 6: path A — each lemma on its own. Translating them as one
        # newline-separated list loses the alignment: the model does not
        # preserve the line count, so answers silently shift onto the wrong
        # words and 42 of 136 candidates came back with no translation.
        for p in prepared:
            p["path_a"] = ("" if p["dict"]
                           else self.translate(p["lemma"], "de", "hu", n=24))

        # 7: path B — the whole sentence
        sent_hu = self.translate(text, "de", "hu", n=200)
        st = hu_stems(list(hu_tokens(sent_hu)) +
                      [t for p in prepared for t in hu_tokens(p.get("path_a", ""))])
        sent_keys = set()
        for t in hu_tokens(sent_hu):
            sent_keys |= st.get(t, {t}) | {t, t[:5]}

        bad = self.spell_bad([p.get("path_a", "") for p in prepared]
                             + [x for p in prepared for x in p["dict"]])

        out = []
        for p in prepared:
            source = "dict" if p["dict"] else "model"
            if p["dict"]:
                # The sentence translation still gets a vote: it picks which of
                # the dictionary's senses fits here. If none match, keep the
                # first rather than dropping — the dictionary is trustworthy
                # about meaning even when the ranker is unsure.
                ranked = [x for x in p["dict"]
                          if {k for t in hu_tokens(x)
                              for k in (st.get(t, {t}) | {t, t[:5]})} & sent_keys]
                hu = ", ".join((ranked or p["dict"])[:2])
                s["kept_dict_ranked" if ranked else "kept_dict_first"] += 1
            else:
                hu = p.get("path_a", "")
                if not hu:
                    s["drop_no_translation"] += 1
                    continue
                # 8: the two paths must agree — generated Hungarian only.
                keys = set()
                for t in hu_tokens(hu):
                    keys |= st.get(t, {t}) | {t, t[:5]}
                if not (keys & sent_keys):
                    s["drop_paths_disagree"] += 1
                    continue
            # 9: leakage and spelling
            if self.leaks_german(hu):
                s["drop_german_leak"] += 1
                continue
            if any(w in bad for w in SPLIT.split(hu) if len(w) > 2):
                s["drop_misspelled"] += 1
                continue
            if non_latin(hu) or non_latin(p["term"]):
                s["drop_charset"] += 1
                continue
            # 10: grammar of the citation form
            if form_check(unicodedata.normalize("NFC", p["term"]), self.genders):
                s["drop_bad_form"] += 1
                continue
            s["kept"] += 1
            s["kept_" + source] += 1
            out.append({"term": p["term"], "hu": hu, "surface": p["surface"],
                        "source": source})
        return out

    def spell_bad(self, fields):
        words = sorted({w for f in fields for w in SPLIT.split(f or "")
                        if len(w) > 2})
        if not words or not os.path.exists(os.path.abspath(HUDICT) + ".dic"):
            return set()
        p = subprocess.run(["hunspell", "-d", os.path.abspath(HUDICT), "-l"],
                           input="\n".join(words), capture_output=True, text=True)
        return {w for w in p.stdout.split("\n") if w.strip()}


def hu_agree(a, b):
    """Same Hungarian word, allowing for inflection."""
    ta, tb = hu_tokens(a), hu_tokens(b)
    if ta & tb:
        return True
    st = hu_stems(list(ta | tb))
    for x in ta:
        kx = st.get(x, {x}) | {x}
        for y in tb:
            if kx & (st.get(y, {y}) | {y}):
                return True
            if len(x) >= 5 and len(y) >= 5 and x[:5] == y[:5]:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(
        HERE, "..", "models", "translategemma-4b-it-4bit"))
    ap.add_argument("--limit", type=int, default=40, help="sentences")
    ap.add_argument("--label", default="4b")
    ap.add_argument("--no-dict", action="store_true",
                    help="generation only, for the A/B against the combination")
    args = ap.parse_args()

    pipe = SentencePipeline(args.model, use_dict=not args.no_dict)
    tasks = [json.loads(l) for l in
             open(os.path.join(HERE, "..", "goldset", "tasks.jsonl"), encoding="utf-8")]

    cases = []
    for t in tasks:
        gold = t["gold"]
        for sent in t["sentences"]:
            if str(sent["i"]) in gold and sent["new"]:
                cases.append((sent["text"], sent["new"], gold[str(sent["i"])]))
    cases = cases[:args.limit]

    hit = total = 0
    rows, unscored = [], []
    for text, cands, gold in cases:
        got = pipe.run_sentence(text, cands)
        # Score only the words the corpus also chose to gloss. Comparing an
        # entry for 'Bruder' against the corpus's entry for some other word in
        # the same sentence marks a correct translation wrong, which understated
        # precision badly on the first pass.
        by_head = {head(g[0]): g for g in gold}
        for e in got:
            g = by_head.get(head(e["term"])) or by_head.get(e["surface"].lower())
            if not g:
                unscored.append(e)
                continue
            total += 1
            # Hungarian is agglutinative: 'elkapni' and 'elkap' are the same
            # answer. Exact token equality marked those wrong and understated
            # precision, so compare stems.
            ok = hu_agree(e["hu"], g[2])
            hit += ok
            rows.append({**e, "ok": ok, "gold": g[2]})

    s = pipe.stats
    print("sentences %d   candidates %d\n" % (len(cases), s["candidates"]))
    print("  dropped, not a German word     %4d" % s["drop_not_a_word"])
    print("  dropped, below A2+             %4d" % s["drop_below_a2"])
    print("  dropped, no citation form      %4d" % s["drop_no_form"])
    print("  dropped, no translation        %4d" % s["drop_no_translation"])
    print("  dropped, PATHS DISAGREE        %4d   <- the new check" % s["drop_paths_disagree"])
    print("  dropped, German leaked through %4d" % s["drop_german_leak"])
    print("  dropped, misspelled Hungarian  %4d" % s["drop_misspelled"])
    print("  dropped, bad citation form     %4d" % s["drop_bad_form"])
    print("  KEPT                           %4d   (dictionary %d, model %d)" % (
        s["kept"], s["kept_dict"], s["kept_model"]))
    print("\n  scoreable (the corpus glossed the same word): %d" % total)
    print("  agree with the corpus: %d/%d  (%.1f%%)  <- PRECISION" % (
        hit, total, 100.0 * hit / (total or 1)))
    print("  produced but the corpus glossed a different word: %d (unscoreable,"
          " not wrong)" % len(unscored))
    print("\n  scored sample:")
    for r in rows[:14]:
        print("    %-32s %-20s gold %-18s %s" % (
            r["term"][:32], r["hu"][:20], str(r["gold"])[:18],
            "ok" if r["ok"] else "differs"))
    print("\n  unscoreable sample (corpus chose other words in that sentence):")
    for r in unscored[:8]:
        print("    %-32s %s" % (r["term"][:32], r["hu"][:24]))

    json.dump(rows, open(os.path.join(HERE, "..", "results",
                                      "sentence-%s.json" % args.label),
                         "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
