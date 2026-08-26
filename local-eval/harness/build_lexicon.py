#!/usr/bin/env python3
"""Boil the German Wiktionary extract down to a citation-form lexicon.

Tests the hypothesis that sub-task B is a lookup, not a generation problem.
Article, plural and the verb's principal parts are facts about the language;
a 4B model guesses them, and a wrong guess is worse than a missing entry
because it becomes the lemma an entry is keyed under forever.

Reads kaikki's dewiktionary raw JSONL (the German edition, so the definitions
are already German and the translation list already carries Hungarian) and
writes lexicon.db:

    forms   surface form   -> lemma            (so 'Ladenfronten' finds 'Front')
    lemma   lemma          -> term, pos, de, hu

    python3 build_lexicon.py --src ../models/lexicon/dewikt.jsonl.gz
"""

import argparse
import gzip
import json
import os
import re
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
ARTICLE_OF = {"das": "das", "die": "die", "der": "der",
              "des": "der", "dem": "der", "den": "die"}


def plural_marker(sing, plur):
    """Render the plural the way the corpus writes it: -n, -e, -̈er, - ."""
    if not plur or plur == sing:
        return "-"
    umlaut = ""
    base, rest = sing, plur
    if not plur.startswith(sing):
        # Try undoing an umlaut on the stem: Anlass -> Anlässe
        for a, b in (("a", "ä"), ("o", "ö"), ("u", "ü"), ("au", "äu")):
            cand = sing.replace(a, b) if a in sing else None
            if cand and plur.startswith(cand):
                umlaut, base = "̈", cand
                break
        else:
            return plur                     # irregular; give the whole word
    rest = plur[len(base):]
    return "-" + umlaut + rest if (umlaut or rest) else "-"


def noun_term(word, forms):
    art = plur = None
    for f in forms:
        tags = f.get("tags") or []
        if "nominative" in tags and "singular" in tags:
            art = art or ARTICLE_OF.get((f.get("article") or "").lower())
        if "nominative" in tags and "plural" in tags:
            plur = plur or f.get("form")
    if not art:
        return None
    if not plur or plur in ("—", "-", "Plural"):
        return "%s %s" % (art, word)        # genuine uncountable
    return "%s %s, %s" % (art, word, plural_marker(word, plur))


def verb_term(word, forms):
    """infinitiv, präteritum, hat/ist partizip — the README's verb shape.

    dewiktionary tags the preterite as plain ["past"] plus a pronoun, not
    ["preterite","third","singular"], and names the auxiliary by its infinitive
    in a ["auxiliary","perfect"] row. Separable verbs already come out split
    ("sammelte ein"), which is the particle placement the README asks for.
    """
    prat = part = aux = None
    for f in forms:
        tags = set(f.get("tags") or [])
        pron = set(f.get("pronouns") or [])
        form = (f.get("form") or "").strip()
        if not form or form in ("—", "-") or form.endswith("!"):
            continue
        if "past" in tags and "subjunctive-ii" not in tags and pron & {"ich", "er"}:
            prat = prat or form
        if "participle-2" in tags:
            part = part or form
        if "auxiliary" in tags:
            low = form.lower()
            if low == "sein":
                aux = "ist"
            elif low == "haben" and aux is None:
                aux = "hat"
    if not (prat and part):
        return None
    return "%s, %s, %s %s" % (word, prat, aux or "hat", part)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(HERE, "..", "models", "lexicon",
                                                  "dewikt.jsonl.gz"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "models", "lexicon",
                                                  "lexicon.db"))
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    if os.path.exists(out):
        os.remove(out)
    db = sqlite3.connect(out)
    db.executescript("""
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE lemma (word TEXT, pos TEXT, term TEXT, de TEXT, hu TEXT);
        CREATE TABLE forms (surface TEXT, word TEXT);
    """)

    op = gzip.open if args.src.endswith(".gz") else open
    n = kept = 0
    lem, frm = [], []
    with op(args.src, "rt", encoding="utf-8") as fh:
        for line in fh:
            n += 1
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("lang_code") != "de":
                continue
            word, pos = d.get("word"), d.get("pos")
            if not word or not pos:
                continue
            forms = d.get("forms") or []
            term = (noun_term(word, forms) if pos == "noun" else
                    verb_term(word, forms) if pos == "verb" else word)
            if not term:
                continue

            senses = d.get("senses") or []
            de = ""
            for s in senses:
                g = s.get("glosses") or []
                if g:
                    de = g[0]
                    break
            hu = ""
            for t in d.get("translations") or []:
                if (t.get("lang_code") or t.get("code")) == "hu" and t.get("word"):
                    hu = t["word"]
                    break

            lem.append((word.lower(), pos, term, de, hu))
            kept += 1
            seen = {word.lower()}
            for f in forms:
                s = (f.get("form") or "").strip().lower()
                if s and s not in seen and " " not in s and len(s) > 2:
                    seen.add(s)
                    frm.append((s, word.lower()))

            if len(lem) > 20000:
                db.executemany("INSERT INTO lemma VALUES (?,?,?,?,?)", lem)
                db.executemany("INSERT INTO forms VALUES (?,?)", frm)
                lem, frm = [], []

    db.executemany("INSERT INTO lemma VALUES (?,?,?,?,?)", lem)
    db.executemany("INSERT INTO forms VALUES (?,?)", frm)
    db.executescript("""
        CREATE INDEX lemma_word ON lemma (word);
        CREATE INDEX forms_surface ON forms (surface);
    """)
    db.commit()
    hu_n = db.execute("SELECT count(*) FROM lemma WHERE hu <> ''").fetchone()[0]
    de_n = db.execute("SELECT count(*) FROM lemma WHERE de <> ''").fetchone()[0]
    f_n = db.execute("SELECT count(*) FROM forms").fetchone()[0]
    print("read %d lines, kept %d lemmas, %d surface forms" % (n, kept, f_n))
    print("  with a German gloss: %d (%.0f%%)" % (de_n, 100.0 * de_n / kept))
    print("  with a Hungarian translation: %d (%.0f%%)" % (hu_n, 100.0 * hu_n / kept))
    print("  wrote %s (%.0f MB)" % (out, os.path.getsize(out) / 1e6))
    db.close()


if __name__ == "__main__":
    main()
