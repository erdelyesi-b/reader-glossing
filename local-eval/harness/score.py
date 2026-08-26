#!/usr/bin/env python3
"""Grade a run against the gold set, keeping the three sub-tasks apart.

Glossing looks like one job but it is three, and they fail independently:

  A  SELECT     which candidates deserve an entry at all — judgment
  B  FORM       the citation form: article + plural, or the verb's three parts
                with hat/ist — morphology, and a lookup could do it perfectly
  C  CONTENT    the German definition and the Hungarian translation

A single quality number would hide the finding that matters: if a model scores
well on A and C but badly on B, the answer is not a bigger model, it is taking
B away from the model. If it fails C, no amount of scaffolding helps, because
that is where Hungarian lives.

    python3 score.py runs/*.jsonl
"""

import argparse
import collections
import glob
import json
import os
import re
import sqlite3
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from german import lemma_keys  # noqa: E402

ARTICLES = ("der ", "die ", "das ")
AUX = ("hat ", "ist ")
WORD = re.compile(r"[A-Za-zÄÖÜäöüß]+")
# Corpus medians from glossary.db: de averages 25.9 chars, hu 13.2. Twice the
# mean is the line where an entry stops reading like the rest of the book.
DE_MAX, HU_MAX = 52, 30


# ---------------------------------------------------------------- normalising

def head(term):
    """The word an entry is about, for matching model entries against gold."""
    t = unicodedata.normalize("NFC", (term or "").strip())
    low = t.lower()
    for a in ARTICLES:
        if low.startswith(a):
            t = t[len(a):]
            break
    t = t.split(",")[0].strip()          # drop plural / principal parts
    if t.lower().startswith("sich "):
        t = t[5:]
    return t.lower()


def keyset(term):
    """Every lemma key a term could match under, so 'zuziehen' meets 'zuzieht'.

    Includes the head's first word, because the corpus and a model routinely
    disagree on whether the government belongs in the term: 'auskommen mit,
    kam aus, ist ausgekommen' and 'auskommen, kam aus, ist ausgekommen' are the
    same decision about the same word, and scoring them as a miss would
    manufacture disagreement that is not there.
    """
    h = head(term)
    if not h:
        return set()
    keys = {h} | {k.split(":", 1)[1] for k in lemma_keys(h)}
    first = h.split()[0] if " " in h else ""
    if len(first) > 3:
        keys |= {first} | {k.split(":", 1)[1] for k in lemma_keys(first)}
    return keys


def match(model_terms, gold_terms):
    """Greedy pairing of model entries to gold entries by lemma overlap."""
    pairs, used = [], set()
    for mi, m in enumerate(model_terms):
        mk = keyset(m)
        for gi, g in enumerate(gold_terms):
            if gi in used:
                continue
            if mk & keyset(g):
                pairs.append((mi, gi))
                used.add(gi)
                break
    return pairs


# ------------------------------------------------------------------ sub-tasks

def classify(term):
    low = (term or "").lower()
    if low.startswith(ARTICLES):
        return "noun"
    if " " in low and any(a in low for a in AUX):
        return "verb"
    if low.count(",") >= 2:
        return "verb"                     # three parts, auxiliary missing
    if " " in (term or "").strip():
        return "phrase"
    return "other"


def form_check(term, genders):
    """What is structurally wrong with this citation form. Empty == clean."""
    bad = []
    t = unicodedata.normalize("NFC", (term or "").strip())
    if not t:
        return ["empty"]
    kind = classify(t)
    if kind == "noun":
        art = t.split()[0].lower()
        if "," not in t:
            bad.append("noun-no-plural")
        want = genders.get(head(t))
        if want and want != art:
            bad.append("noun-wrong-article")
    elif kind == "verb":
        parts = [p.strip() for p in t.split(",")]
        if len(parts) < 3:
            bad.append("verb-missing-parts")
        if not any(a.strip() in t.lower().split() for a in AUX):
            bad.append("verb-no-aux")
    elif kind == "other" and t[:1].isupper() and len(t) > 2:
        # A capitalised bare word in German is a noun that lost its article.
        bad.append("noun-no-article")
    return bad


def content_check(entry):
    term, de, hu = (list(entry) + ["", "", ""])[:3]
    bad = []
    if not de.strip():
        bad.append("de-empty")
    if not hu.strip():
        bad.append("hu-empty")
    if len(de) > DE_MAX:
        bad.append("de-too-long")
    if len(hu) > HU_MAX:
        bad.append("hu-too-long")
    if de.strip().endswith("."):
        bad.append("de-is-sentence")
    # README §4: don't define a word with itself. Matched on whole words, not
    # substrings — 'gehen' inside 'vergehen' is a different verb, and counting
    # it flagged a fifth of the frontier corpus for nothing.
    h = head(term)
    if h and len(h) > 3 and h in {w.lower() for w in WORD.findall(de)}:
        bad.append("de-repeats-headword")
    return bad


def non_latin(field):
    """The homoglyph guard from README §5, which is a hard ship-blocker."""
    return [c for c in field or ""
            if c.isalpha() and not unicodedata.name(c, "").startswith("LATIN")]


def hu_tokens(s):
    return {w for w in re.split(r"[,;/()\s]+", (s or "").lower()) if len(w) > 2}


# --------------------------------------------------------------------- report

def load_forms(lex_path):
    """surface form -> lemma, from Wiktionary's inflection tables.

    german.py cannot get 'brachten' to 'bringen' — it strips suffixes, and
    strong verbs change their stem. Without this the on-candidate measure
    silently fails on exactly the verbs worth glossing.
    """
    forms = {}
    if not os.path.exists(lex_path):
        return forms
    db = sqlite3.connect(lex_path)
    for surface, word in db.execute("SELECT surface, word FROM forms"):
        forms.setdefault(surface, word)
    for (word,) in db.execute("SELECT DISTINCT word FROM lemma"):
        forms.setdefault(word, word)
    db.close()
    return forms


def load_db(db_path):
    """Gender per noun lemma, and every Hungarian gloss the corpus has per lemma."""
    genders, hu = {}, collections.defaultdict(set)
    if not os.path.exists(db_path):
        return genders, hu
    db = sqlite3.connect(db_path)
    for lemma, gender, term, h in db.execute(
            "SELECT lemma, gender, term, hu FROM entries"):
        key = lemma.split(":", 1)[1] if ":" in lemma else lemma
        if gender:
            genders[key] = gender.strip().lower()
            genders[head(term)] = gender.strip().lower()
        hu[key] |= hu_tokens(h)
        hu[head(term)] |= hu_tokens(h)
    db.close()
    return genders, hu


def score_run(path, tasks, genders, hudb, forms):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    s = collections.Counter()
    form_bad = collections.Counter()
    cont_bad = collections.Counter()
    secs = 0.0
    hu_hit = hu_seen = 0

    for row in rows:
        task = tasks[row["id"]]
        secs += row.get("seconds") or 0
        s["tasks"] += 1
        s["parse_" + (row.get("parse") or "none")] += 1
        if row.get("error"):
            s["errors"] += 1
        ents = row.get("entries")
        gold_all = task["gold"]
        s["gold"] += sum(len(v) for v in gold_all.values())
        if not isinstance(ents, dict):
            continue
        s["parsed_tasks"] += 1

        seen_terms = set()
        for k, gold in gold_all.items():
            got = ents.get(k) or ents.get(str(k)) or []
            if not isinstance(got, list):
                s["malformed"] += 1
                continue
            clean = [e for e in got
                     if isinstance(e, (list, tuple)) and len(e) >= 3
                     and all(isinstance(x, str) for x in e[:3])]
            s["malformed"] += len(got) - len(clean)
            s["model"] += len(clean)

            # A — selection
            for mi, gi in match([e[0] for e in clean], [g[0] for g in gold]):
                s["overlap"] += 1

            # Does the entry answer a word the batch actually offered? A:prec
            # alone punishes a model for not guessing which off-list idiom the
            # frontier run chose to add, which is not a quality difference.
            cand = set()
            for sent in task["sentences"]:
                if str(sent["i"]) == str(k):
                    for w in sent["new"]:
                        cand |= keyset(w)
                        lem = forms.get(w.lower())
                        if lem:
                            cand |= keyset(lem)
            for e in clean:
                ks = keyset(e[0])
                lem = forms.get(head(e[0]))
                if lem:
                    ks |= keyset(lem)
                if cand & ks:
                    s["on_candidate"] += 1

            for e in clean:
                term, de, hu = e[0], e[1], e[2]
                # ship-blockers first: these are what README §5 refuses to merge
                if non_latin(term) or non_latin(de) or non_latin(hu):
                    s["charset_fail"] += 1
                h = head(term)
                if h in seen_terms:
                    s["dup_term"] += 1
                seen_terms.add(h)

                # B — citation form
                bad = form_check(term, genders)
                if bad:
                    s["form_fail"] += 1
                    for b in bad:
                        form_bad[b] += 1

                # C — content
                cbad = content_check(e)
                if cbad:
                    s["content_fail"] += 1
                    for b in cbad:
                        cont_bad[b] += 1
                s["de_len"] += len(de)
                s["hu_len"] += len(hu)

                # C — Hungarian, against every gloss the corpus has for the lemma.
                # Only scoreable where the corpus knows the word, but that is
                # 50k entries deep, so it is the closest thing to a reference.
                ref = set()
                for key in {h} | {kk.split(":", 1)[1] for kk in lemma_keys(h)}:
                    ref |= hudb.get(key, set())
                if ref:
                    hu_seen += 1
                    if hu_tokens(hu) & ref:
                        hu_hit += 1

        # any entries the model put on sentences the gold left alone
        extra = set(ents) - set(gold_all)
        s["off_target_sentences"] += len(extra)

    s["hu_hit"], s["hu_seen"] = hu_hit, hu_seen
    s["seconds"] = secs
    return s, form_bad, cont_bad


def pct(a, b):
    return "%5.1f%%" % (100.0 * a / b) if b else "    - "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tasks", default=os.path.join(HERE, "..", "goldset", "tasks.jsonl"))
    ap.add_argument("--db", default=os.path.join(ROOT, "glossary.db"))
    ap.add_argument("--detail", action="store_true", help="break out failure kinds")
    args = ap.parse_args()

    tasks = {t["id"]: t for t in
             (json.loads(l) for l in open(args.tasks, encoding="utf-8"))}
    genders, hudb = load_db(args.db)
    forms = load_forms(os.path.abspath(os.path.join(
        HERE, "..", "models", "lexicon", "lexicon.db")))

    paths = []
    for p in args.runs:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?") else [p])

    print("%-22s %6s %7s %7s  | %7s %7s %8s | %7s %7s | %6s" % (
        "run", "tasks", "JSONok", "entries", "A:prec", "A:rec", "A:onCand",
        "B:form", "C:hu", "s/task"))
    print("-" * 110)
    results = []
    for p in paths:
        if not os.path.exists(p):
            continue
        s, fb, cb = score_run(p, tasks, genders, hudb, forms)
        prec = s["overlap"] / s["model"] if s["model"] else 0
        rec = s["overlap"] / s["gold"] if s["gold"] else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        formok = 1 - (s["form_fail"] / s["model"]) if s["model"] else 0
        print("%-22s %6d %7s %7d  | %6.1f%% %6.1f%% %7s | %6.1f%% %7s | %6.1f" % (
            os.path.basename(p)[:-6], s["tasks"],
            pct(s["parsed_tasks"], s["tasks"]), s["model"],
            100 * prec, 100 * rec, pct(s["on_candidate"], s["model"]),
            100 * formok, pct(s["hu_hit"], s["hu_seen"]),
            s["seconds"] / s["tasks"] if s["tasks"] else 0))
        results.append((p, s, fb, cb))

    for p, s, fb, cb in results:
        print("\n== %s" % os.path.basename(p))
        print("   parse:      " + "  ".join(
            "%s=%d" % (k[6:], v) for k, v in sorted(s.items())
            if k.startswith("parse_")))
        print("   entries:    model %d vs gold %d   (%.2f per sentence, gold %.2f)" % (
            s["model"], s["gold"], s["model"] / (s["tasks"] * 10 or 1),
            s["gold"] / (s["tasks"] * 10 or 1)))
        print("   ship-blockers: charset %d   duplicate term %d   malformed entry %d"
              % (s["charset_fail"], s["dup_term"], s["malformed"]))
        print("   lengths:    de %.1f chars (corpus 25.9)   hu %.1f (corpus 13.2)" % (
            s["de_len"] / s["model"] if s["model"] else 0,
            s["hu_len"] / s["model"] if s["model"] else 0))
        if args.detail:
            if fb:
                print("   B failures: " + "  ".join("%s=%d" % kv for kv in fb.most_common()))
            if cb:
                print("   C failures: " + "  ".join("%s=%d" % kv for kv in cb.most_common()))


if __name__ == "__main__":
    main()
