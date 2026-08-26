#!/usr/bin/env python3
"""Build the Reader glossary database from the already-glossed chapter JSONs.

Every vocab entry the reader has ever produced becomes a row keyed by lemma, so a
new book only has to be glossed for the words that aren't in here yet.

    python3 build_reader_glossary.py [--root <dir>]... [--db glossary.db]

Defaults to the finished books plus results/, with results/ winning.
"""

import argparse
import glob
import json
import os
import re
import sqlite3
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths

ARTICLES = ("der", "die", "das")


def lemma_of(term):
    """Collapse a display term to the key we look words up by.

    'das Wirtshaus, -"er'                  -> N:wirtshaus
    'sich anstellen, stellte sich an, ...' -> W:sich anstellen
    'starren (auf etw.)'                   -> W:starren
    """
    t = unicodedata.normalize("NFC", term).strip()
    m = re.match(r"^(der|die|das)\s+([^,(]+)", t, re.I)
    if m:
        return "N:" + re.sub(r"\s+", " ", m.group(2)).strip().lower()
    head = t.split(",")[0]
    head = re.sub(r"\s*\([^)]*\)", "", head)
    return "W:" + re.sub(r"\s+", " ", head).strip().lower()


def parse_term(term):
    """Split a display term into (pos, gender, plural, forms)."""
    t = unicodedata.normalize("NFC", term).strip()
    m = re.match(r"^(der|die|das)\s+(.+)$", t, re.I)
    if m:
        gender = m.group(1).lower()
        rest = m.group(2)
        plural = None
        if "," in rest:
            plural = rest.split(",", 1)[1].strip() or None
        return "noun", gender, plural, None
    parts = [p.strip() for p in t.split(",")]
    if len(parts) >= 3:
        return "verb", None, None, ", ".join(parts[1:])
    if " " in re.sub(r"\s*\([^)]*\)", "", t).strip():
        return "phrase", None, None, None
    return "other", None, None, None


SCHEMA = """
CREATE TABLE entries (
    id      INTEGER PRIMARY KEY,
    lemma   TEXT NOT NULL,
    pos     TEXT NOT NULL,
    term    TEXT NOT NULL,
    gender  TEXT,
    plural  TEXT,
    forms   TEXT,
    de      TEXT NOT NULL,
    hu      TEXT NOT NULL,
    freq    INTEGER NOT NULL DEFAULT 0,
    UNIQUE (term, de, hu)
);
CREATE INDEX entries_lemma ON entries (lemma);
CREATE INDEX entries_pos   ON entries (pos);

CREATE TABLE occurrences (
    entry_id INTEGER NOT NULL REFERENCES entries (id),
    book     TEXT NOT NULL,
    chapter  TEXT NOT NULL,
    n        INTEGER NOT NULL,
    PRIMARY KEY (entry_id, book, chapter)
);

-- Word forms that occur in the text but were never once glossed, i.e. words
-- previously judged to sit below the A2+ bar. Ranked by corpus frequency: the
-- head is function words and is safe to skip outright, but the tail picks up
-- proper nouns and inflected forms of words that ARE glossed elsewhere, so past
-- a few hundred ranks treat this as a hint rather than a verdict.
CREATE TABLE stoplist (
    word TEXT PRIMARY KEY,
    n    INTEGER NOT NULL,
    rank INTEGER NOT NULL
);
CREATE INDEX stoplist_rank ON stoplist (rank);
"""

WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")
STOPLIST_KEEP = 2000


def build(roots, db_path):
    if os.path.exists(db_path):
        os.remove(db_path)
    db = sqlite3.connect(db_path)
    db.executescript(SCHEMA)

    entry_ids = {}
    counts = {}
    word_freq = {}

    # A chapter can exist in more than one root — finished on the server, or freshly
    # merged under results/. Later roots win, so a new merge supersedes the copy
    # already shipped.
    by_chapter = {}
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "*", "ch*_glossed.json"))):
            key = (os.path.basename(os.path.dirname(path)), os.path.basename(path))
            by_chapter[key] = path
    files = [by_chapter[k] for k in sorted(by_chapter)]
    if not files:
        raise SystemExit("no *_glossed.json under %s" % ", ".join(roots))

    for path in files:
        book = os.path.basename(os.path.dirname(path))
        chapter = os.path.basename(path).split("_")[0]
        chapter_data = json.load(open(path, encoding="utf-8"))
        for para in chapter_data["paragraphs"]:
            for sent in para:
                for word in WORD_RE.findall(
                    unicodedata.normalize("NFC", sent["sentence"])
                ):
                    word = word.lower()
                    word_freq[word] = word_freq.get(word, 0) + 1
                for term, de, hu in sent["vocab"]:
                    term = unicodedata.normalize("NFC", term).strip()
                    de = de.strip()
                    hu = hu.strip()
                    key = (term, de, hu)
                    eid = entry_ids.get(key)
                    if eid is None:
                        pos, gender, plural, forms = parse_term(term)
                        cur = db.execute(
                            "INSERT INTO entries "
                            "(lemma, pos, term, gender, plural, forms, de, hu) "
                            "VALUES (?,?,?,?,?,?,?,?)",
                            (lemma_of(term), pos, term, gender, plural, forms, de, hu),
                        )
                        eid = cur.lastrowid
                        entry_ids[key] = eid
                    counts[(eid, book, chapter)] = counts.get((eid, book, chapter), 0) + 1

    db.executemany(
        "INSERT INTO occurrences (entry_id, book, chapter, n) VALUES (?,?,?,?)",
        [(e, b, c, n) for (e, b, c), n in counts.items()],
    )
    db.execute(
        "UPDATE entries SET freq = "
        "(SELECT COALESCE(SUM(n), 0) FROM occurrences WHERE entry_id = entries.id)"
    )

    glossed = {
        row[0].split(":", 1)[1]
        for row in db.execute("SELECT DISTINCT lemma FROM entries")
    }
    never = sorted(
        ((w, n) for w, n in word_freq.items() if w not in glossed),
        key=lambda wn: (-wn[1], wn[0]),
    )[:STOPLIST_KEEP]
    db.executemany(
        "INSERT INTO stoplist (word, n, rank) VALUES (?,?,?)",
        [(w, n, i + 1) for i, (w, n) in enumerate(never)],
    )
    db.commit()

    stats = {
        "files": len(files),
        "entries": db.execute("SELECT COUNT(*) FROM entries").fetchone()[0],
        "lemmas": db.execute("SELECT COUNT(DISTINCT lemma) FROM entries").fetchone()[0],
        "occurrences": db.execute("SELECT COALESCE(SUM(n), 0) FROM occurrences").fetchone()[0],
        "stoplist": db.execute("SELECT COUNT(*) FROM stoplist").fetchone()[0],
    }
    db.close()
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=None,
                    help="repeatable; defaults to the live server dir then results/")
    ap.add_argument("--db", default=paths.DB)
    args = ap.parse_args()
    roots = args.root or [paths.LIVE, paths.RESULTS]
    roots = [r for r in roots if os.path.isdir(r)]
    s = build(roots, args.db)
    print("roots: %s" % ", ".join(roots))
    db_path = args.db
    print("built %s" % db_path)
    for k, v in s.items():
        print("  %-12s %d" % (k, v))
