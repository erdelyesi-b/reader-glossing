#!/usr/bin/env python3
"""Reassemble a chapter's glossed JSON from DB hits plus the model's new entries.

The model only ever returns entries for words the DB didn't already know, keyed by
sentence index — it never echoes the sentences back. This script re-attaches
everything and writes the file the app actually loads.

Model output (one file per batch, glossed_NNN.json):
    {"entries": {"<sentence index>": [[term, de, hu], ...], ...}}

    python3 merge_glossed.py --chapter ".../ch01.json" --work ./work/book
"""

import argparse
import glob
import json
import os
import sqlite3
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths


def non_latin(field):
    """Letters from outside the Latin block, which German and Hungarian never use.

    Homoglyphs -- Cyrillic 'а' for Latin 'a', Arabic seen for 's' -- are invisible
    on screen and survive every other check here: the JSON parses, the entry has
    its three strings, the vocab-per-sentence ratio is unmoved. They reach the
    reader as garbled text, and when one lands in a term it corrupts that entry's
    lemma, so the word can never be matched again. Thirteen had already made it
    into the corpus before this ran.

    Combining marks pass: the umlaut-plural notation (-"e) is not a letter.
    """
    return [c for c in field
            if c.isalpha() and not unicodedata.name(c, "").startswith("LATIN")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapter", required=True)
    ap.add_argument("--work", default=paths.WORK)
    ap.add_argument("--out", default=None)
    ap.add_argument("--allow-partial", action="store_true",
                    help="write even if some batches have no model output yet")
    ap.add_argument("--max-db-vocab", type=int, default=2,
                    help="cap on entries re-used from the DB per sentence; the "
                         "model's new entries are always kept. Existing corpus "
                         "averages 2.0 vocab/sentence — attaching every known "
                         "word instead buries the page in glosses.")
    ap.add_argument("--db", default=paths.DB)
    args = ap.parse_args()

    chapter = json.load(open(args.chapter, encoding="utf-8"))
    name = os.path.basename(args.chapter).split(".")[0]
    book = os.path.basename(os.path.dirname(os.path.abspath(args.chapter)))
    workdir = os.path.join(args.work, book, name)

    known = json.load(open(os.path.join(workdir, "known.json"), encoding="utf-8"))

    # Rarer words earn their place on the page; the commonest re-used entries are
    # the ones a reader at this point least needs explained again.
    freq = {}
    if os.path.exists(args.db):
        con = sqlite3.connect(args.db)
        freq = {t: f for t, f in con.execute("SELECT term, freq FROM entries")}
        con.close()
    for idx, entries in known.items():
        entries.sort(key=lambda e: freq.get(e[0], 0))
        known[idx] = entries[: args.max_db_vocab]

    n_batches = len(glob.glob(os.path.join(workdir, "batch_*.json")))
    produced = sorted(glob.glob(os.path.join(workdir, "glossed_*.json")))
    if len(produced) < n_batches and not args.allow_partial:
        raise SystemExit(
            "only %d of %d batches glossed; pass --allow-partial to write anyway"
            % (len(produced), n_batches)
        )

    new = {}
    offenders = []
    seen_terms = {}
    repeats = []
    for path in produced:
        payload = json.load(open(path, encoding="utf-8"))
        for idx, entries in payload["entries"].items():
            for entry in entries:
                for field, value in zip(("term", "de", "hu"), entry):
                    stray = non_latin(value)
                    if stray:
                        offenders.append(
                            "%s sentence %s %s: %r contains %s"
                            % (os.path.basename(path), idx, field, value,
                               ", ".join(unicodedata.name(c, "?") for c in stray)))
                # The same term twice in one chapter means the reader is told the
                # same thing twice and the glossary gets two rows competing under
                # one lemma. The de-duplication below only looks within a single
                # sentence, so repeats across sentences would otherwise pass.
                term = unicodedata.normalize("NFC", entry[0]).strip()
                if term in seen_terms:
                    repeats.append("%r at sentences %s and %s"
                                   % (term, seen_terms[term], idx))
                else:
                    seen_terms[term] = idx
            new.setdefault(idx, []).extend(entries)

    if offenders:
        raise SystemExit(
            "refusing to merge: %d field(s) contain non-Latin letters\n  %s"
            % (len(offenders), "\n  ".join(offenders)))

    if repeats:
        raise SystemExit(
            "refusing to merge: %d term(s) glossed more than once\n  %s"
            % (len(repeats), "\n  ".join(repeats)))

    out_paragraphs = []
    idx = 0
    n_known = n_new = n_sent = 0
    for para in chapter["paragraphs"]:
        built = []
        for sentence in para:
            vocab, seen = [], set()
            for entry in known.get(str(idx), []) + new.get(str(idx), []):
                term = entry[0]
                if term in seen:
                    continue
                seen.add(term)
                vocab.append(entry)
            n_known += len(known.get(str(idx), []))
            n_new += len(new.get(str(idx), []))
            built.append({"sentence": sentence, "vocab": vocab})
            idx += 1
            n_sent += 1
        out_paragraphs.append(built)

    # Land in results/ for a human to copy across. Never write into the live
    # directory the app serves from.
    out_path = args.out
    if not out_path:
        out_dir = os.path.join(paths.RESULTS, book)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "%s_glossed.json" % name)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"title": chapter["title"], "paragraphs": out_paragraphs},
                  fh, ensure_ascii=False, indent=1)

    total = sum(len(s["vocab"]) for p in out_paragraphs for s in p)
    print("%s -> %s" % (name, out_path))
    print("  sentences %d   vocab %d  (%.2f per sentence)"
          % (n_sent, total, total / n_sent if n_sent else 0))
    print("  from DB %d   from model %d   dropped as duplicate %d"
          % (n_known, n_new, n_known + n_new - total))
    print("  batches glossed %d/%d" % (len(produced), n_batches))


if __name__ == "__main__":
    main()
