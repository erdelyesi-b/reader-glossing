#!/usr/bin/env python3
"""Split a chapter into glossing batches, with everything the DB already knows
resolved locally so it never reaches the model.

Writes into <work>/<chNN>/:
    batch_NNN.json  what the model reads: sentences + the words still unresolved
    known.json      DB hits, keyed by sentence index — merged back for free

    python3 make_batches.py --chapter ".../ch01.json" --work ./work/book
"""

import argparse
import collections
import json
import os
import re
import sqlite3
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
from german import lemma_keys  # noqa: E402

WORD = re.compile(r"[A-Za-zÄÖÜäöüß]+")


def load_index(db_path, stop_rank):
    db = sqlite3.connect(db_path)
    stop = {r[0] for r in db.execute(
        "SELECT word FROM stoplist WHERE rank <= ?", (stop_rank,))}
    best = {}
    for lemma, term, de, hu, freq in db.execute(
        "SELECT lemma, term, de, hu, freq FROM entries ORDER BY freq ASC"
    ):
        best[lemma] = [term, de, hu]  # ascending freq: last write wins = highest
    db.close()
    return stop, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chapter", required=True)
    ap.add_argument("--work", default=paths.WORK)
    ap.add_argument("--db", default=paths.DB)
    # 1200 keeps us consistent with what the first three books chose to gloss:
    # everything in the stoplist was declined there at least once. Lower it if a
    # book starts losing glosses you want; --exact-only turns off fuzzy matching.
    ap.add_argument("--stop-rank", type=int, default=1200)
    ap.add_argument("--target", type=int, default=250,
                    help="new distinct candidates per batch")
    ap.add_argument("--exact-only", action="store_true",
                    help="skip morphological matching; only exact lemma hits count as known")
    ap.add_argument("--morph-display", action="store_true",
                    help="also show morphologically matched entries to the reader; "
                         "off because ~7%% of those matches are wrong (see below)")
    args = ap.parse_args()

    stop, best = load_index(args.db, args.stop_rank)
    exact = {l.split(":", 1)[1]: l for l in best}

    chapter = json.load(open(args.chapter, encoding="utf-8"))
    name = os.path.basename(args.chapter).split(".")[0]
    book = os.path.basename(os.path.dirname(os.path.abspath(args.chapter)))
    outdir = os.path.join(args.work, book, name)
    os.makedirs(outdir, exist_ok=True)

    flat = [(pi, si, s)
            for pi, para in enumerate(chapter["paragraphs"])
            for si, s in enumerate(para)]

    known = {}
    stats = collections.Counter()
    batches, cur, seen = [], [], set()

    for idx, (pi, si, sentence) in enumerate(flat):
        unresolved = []
        hits = []
        for raw in WORD.findall(unicodedata.normalize("NFC", sentence)):
            w = raw.lower()
            if w in stop:
                stats["stop"] += 1
                continue
            lemma = exact.get(w)
            if lemma:
                stats["exact"] += 1
                hits.append(best[lemma])
                continue
            if not args.exact_only:
                lemma = next((k for k in lemma_keys(w) if k in best), None)
                if lemma:
                    stats["morph"] += 1
                    # Suppress the word, but do NOT put the entry on the page.
                    # german.py is tuned for the skip decision, where a false hit
                    # only costs a gloss; re-using the hit as a *displayed* entry
                    # turns that same 7% into wrong glosses the reader sees.
                    # 'Schlangen' stems to 'schlang' and picked up a legacy entry
                    # for the preterite of 'schlingen' — the noun and the verb
                    # form are indistinguishable to a stemmer this cheap. Costs
                    # ~5% of re-used entries: the rarest-first cap drops most
                    # fuzzy hits anyway. --morph-display restores the old way.
                    if args.morph_display:
                        hits.append(best[lemma])
                    continue
            stats["candidate"] += 1
            if w not in seen:
                unresolved.append(raw)

        if hits:
            # de-duplicate within the sentence, keep order
            uniq, seen_t = [], set()
            for h in hits:
                if h[0] not in seen_t:
                    seen_t.add(h[0])
                    uniq.append(h)
            known[str(idx)] = uniq

        cur.append({"i": idx, "text": sentence, "new": unresolved})
        seen.update(w.lower() for w in unresolved)
        if len(seen) >= args.target * (len(batches) + 1):
            batches.append(cur)
            cur = []
    if cur:
        batches.append(cur)

    # One sentence per line, each compact. Indenting this file put every candidate
    # word on its own padded line and spent ~36% of the batch's input tokens on
    # whitespace — more than the candidate lists themselves cost. Staying
    # line-per-sentence keeps it greppable for a fraction of that.
    for n, batch in enumerate(batches, 1):
        path = os.path.join(outdir, "batch_%03d.json" % n)
        head = {"chapter": name, "batch": n}
        with open(path, "w", encoding="utf-8") as fh:
            fh.write('{"chapter": %s, "batch": %d, "sentences": [\n'
                     % (json.dumps(head["chapter"]), n))
            fh.write(",\n".join(
                json.dumps(s, ensure_ascii=False, separators=(",", ":"))
                for s in batch))
            fh.write("\n]}\n")

    with open(os.path.join(outdir, "known.json"), "w", encoding="utf-8") as fh:
        json.dump(known, fh, ensure_ascii=False, indent=1)

    with open(os.path.join(outdir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump({"chapter": name, "title": chapter["title"],
                   "shape": [len(p) for p in chapter["paragraphs"]],
                   "batches": len(batches)}, fh, ensure_ascii=False)

    total = sum(stats.values())
    print("%s  %s" % (name, chapter["title"]))
    print("  sentences %d   paragraphs %d" % (len(flat), len(chapter["paragraphs"])))
    for k in ("stop", "exact", "morph", "candidate"):
        print("  %-10s %6d  %4.1f%%" % (k, stats[k], 100.0 * stats[k] / total))
    print("  distinct candidates %d  ->  %d batch(es)" % (len(seen), len(batches)))
    print("  wrote %s" % outdir)


if __name__ == "__main__":
    main()
