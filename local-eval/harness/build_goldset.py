#!/usr/bin/env python3
"""Turn the finished work/ directory into a gold set for testing local models.

Every batch_NNN.json in work/ has a glossed_NNN.json beside it — the input a
frontier model saw and the output it produced, 210 pairs across the corpus.
That is the target any local candidate has to reproduce, so the bench is built
straight from it rather than from anything hand-written.

A *task* is a small slice of one batch: N sentences that actually carry
candidates, plus the gold entries for them. Sentences with `new: []` are
dropped — they are 62% of the prompt and carry only 0.9% of the gold entries,
so keeping them would spend the whole local prefill budget on nothing.

    python3 build_goldset.py --tasks 60 --sentences 10
"""

import argparse
import collections
import glob
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# Two conventions live in work/. The README one puts the citation form in the
# term ("die Front, -en"); two books instead put the *inflected token* there
# ("Ladenfronten") and fold the citation form into the definition
# ("die Front, -en: die Vorderseite der Läden"). 5,426 glossary entries are
# keyed that second way. Benchmarking a local model against those would score
# it on reproducing a defect, so they are dropped from the gold set.
FOLDED = re.compile(r"^(der|die|das)\s.*:\s|^[a-zäöüß]+(,\s*\S+){1,2}:\s|^\S+:\s")
CITED = re.compile(r"^(der|die|das)\s+\S|,\s*(hat|ist)\s|^sich\s")

def load_pairs(work):
    """Yield (book, chapter, batch_no, sentences, gold) for every paired file."""
    for bp in sorted(glob.glob(os.path.join(work, "*", "ch*", "batch_*.json"))):
        gp = bp.replace("batch_", "glossed_")
        if not os.path.exists(gp):
            continue
        try:
            batch = json.load(open(bp, encoding="utf-8"))
            gold = json.load(open(gp, encoding="utf-8"))["entries"]
        except (ValueError, KeyError):
            continue
        parts = bp.split(os.sep)
        yield parts[-3], parts[-2], batch["batch"], batch["sentences"], gold


def contract_clean(entry):
    """True if the entry follows the README's term/de/hu split."""
    if not (isinstance(entry, (list, tuple)) and len(entry) >= 3):
        return False
    term, de = entry[0], entry[1]
    if not (isinstance(term, str) and isinstance(de, str)) or not term or not de:
        return False
    if FOLDED.match(de.strip()):
        return False                      # citation form hiding in the definition
    # A capitalised single word with no article is a noun that lost its gender.
    if term[:1].isupper() and " " not in term and "," not in term:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(ROOT, "work"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "goldset", "tasks.jsonl"))
    ap.add_argument("--tasks", type=int, default=60, help="total tasks to emit")
    ap.add_argument("--sentences", type=int, default=10,
                    help="sentences-with-candidates per task")
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--keep-folded", action="store_true",
                    help="keep the 'inflected term + folded definition' "
                         "entries instead of dropping them")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Group candidate slices by book so the sample is stratified: Hoffmann is
    # 12 batches against a long book's 68, and it is the hardest text in the corpus, so
    # uniform sampling would bury the one register that actually discriminates.
    by_book = collections.defaultdict(list)
    dropped = kept = 0
    for book, chap, bno, sentences, gold in load_pairs(args.work):
        live = [s for s in sentences if s["new"]]
        for start in range(0, len(live) - args.sentences + 1, args.sentences):
            window = live[start:start + args.sentences]
            g = {}
            for s in window:
                raw = gold.get(str(s["i"]))
                if not raw:
                    continue
                good = raw if args.keep_folded else [
                    e for e in raw if contract_clean(e)]
                dropped += len(raw) - len(good)
                kept += len(good)
                if good:
                    g[str(s["i"])] = good
            if not g:
                continue                      # nothing to score against
            by_book[book].append({
                "book": book, "chapter": chap, "batch": bno,
                "sentences": window, "gold": g,
            })

    books = sorted(by_book)
    if not books:
        sys.exit("no paired batch/glossed files under %s" % args.work)

    per_book = max(1, args.tasks // len(books))
    tasks = []
    for b in books:
        pool = by_book[b]
        rng.shuffle(pool)
        tasks.extend(pool[:per_book])
    rng.shuffle(tasks)
    tasks = tasks[:args.tasks]
    for n, t in enumerate(tasks, 1):
        t["id"] = "t%03d" % n

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    ents = sum(sum(len(v) for v in t["gold"].values()) for t in tasks)
    sents = sum(len(t["sentences"]) for t in tasks)
    chars = sum(len(s["text"]) for t in tasks for s in t["sentences"])
    print("wrote %s" % out)
    print("  tasks %d   sentences %d   gold entries %d" % (len(tasks), sents, ents))
    if not args.keep_folded:
        print("  corpus-wide: kept %d contract-clean entries, dropped %d folded "
              "(%.1f%%)" % (kept, dropped, 100.0 * dropped / (kept + dropped or 1)))
    print("  prompt text %d chars  (~%d tokens/task)" % (chars, chars / 3.5 / len(tasks)))
    for b in books:
        n = sum(1 for t in tasks if t["book"] == b)
        print("  %-45s %d tasks (pool %d)" % (b[:45], n, len(by_book[b])))


if __name__ == "__main__":
    main()
