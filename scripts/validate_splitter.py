#!/usr/bin/env python3
"""Round-trip the splitter against the already-split chapters.

Rejoins each known-good paragraph's sentences, re-splits it, and reports how often
we reproduce the original boundaries. Prints only counts and short diff fragments,
never whole sentences.

    python3 validate_splitter.py [--root <dir of finished books>]
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
from sentences import split_sentences  # noqa: E402


def rejoin(sentences):
    """Rebuild the source paragraph from its sentences.

    A plain ' '.join() would invent a space before an orphaned closing guillemet
    (the corpus contains none), so the splitter would be judged against text that
    never existed. Join tight when the next piece opens with a closer.
    """
    out = sentences[0]
    for piece in sentences[1:]:
        out += ("" if piece[:1] in "«)]" else " ") + piece
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=paths.LIVE)
    ap.add_argument("--show", type=int, default=8)
    args = ap.parse_args()

    paras = exact = 0
    got = want = 0
    over = under = 0
    samples = []

    # A chapter may be present as extracted text or as a finished gloss; the two
    # hold the same sentences, so take one per chapter and prefer the glossed copy,
    # which is the known-good corpus the splitter is actually judged against.
    by_chapter = {}
    for path in sorted(glob.glob(os.path.join(args.root, "*", "ch*.json"))):
        if "_part" in path:
            continue
        book = os.path.basename(os.path.dirname(path))
        chapter = os.path.basename(path).split("_")[0]
        if (book, chapter) not in by_chapter or "_glossed" in path:
            by_chapter[(book, chapter)] = path

    for path in [by_chapter[k] for k in sorted(by_chapter)]:
        for para in json.load(open(path, encoding="utf-8"))["paragraphs"]:
            if not para:
                continue
            # extracted chapters hold bare strings, glossed ones {"sentence", "vocab"}
            original = [
                re.sub(r"\s+", " ", s if isinstance(s, str) else s["sentence"]).strip()
                for s in para
            ]
            rebuilt = split_sentences(rejoin(original))
            paras += 1
            want += len(original)
            got += len(rebuilt)
            if rebuilt == original:
                exact += 1
            else:
                if len(rebuilt) > len(original):
                    over += 1
                elif len(rebuilt) < len(original):
                    under += 1
                if len(samples) < args.show:
                    # show only the boundary neighbourhood, never a full sentence
                    for a, b in zip(original, rebuilt):
                        if a != b:
                            samples.append(
                                "want …%-28s | got …%-28s"
                                % (repr(a[-28:]), repr(b[-28:]))
                            )
                            break

    print("paragraphs         %d" % paras)
    print("exact match        %d  (%.2f%%)" % (exact, 100.0 * exact / paras))
    print("sentences want/got %d / %d" % (want, got))
    print("paras over-split   %d" % over)
    print("paras under-split  %d" % under)
    if samples:
        print("\nfirst disagreements (boundary tails only):")
        for s in samples:
            print("  " + s)


if __name__ == "__main__":
    main()
