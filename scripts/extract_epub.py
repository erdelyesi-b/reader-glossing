#!/usr/bin/env python3
"""Extract an epub into the Reader's per-chapter JSON.

Writes chNN.json files shaped exactly like the existing corpus:

    {"title": "...", "paragraphs": [["sentence", ...], ...]}

Chapters are taken in spine order (the reading order), not zip order. Text is
normalised conservatively — entities decoded, Unicode NFC, whitespace collapsed —
and nothing else. Deliberately no OCR "correction": on this corpus the classic
fixes do more harm than good (a dehyphenator would wreck legitimate German
suspended compounds like 'hin- und her'). Run audit_text.py to check a new book
before assuming it needs more.

    python3 extract_epub.py --epub "source/BOOK.epub" --book "Title"

Output lands under chapters/<Book>/; never write into the live app directory.
"""

import argparse
import html
import json
import os
import re
import sys
import unicodedata
import zipfile
from xml.etree import ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
from sentences import split_sentences  # noqa: E402

OPF_NS = {"o": "http://www.idpf.org/2007/opf"}
CONTAINER = "META-INF/container.xml"

# <p> classes that carry layout, not prose.
SKIP_CLASSES = {"pagebreak", "break"}


def spine_hrefs(z):
    """Documents in reading order, resolved relative to the OPF."""
    container = ET.fromstring(z.read(CONTAINER))
    opf_path = container[0][0].get("full-path")
    base = os.path.dirname(opf_path)
    opf = ET.fromstring(z.read(opf_path))
    manifest = {i.get("id"): i.get("href") for i in opf.find("o:manifest", OPF_NS)}
    out = []
    for ref in opf.find("o:spine", OPF_NS):
        href = manifest.get(ref.get("idref"))
        if href:
            out.append(os.path.normpath(os.path.join(base, href)) if base else href)
    return out


def clean(text):
    text = re.sub(r"(?s)<[^>]+>", "", text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("­", "")           # soft hyphen
    text = text.replace(" ", " ")          # nbsp
    return re.sub(r"\s+", " ", text).strip()


def split_sections(source, pat):
    """Cut one spine document into sections at headings matching pat.

    Some epubs — Project Gutenberg's especially — put many chapters in one
    document and mark each with a heading rather than giving it its own spine
    entry. A section runs from its heading to the next one; anything before the
    first heading is front matter (PG's boilerplate header) and is dropped.
    """
    starts = [
        m.start()
        for m in re.finditer(r"(?s)<h[1-6][^>]*>.*?</h[1-6]>", source)
        if pat.search(m.group(0))
    ]
    return [
        source[s:e]
        for s, e in zip(starts, starts[1:] + [len(source)])
    ]


def parse_chapter(source):
    source = re.sub(r"(?s)<(script|style).*?</\1>", " ", source)

    title = ""
    m = re.search(r"(?s)<h[1-6][^>]*>(.*?)</h[1-6]>", source)
    if m:
        title = clean(m.group(1))

    paragraphs = []
    for pm in re.finditer(r"(?s)<p([^>]*)>(.*?)</p>", source):
        attrs, body = pm.group(1), pm.group(2)
        cls = re.search(r'class="([^"]*)"', attrs)
        if cls and set(cls.group(1).split()) & SKIP_CLASSES:
            continue
        text = clean(body)
        if not text:
            continue
        sents = split_sentences(text)
        if sents:
            paragraphs.append(sents)
    return title, paragraphs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epub", required=True)
    ap.add_argument("--book", default=None,
                    help="book title; chapters land in <root>/chapters/<book>")
    ap.add_argument("--out", default=None,
                    help="explicit output dir (overrides --book)")
    ap.add_argument(
        "--chapter-pattern",
        default=r"_ch\d+_",
        help="regex selecting chapter documents within the spine",
    )
    ap.add_argument(
        "--split-heading",
        default=None,
        help="regex on a heading tag; splits each matched spine document into "
             "one chapter per heading, dropping whatever precedes the first",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    z = zipfile.ZipFile(args.epub)
    pat = re.compile(args.chapter_pattern)
    hrefs = [h for h in spine_hrefs(z) if pat.search(h)]
    if not hrefs:
        raise SystemExit("no spine documents match %r" % args.chapter_pattern)

    docs = [z.read(h).decode("utf-8", "replace") for h in hrefs]
    if args.split_heading:
        spat = re.compile(args.split_heading)
        docs = [s for d in docs for s in split_sections(d, spat)]
        if not docs:
            raise SystemExit("no headings match %r" % args.split_heading)

    out = args.out or os.path.join(paths.CHAPTERS, args.book or "Untitled")
    if not args.dry_run:
        os.makedirs(out, exist_ok=True)

    total_s = total_c = 0
    for i, source in enumerate(docs, 1):
        title, paragraphs = parse_chapter(source)
        n_s = sum(len(p) for p in paragraphs)
        n_c = sum(len(s) for p in paragraphs for s in p)
        total_s += n_s
        total_c += n_c
        name = "ch%02d.json" % i
        print(
            "%-11s %-42s %4d para %5d sent %7d chars"
            % (name, title[:42], len(paragraphs), n_s, n_c)
        )
        if not args.dry_run:
            with open(os.path.join(out, name), "w", encoding="utf-8") as fh:
                json.dump(
                    {"title": title, "paragraphs": paragraphs},
                    fh,
                    ensure_ascii=False,
                    indent=1,
                )

    print("\n%d chapters, %d sentences, %d chars" % (len(docs), total_s, total_c))
    if args.dry_run:
        print("(dry run — nothing written)")


if __name__ == "__main__":
    main()
