#!/usr/bin/env python3
"""Stage 4 — the Hungarian, from Balázs's own de-hu dictionary rather than a model.

This is the stage the monolithic 26B failed hardest: 31.6% of its entries
contained a word that is not Hungarian ('megjámul', 'házasmertek'). A
dictionary cannot invent a word, so if coverage is good enough the whole
failure mode disappears rather than being mitigated.

dictionary.sqlite holds 22,113 distinct German terms with Hungarian
translations, and encodes gender in the headword as a suffix: '(r)' der,
'(e)' die, '(s)' das. So it supplies the article too, and Wiktionary is left
supplying only the plural and the verb's principal parts.

Multiple senses per word are the norm ('schleichen' has five). Picking the one
that fits the sentence is a ranking problem over a fixed list — a model can do
that badly and still not produce a non-word, which is the point.

    python3 stage_hu.py
"""

import collections
import json
import os
import re
import sqlite3
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from score import head, hu_tokens          # noqa: E402
from german import base_forms              # noqa: E402
from stage_form import _charsplit          # noqa: E402
from lemma import Lemmatizer               # noqa: E402

DEFAULT_DICT = os.path.expanduser(
    "~/Documents/Codebase/dietpi_home_page/data/dictionary.sqlite")
GENDER = {"r": "der", "e": "die", "s": "das"}
MARKER = re.compile(r"\s*\(([res])\)\s*$")
# 'ld. beißen' is Hungarian for "see: beißen" — a redirect to another headword,
# not a translation. Emitting it put the German words 'treiben', 'waschen' and
# 'wissen' into the Hungarian field of finished entries.
REDIRECT = re.compile(r"^\s*ld\.\s*(.+)$", re.I)
# Domain labels: '(átv.)' figurative, '(kat.)' military. The trailing dot is
# what separates them from government markers like '(vmire)', which the house
# style keeps on purpose.
# Domain labels appear as '(átv.)', '(átv. is)' and bare '(kat)', so keying on
# the trailing dot alone missed two of the three. Match the abbreviation itself
# instead, and leave government markers like '(vmire)' untouched — the house
# style keeps those on purpose.
LABEL = re.compile(
    r"\s*\((?:[^()]*\b(?:átv|kat|orv|jog|műsz|növ|áll|zene|sp|vall|nyelvt|"
    r"pej|biz|rég|tréf|ép|vegy|csill|mat|fiz|isk|hajó|rep|vasút|ker|"
    r"nyomd|film|szính|zool|bot)\b[^()]*)\)", re.I)


class Dictionary:
    """German headword -> (gender, [hungarian senses])."""

    def __init__(self, path=DEFAULT_DICT, lemmatizer=None):
        db = sqlite3.connect(path)
        self.by_word = collections.defaultdict(list)
        # German distinguishes 'Laden' (shop) from 'laden' (to load) by case
        # alone, and lowercasing the key merges them — which is how
        # 'Antiquitätenladen' ended up meaning 'hív' (to call). Compound heads
        # are always nouns, so they are resolved against this cased index.
        self.by_cased = collections.defaultdict(list)
        self.gender = {}
        self.redirect = {}
        self.lem = lemmatizer or Lemmatizer()
        self._lex = sqlite3.connect(os.path.abspath(os.path.join(
            HERE, "..", "models", "lexicon", "lexicon.db")))
        for term, tr in db.execute(
                "SELECT term, translation FROM entries WHERE lang = 'de'"):
            if not term or not tr:
                continue
            m = MARKER.search(term)
            key = MARKER.sub("", term).strip().lower()
            if m:
                self.gender.setdefault(key, GENDER[m.group(1)])
            red = REDIRECT.match(tr)
            if red:
                self.redirect.setdefault(key, red.group(1).strip().lower())
                continue
            clean = LABEL.sub("", tr).strip()
            if not clean:
                continue
            if clean not in self.by_word[key]:
                self.by_word[key].append(clean)
            cased = MARKER.sub("", term).strip()
            if clean not in self.by_cased[cased]:
                self.by_cased[cased].append(clean)
        db.close()

    def _is_noun(self, word):
        row = self._lex.execute(
            "SELECT pos FROM lemma WHERE word = ? AND pos IN ('noun','verb','adj','adv')"
            " ORDER BY CASE pos WHEN 'noun' THEN 0 ELSE 1 END LIMIT 1",
            (word.lower(),)).fetchone()
        if row:
            return row[0] == "noun"
        # Unknown to Wiktionary: almost always a novel compound noun, which is
        # exactly the case this fallback exists for.
        return True

    def look(self, word):
        """(gender, senses, how) or None. Surface, then lemma, then stems."""
        w = word.lower().strip()
        if w in self.by_word:
            return self.gender.get(w), self.by_word[w], "exact"
        if w in self.redirect:
            r = self.redirect[w]
            if r in self.by_word:
                return self.gender.get(r), self.by_word[r], "redirect"
        # Lemmatise before giving up: the text says 'angenommen', the
        # dictionary knows 'annehmen'. This alone recovered 118 of 373 misses.
        lem = self.lem.of(w)
        if lem != w and lem in self.by_word:
            return self.gender.get(lem), self.by_word[lem], "lemma"
        for f in base_forms(w)[:8]:
            if f in self.by_word:
                return self.gender.get(f), self.by_word[f], "stem"
        # Compounds: the dictionary rarely has a novel compound, but the head
        # noun carries the gender and most of the meaning. CharSplit picks the
        # head properly; scanning every suffix for dictionary membership finds
        # spurious ones, because a long German word ends in *some* known word.
        # Nouns only. 'bergeweise' is an adverb, and splitting it yields
        # 'weise' -> bölcs (wise); 'Artischocke' yields 'Schocke' -> five dozen.
        # Both are confident, wrong, real words, which no checker can catch.
        # Capitalisation cannot decide this because callers pass lowercased
        # headwords, so ask the lexicon what part of speech the word is.
        if len(w) > 7 and self._is_noun(w):
            for score, mod, tail in _charsplit(w[0].upper() + w[1:]):
                t = tail.lower()
                # Require a gender on the head: this dictionary marks nouns
                # with (r)/(e)/(s), so its absence means the entry is the verb.
                # 'Antiquitätenladen' -> 'Laden' otherwise picks up *laden*
                # (to load) and yields 'hív'.
                noun = tail[0].upper() + tail[1:]
                if score >= 0 and self.by_cased.get(noun) and self.gender.get(t):
                    return self.gender[t], self.by_cased[noun], "compound"
        return None


def main():
    d = Dictionary()
    print("dictionary: %d headwords, %d with gender\n" % (
        len(d.by_word), len(d.gender)))

    tasks = [json.loads(l) for l in
             open(os.path.join(HERE, "..", "goldset", "tasks.jsonl"), encoding="utf-8")]
    stat = collections.Counter()
    how = collections.Counter()
    misses, hits = [], []

    for t in tasks:
        for entries in t["gold"].values():
            for e in entries:
                term, gold_hu = unicodedata.normalize("NFC", e[0]), e[2]
                stat["n"] += 1
                got = d.look(head(term))
                if not got:
                    stat["miss"] += 1
                    misses.append(term)
                    continue
                stat["found"] += 1
                how[got[2]] += 1
                senses = got[1]
                gtok = hu_tokens(gold_hu)
                if any(hu_tokens(s) & gtok for s in senses):
                    stat["sense_present"] += 1
                    hits.append((term, gold_hu, senses))
                if senses and hu_tokens(senses[0]) & gtok:
                    stat["first_sense_right"] += 1
                stat["senses"] += len(senses)

    n = stat["n"] or 1
    print("gold entries: %d" % n)
    print("  found in dictionary        %4d  %5.1f%%   (%s)" % (
        stat["found"], 100.0 * stat["found"] / n, dict(how)))
    print("  gold's Hungarian is among   %4d  %5.1f%%  <- ceiling if a model ranks" % (
        stat["sense_present"], 100.0 * stat["sense_present"] / n))
    print("  ...and is the FIRST sense   %4d  %5.1f%%  <- score with no model at all" % (
        stat["first_sense_right"], 100.0 * stat["first_sense_right"] / n))
    print("  average senses offered      %5.1f" % (
        stat["senses"] / (stat["found"] or 1)))
    print("\n  not in dictionary: %s" % ", ".join(m[:34] for m in misses[:10]))
    print("\n  what ranking would choose from:")
    for term, gold, senses in hits[:8]:
        print("    %-34s gold %-22s from %s" % (
            term[:34], gold[:22], senses[:6]))


if __name__ == "__main__":
    main()
