#!/usr/bin/env python3
"""Surface form -> lemma, from Wiktionary's inflection tables.

The pipeline was looking words up exactly as they appear in the text, so
'angenommen', 'abzugeben' and 'aufgebrochen' missed every dictionary that keys
on 'annehmen', 'abgeben' and 'aufbrechen'. That single omission accounted for
118 of the 373 words with no Hungarian.

Worth its own module because every stage needs the same answer and the naive
alternatives are both wrong in the same direction: `german.py` strips suffixes
and cannot reach a strong verb's stem, while splitting the word and translating
its tail invents meanings ('Artischocke' -> 'Schocke' -> five dozen).

    from lemma import Lemmatizer
    Lemmatizer().of("angenommen")   -> "annehmen"
"""

import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from german import base_forms  # noqa: E402

LEX = os.path.join(HERE, "..", "models", "lexicon", "lexicon.db")


class Lemmatizer:
    def __init__(self, path=LEX):
        self.db = sqlite3.connect(os.path.abspath(path))
        self.cache = {}

    def of(self, word):
        """The lemma, or the word itself when nothing better is known."""
        w = (word or "").lower().strip()
        if not w:
            return w
        if w in self.cache:
            return self.cache[w]
        out = w
        # A word that is itself a headword is already a lemma — check that
        # first, or 'laden' (to load) gets rewritten to some other entry's base.
        row = self.db.execute(
            "SELECT 1 FROM lemma WHERE word = ? LIMIT 1", (w,)).fetchone()
        if not row:
            hit = self.db.execute(
                "SELECT word FROM forms WHERE surface = ? LIMIT 1", (w,)).fetchone()
            if hit:
                out = hit[0]
            else:
                for f in base_forms(w)[:6]:
                    if self.db.execute("SELECT 1 FROM lemma WHERE word = ? LIMIT 1",
                                       (f,)).fetchone():
                        out = f
                        break
        self.cache[w] = out
        return out

    def candidates(self, word):
        """Surface first, then lemma — for callers that want to try both."""
        w = (word or "").lower().strip()
        lem = self.of(w)
        return [w] if lem == w else [w, lem]
