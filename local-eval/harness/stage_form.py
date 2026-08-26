#!/usr/bin/env python3
"""Stage 1 — the citation form, with no model involved.

Measured failure the monolithic run makes: it writes inflected tokens as terms
('herhüpfte'), which is the same defect already sitting in 5,426 glossary rows.
A lookup cannot make that mistake, so this stage takes the job away entirely.

Plain lexicon lookup covers 82% of nouns and verbs. The misses are almost all
novel compounds — Sockenpaar, Riesenschachtel, Silberglöckchen — which are not
in Wiktionary and never will be. But a German compound takes its gender and its
plural from its LAST element, so 'Sockenpaar' is decidable from 'Paar' even
though the compound itself is unknown. That rule is exact, not a heuristic, and
it is what lifts coverage past the dictionary's own vocabulary.

    python3 stage_form.py            # measures coverage against the gold set
"""

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
from score import head, classify           # noqa: E402
from german import base_forms              # noqa: E402

try:
    from compound_split import char_split as _cs
except ImportError:                       # the membership scan still works
    _cs = None


def _charsplit(word):
    """(score, modifier, head) best-first, or empty if CharSplit isn't installed."""
    if _cs is None:
        return []
    try:
        return _cs.split_compound(word)[:3]
    except (ValueError, IndexError, KeyError):
        return []


LEX = os.path.join(HERE, "..", "models", "lexicon", "lexicon.db")
# Linking elements German glues compounds with: Sockenpaar has none,
# Riesenschachtel none, but Arbeitszimmer has -s- and Sonnenschein -n-.
LINKERS = ("es", "en", "er", "s", "n", "e")
# Separable prefixes, longest first so 'hinab' is tried before 'hin'.
VERB_PREFIXES = (
    "hinunter", "hinüber", "herunter", "herüber", "zusammen", "auseinander",
    "entgegen", "gegenüber", "hinter", "hinauf", "hinaus", "hinein", "hinab",
    "herauf", "heraus", "herein", "herum", "herbei", "vorbei", "vorüber", "drauf", "drein",
    "zurück", "zurecht", "davon", "dazu", "empor", "fest", "fort", "statt",
    "voran", "voraus", "weiter", "wieder", "durch", "gegen", "unter", "über",
    "wider", "groß", "klein", "frei", "hoch", "nieder", "heim", "teil",
    "auf", "aus", "ein", "mit", "nach", "vor", "zu", "ab", "an", "bei",
    "los", "weg", "her", "hin", "da", "um",
)
MIN_HEAD = 4
# Wiktionary's inflected-form pages describe a grammatical slot instead of a
# meaning. Matching the opening of that description is enough to spot them.
FORM_PAGE = re.compile(
    r"^\s*(Nominativ|Genitiv|Dativ|Akkusativ|Positiv|Komparativ|Superlativ|"
    r"Singular|Plural|Flexion|Partizip|Grundform|starke|schwache|gemischte|"
    r"[123]\.\s*Person|Imperativ|Konjunktiv|Präteritum|Präsens|Perfekt)\b",
    re.I)


class Former:
    def __init__(self, path=LEX):
        self.db = sqlite3.connect(os.path.abspath(path))
        self.cache = {}

    def _rows(self, word):
        rows = self.db.execute(
            "SELECT pos, term, de, hu FROM lemma WHERE word = ?",
            (word.lower(),)).fetchall()
        # German Wiktionary gives every inflected form its own page — 297k of
        # the 'adj' entries are things like 'berichtigte', glossed "Nominativ
        # Singular Femininum der starken Flexion". Returning one of those makes
        # the inflected token the citation form, which is precisely the defect
        # this stage exists to prevent. They are only ever a route to the lemma.
        real = [r for r in rows if not FORM_PAGE.match(r[2] or "")]
        return real or []

    def _direct(self, word, want):
        rows = self._rows(word)
        if not rows:
            hit = self.db.execute(
                "SELECT word FROM forms WHERE surface = ? LIMIT 1",
                (word.lower(),)).fetchone()
            if hit:
                rows = self._rows(hit[0])
        if not rows:
            for f in base_forms(word)[:6]:
                rows = self._rows(f)
                if rows:
                    break
        if not rows:
            return None
        for r in rows:
            if r[0] == want:
                return r
        return rows[0]

    def compound(self, word):
        """Gender and plural of the last element, re-attached to the whole word.

        Only for nouns, and only when the tail is long enough to be a real
        head — splitting 'Vater' into 'Va'+'ter' would invent a word.
        """
        w = word[0].upper() + word[1:]

        # CharSplit scores every split point from ngram statistics and is right
        # about the head roughly 95% of the time, which beats anything derived
        # from dictionary membership alone: any long German word ends in
        # *something* the dictionary knows, which is how a hand-rolled version
        # turned a rare compound into 'Äcker' (fields) and Sockenpaar into 'Aar'.
        for score, mod, tail in _charsplit(w):
            if score < 0 or len(tail) < MIN_HEAD or len(mod) < MIN_HEAD - 1:
                continue
            row = self._direct(tail, "noun")
            if not row or not row[1].lower().startswith(("der ", "die ", "das ")):
                continue
            art, _, rest = row[1].partition(" ")
            plural = rest.split(",", 1)[1].strip() if "," in rest else None
            term = "%s %s%s" % (art, w, ", " + plural if plural else "")
            return term, "charsplit:" + tail

        # Fall back to the membership scan when CharSplit has no confident split.
        for cut in range(MIN_HEAD, len(w) - MIN_HEAD + 1):
            tail = w[cut:]
            modifier = w[:cut]
            row = self._direct(tail[0].upper() + tail[1:], "noun")
            if not row or not row[1].lower().startswith(("der ", "die ", "das ")):
                continue
            # The modifier must itself be a word, allowing one linking element.
            # Without this check any suffix that happens to be a noun wins.
            stem = None
            for link in ("",) + LINKERS:
                cand = modifier[:-len(link)] if link else modifier
                if link and not modifier.lower().endswith(link):
                    continue
                if len(cand) < MIN_HEAD - 1:
                    continue
                if self._direct(cand[0].upper() + cand[1:], None) or \
                        self._direct(cand.lower(), None):
                    stem = cand
                    break
            if stem is None:
                continue
            art, _, rest = row[1].partition(" ")
            plural = rest.split(",", 1)[1].strip() if "," in rest else None
            term = "%s %s%s" % (art, w, ", " + plural if plural else "")
            return term, "compound:" + tail
        return None

    def prefixed_verb(self, word):
        """Build a separable verb's parts from the base verb's.

        Every remaining miss is one of these — hinabspähen, hineinkritzeln,
        ankämpfen — because Wiktionary has 'spähen' but not every prefixed
        form. The transform is mechanical: the prefix detaches in the
        preterite ('spähte hinab') and takes 'ge' inside in the participle
        ('hinabgespäht'), so the base verb's entry determines all three parts.
        """
        for p in VERB_PREFIXES:
            if not word.lower().startswith(p) or len(word) - len(p) < 4:
                continue
            base = word[len(p):]
            row = self._direct(base, "verb")
            if not row or row[0] != "verb":
                continue
            parts = [x.strip() for x in row[1].split(",")]
            if len(parts) < 3:
                continue
            prat = "%s %s" % (parts[1], p)               # spähte hinab
            aux, _, part2 = parts[2].partition(" ")
            if part2.startswith("ge"):
                part2 = p + part2                        # hinab + gespäht
            else:
                part2 = p + part2
            return "%s, %s, %s %s" % (word, prat, aux, part2), "prefix:" + p
        return None

    def form(self, word, kind=None):
        """(term, how) or None. `how` records which route produced it."""
        key = (word.lower(), kind)
        if key in self.cache:
            return self.cache[key]
        want = kind if kind in ("noun", "verb") else None
        row = self._direct(word, want)
        out = None
        if row and (not want or row[0] == want):
            out = (row[1], "lexicon")
        elif word[:1].isupper() or (want == "noun"):
            out = self.compound(word)
        if out is None and want == "verb":
            out = self.prefixed_verb(word)
        if out is None and row:
            out = (row[1], "lexicon:pos-mismatch")
        self.cache[key] = out
        return out


def main():
    f = Former()
    tasks = [json.loads(l) for l in
             open(os.path.join(HERE, "..", "goldset", "tasks.jsonl"), encoding="utf-8")]
    stat = {"n": 0, "hit": 0, "exact": 0, "same": 0, "wrong": 0, "miss": 0}
    how = {}
    misses, diffs = [], []
    for t in tasks:
        for entries in t["gold"].values():
            for e in entries:
                gold = unicodedata.normalize("NFC", e[0])
                kind = classify(gold)
                if kind not in ("noun", "verb"):
                    continue
                stat["n"] += 1
                got = f.form(head(gold), kind)
                if not got:
                    stat["miss"] += 1
                    misses.append(gold)
                    continue
                stat["hit"] += 1
                how[got[1].split(":")[0]] = how.get(got[1].split(":")[0], 0) + 1
                term = unicodedata.normalize("NFC", got[0])
                if term == gold:
                    stat["exact"] += 1
                elif head(term) == head(gold):
                    stat["same"] += 1
                    diffs.append((gold, term))
                else:
                    stat["wrong"] += 1
                    diffs.append((gold, term))

    n = stat["n"] or 1
    print("nouns + verbs in the gold set: %d" % n)
    for k in ("hit", "exact", "same", "wrong", "miss"):
        print("  %-6s %4d  %5.1f%%" % (k, stat[k], 100.0 * stat[k] / n))
    print("  routes: %s" % how)
    print("\n  still missing: %s" % ", ".join(misses[:10]))
    print("\n  differs from gold:")
    for a, b in diffs[:10]:
        print("    %-46s -> %s" % (a, b))


if __name__ == "__main__":
    main()
