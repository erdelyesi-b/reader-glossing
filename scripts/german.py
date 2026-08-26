"""Cheap German morphology, just enough to match an inflected surface form to a
glossary lemma.

This is not a lemmatiser. It generates a handful of plausible base forms for a
surface word so we can ask "have we already glossed this?" without sending the
word to the model. Matching is used only to SKIP words, so the failure modes are
asymmetric:

  - a miss costs tokens (the word gets glossed again)
  - a false hit costs a gloss the reader wanted

so the rules stay conservative and every fuzzy hit is reported separately from
the exact ones by make_batches.py, and can be turned off with --exact-only.
"""

import re

UMLAUT = str.maketrans({"ä": "a", "ö": "o", "ü": "u", "Ä": "A", "Ö": "O", "Ü": "U"})

# Ordered longest-first so 'ungen' is tried before 'en'.
NOUN_SUFFIXES = ("ern", "er", "en", "es", "em", "e", "n", "s")
VERB_SUFFIXES = (
    "etest", "etet", "eten", "etest", "test", "tet", "ten", "te",
    "endes", "enden", "endem", "ender", "ende", "end",
    "est", "et", "st", "t", "en", "e",
)
ADJ_SUFFIXES = ("sten", "ste", "eren", "ere", "er", "es", "em", "en", "e")


def _dedupe(seq):
    seen = set()
    out = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def base_forms(surface):
    """Plausible base forms for a surface word, most likely first."""
    w = surface.lower().strip()
    if len(w) < 3:
        return [w]

    out = [w]

    # Strip one inflectional suffix, optionally undoing plural umlaut.
    for suffixes in (NOUN_SUFFIXES, ADJ_SUFFIXES, VERB_SUFFIXES):
        for suf in suffixes:
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                stem = w[: -len(suf)]
                out.append(stem)
                out.append(stem.translate(UMLAUT))
                out.append(stem + "en")           # verb infinitive
                out.append(stem + "e")
                out.append(stem.translate(UMLAUT) + "en")

    # Past participle: ge...t / ge...en, with or without a separable prefix.
    m = re.match(r"^(.*?)ge(.+?)(t|en)$", w)
    if m:
        prefix, stem = m.group(1), m.group(2)
        out.append(stem + "en")
        out.append(prefix + stem + "en")

    return _dedupe(out)


def lemma_keys(surface):
    """Glossary keys to try, both noun (N:) and word (W:) namespaces."""
    keys = []
    for form in base_forms(surface):
        keys.append("N:" + form)
        keys.append("W:" + form)
    return keys
