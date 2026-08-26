"""German sentence splitting, tuned to the conventions in the existing Reader corpus.

Dialogue uses guillemets (»…«) and a closing « stays welded to the sentence it
ends. The hazards are abbreviations and ordinals, both of which put a period in
the middle of a sentence: 'Mr. Dursley', 'am 3. September'.

Validated by round-tripping the 21,679 already-split sentences on the server —
see validate_splitter.py.
"""

import re

# Abbreviations that take a period and are followed by a capitalised word, so the
# usual "period + space + uppercase" signal would wrongly fire.
ABBREV = {
    # titles / address
    "mr", "mrs", "ms", "dr", "prof", "hr", "fr", "st", "sankt",
    # common German abbreviations
    "z", "b", "d", "h", "u", "a", "s", "o", "ca", "bzw", "usw", "etc", "vgl",
    "ggf", "inkl", "exkl", "evtl", "bes", "geb", "gest", "jhd", "jh", "nr",
    "bd", "abb", "tab", "kap", "vs", "engl", "dt", "lat", "franz",
}

_TERM = r"[.!?…]"
_CLOSERS = r"[»«\"'’”\)\]]*"

# A boundary is terminal punctuation, optional closing quotes/brackets, whitespace,
# then something that can begin a sentence: a capital, an opening guillemet, a dash,
# or a digit.
_BOUNDARY = re.compile(
    r"(?<=" + _TERM + r")(" + _CLOSERS + r")\s+(?=[»„\"'A-ZÄÖÜÉÀÈ0-9–—-])"
)

_ORDINAL = re.compile(r"\d+\.$")
_INITIAL = re.compile(r"\b[A-ZÄÖÜ]\.$")


def _is_false_boundary(left):
    """True when the period ending `left` belongs to an abbreviation, not a sentence."""
    tail = left.rstrip()
    if not tail.endswith("."):
        return False  # ! ? … are never abbreviation marks
    if _ORDINAL.search(tail):
        return True  # 'am 3. September', 'im 19. Jahrhundert'
    if _INITIAL.search(tail):
        return True  # 'E. T. A. Hoffmann'
    word = re.split(r"[\s»«„\"'\(\[-]", tail[:-1])[-1].lower()
    return word in ABBREV


def split_sentences(text):
    """Split a paragraph into sentences. Returns a list of stripped strings."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    out = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        end = m.end(1)  # include any closing quote in the left-hand sentence
        if _is_false_boundary(text[start:end]):
            continue
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out
