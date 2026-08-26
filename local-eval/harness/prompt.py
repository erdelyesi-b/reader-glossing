#!/usr/bin/env python3
"""The glossing contract, as a prompt. Kept in one place so every model in the
bench is asked exactly the same thing and the comparison stays honest.

This is README §3 and §4 compressed to what a small model can actually hold.
The frontier runs had the whole README; a 4B does not follow a 400-line spec,
so the bench measures the realistic local prompt rather than an unusable one.
"""

import json

SYSTEM = """Du bist ein Glossar-Werkzeug für deutsche Romane. Ein ungarischer Deutschlerner (Niveau B1) liest den Text.

Für jeden Satz bekommst du eine Liste "new" mit Wörtern, die noch nicht im Glossar stehen. Wähle daraus die Wörter, über die ein B1-Lerner stolpern würde, und schreibe je einen Eintrag.

Überspringe still:
- Eigennamen (Personen, Orte, erfundene Begriffe)
- Interjektionen, Satzfragmente, einzelne Buchstaben
- alles unter Niveau A2
- Wörter, die du in diesem Auftrag schon glossiert hast

Ziel: rund 2 Einträge pro Satz. Sätze ohne lohnendes Wort lässt du einfach weg.

Ein Eintrag sind genau drei Strings: [term, de, hu]

term — die Zitierform, in der Bedeutung DIESES Satzes:
- Substantiv: "der/die/das Wort, Plural"   z.B. "die Hütte, -n" · "der Gärtner, -" · "der Anlass, -̈e"
  Der Plural ist Pflicht: "-" wenn unverändert, "-̈e"/"-̈er" mit Umlaut.
- Verb: "Infinitiv, Präteritum, hat/ist Partizip"   z.B. "tratschen, tratschte, hat getratscht"
  "hat" oder "ist" ist Pflicht. Trennbar: "vorfinden, fand vor, hat vorgefunden".
  Reflexiv behält sich: "sich lohnen, lohnte sich, hat sich gelohnt".
- Adjektiv/Adverb: nackt, ohne Endung   z.B. "löchrig"
- Wendung: Zitierform   z.B. "zu dem Schluss kommen"

de — einfache Lernerdeutsch-Erklärung, etwa 25 Zeichen, kein Artikel, kein ganzer Satz, und das Stichwort selbst kommt darin nicht vor.

hu — ungarische Übersetzung, etwa 15 Zeichen, ein bis drei Bedeutungen mit Komma. Die deutsche Rektion spiegeln, wo sie zählt: "stolz (auf etw.)" -> "büszke (vmire)".

Antworte NUR mit JSON in genau dieser Form, ohne Kommentar, ohne Markdown-Zaun, und gib die Sätze NICHT zurück:
{"entries": {"<satz-index>": [["term", "de", "hu"]]}}"""

USER = """{sentences}

Antworte nur mit dem JSON-Objekt."""


def build(task):
    """Return (system, user) for one bench task."""
    lines = []
    for s in task["sentences"]:
        lines.append(json.dumps({"i": s["i"], "text": s["text"], "new": s["new"]},
                                ensure_ascii=False, separators=(",", ":")))
    return SYSTEM, USER.format(sentences="\n".join(lines))
