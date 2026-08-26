"""Where everything lives.

One working root holds the lot — source epubs, extracted chapters, batch scratch,
and the glossary. The only things that ever leave it are the finished
chNN_glossed.json files, and even those land in results/ for a human to copy to
the server rather than being written into the live app directory.

    <root>/source/     epubs as dropped in
    <root>/chapters/   <Book>/chNN.json          extracted text (intermediate)
    <root>/work/       <Book>/<chNN>/            batches + model output (scratch)
    <root>/results/    <Book>/chNN_glossed.json  ← copy these to the server
    <root>/live/       <Book>/chNN_glossed.json  mirror of the finished books
    <root>/glossary.db

LIVE is read-only from this toolchain's point of view: it is where the already
finished books sit, and the glossary is built from those plus results/.
"""

import os

ROOT = os.environ.get(
    "GLOSSING_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

SOURCE = os.path.join(ROOT, "source")
CHAPTERS = os.path.join(ROOT, "chapters")
WORK = os.path.join(ROOT, "work")
RESULTS = os.path.join(ROOT, "results")
DB = os.path.join(ROOT, "glossary.db")

# The finished books the glossary is built from, alongside results/. Never written
# to by these scripts. On the server that is the directory the app serves from; off
# the server it is a local mirror of it, so the backlog still counts. Set
# READER_LIVE to the served directory; without it the local mirror is used.
SERVER_LIVE = os.environ.get("READER_LIVE", "")
LIVE = os.environ.get(
    "GLOSSING_LIVE",
    SERVER_LIVE if SERVER_LIVE and os.path.isdir(SERVER_LIVE)
    else os.path.join(ROOT, "live"),
)


def ensure():
    for d in (SOURCE, CHAPTERS, WORK, RESULTS):
        os.makedirs(d, exist_ok=True)
