# reader-glossing

Turns a German epub into the glossed chapter JSON that a reading app serves, and
re-uses every word ever defined so a new book only costs tokens for what's
actually new. Built to read German novels with vocabulary help on a self-hosted
e-reader; 217 chapters and ~29,900 lemmas through it so far.

> **What this document is.** The operating manual for the pipeline, addressed to
> the LLM that runs it. The toolchain is deliberately split so that the
> deterministic work (extraction, sentence splitting, candidate selection,
> merging, glossary rebuild) is Python, and only the judgement call that actually
> needs a language model — *which words does a B1 reader need, and what is the
> entry* — is delegated. §3–§5 are the contract that delegation runs under: what
> the model receives, what it must return, and the checks a chapter has to pass
> before it counts as done.

**To run a session, say:** *"Read README.md and gloss ch05"* — or `ch05-ch08`, or
"the next unglossed chapter". Everything below should then happen without further
instruction.

Kept separate from the dashboard app on purpose: the app only reads finished
`chNN_glossed.json` files, so none of this ships to the server.

---

## 1. Where things are

One working root — this repo, on the Mac. Everything runs locally; the server is
only where finished chapters end up. Paths come from `scripts/paths.py`;
`GLOSSING_ROOT` moves the whole tree.

| Path | What | In git |
|---|---|---|
| `scripts/` | the toolchain | yes |
| `source/` | epubs as dropped in | no |
| `chapters/<Book>/chNN.json` | extracted text (intermediate) | no |
| `work/<Book>/<chNN>/` | batches + model output (scratch) | no |
| `results/<Book>/chNN_glossed.json` | **finished chapters — copied to the app by hand** | no |
| `live/<Book>/chNN_glossed.json` | mirror of the books already shipped | no |
| `glossary.db` | the glossary (derived, rebuildable) | no |

Only `scripts/` is in git; everything else is book text or derived from it.

`live/` is a local mirror of the directory the Reader app serves from, kept
because the glossary is built from the finished books *plus* `results/` — without
it the backlog vanishes and every chapter gets expensive again. `paths.py` uses
the served directory when it is mounted (`$READER_LIVE`) and this mirror
otherwise, so the same commands work on either machine.

**These scripts never write to the app directory.** Finished chapters land in
`results/` and are copied to the server by hand — deliberately, so a bad run
cannot reach the app:

```bash
scp "results/<Book>/chNN_glossed.json" <server>:"$READER_LIVE/<Book>/"
```

After shipping a chapter, drop a copy into `live/` too, so the local mirror keeps
matching the server.

Current books: `E.T.A. Hoffmann - Der goldene Topf` (12 Vigilien) and
`E.T.A. Hoffmann - Der Sandmann` (9 chapters) — both finished.

## 2. The loop, per chapter

```bash
cd ~/Documents/Codebase/reader-glossing
BOOK="chapters/E.T.A. Hoffmann - Der goldene Topf"

# 1. build the batch — everything the glossary already knows is resolved locally
python3 scripts/make_batches.py --chapter "$BOOK/ch05.json"

# 2. read the batch, write the glosses (§3, §4)
#    work/<Book>/ch05/batch_001.json  ->  work/<Book>/ch05/glossed_001.json

# 3. merge -> results/<Book>/ch05_glossed.json
python3 scripts/merge_glossed.py --chapter "$BOOK/ch05.json"

# 4. fold the new words in, so the next chapter is cheaper
python3 scripts/build_reader_glossary.py
```

Every path defaults correctly — `--chapter` is normally the only argument.

Step 4 is not optional; it's the whole economics. The builder reads `live/` *and*
`results/`, with `results/` winning, so a freshly merged chapter supersedes the
shipped copy.

**Only finished chapters belong in `results/`.** `--allow-partial` will happily
write a half-glossed file there, and the copy is made from `results/` without
re-checking — so delete any partial output when you're done iterating.

### Merging is not idempotent — don't re-merge casually

`merge_glossed.py` sorts the glossary's candidates by `freq` and keeps the rarest
two per sentence. Those frequencies change every time step 4 runs, so **the same
inputs merged later can pick a different pair of glossary entries** and produce a
slightly different chapter. Both versions are correct and both stay in band; they
just aren't identical.

Two rules follow:

- **Never re-merge a finished chapter to "check something".** Merge to a scratch
  path instead: `--out /tmp/check.json`. A bare re-merge overwrites `results/`,
  and if that chapter was already copied to the server the two silently diverge.
- **Keep `work/<Book>/<chNN>/` until the chapter is truly done.** Deleting it
  leaves no way to re-merge — the model output is gone and the chapter can only be
  recovered by glossing it again from scratch. Deleting scratch is cheap to
  postpone and expensive to undo.

### Re-glossing a chapter that is already merged

The chapter's own vocabulary is in the glossary, so a plain `make_batches.py` run
resolves its words against itself and yields a hollow batch. Take it out first:

```bash
mv "results/<Book>/chNN_glossed.json" /tmp/hold.json   # and live/ if it's there
python3 scripts/build_reader_glossary.py               # glossary without chNN
python3 scripts/make_batches.py --chapter "$BOOK/chNN.json"
# gloss, merge, then rebuild again
```

Expect fewer candidates than the first time round: every chapter glossed since has
been feeding the glossary, so more of chNN now resolves for free.

## 3. What the model reads and writes

Worked examples of all three files live in [`samples/`](samples/) — four real
sentences, the only book text in the repo. The shapes below are the contract.

**Reads** `batch_NNN.json`:

```json
{"chapter": "ch05", "batch": 1, "sentences": [
  {"i": 0, "text": "<the sentence>", "new": ["Wort", "Anwesen"]}
]}
```

`i` is the sentence index. `new` lists the words the glossary couldn't resolve —
the only ones worth considering.

**Writes** `glossed_NNN.json`, keyed by that same index:

```json
{"entries": {
  "0": [["das Anwesen, -", "ein großes Haus mit dem Land, das dazugehört", "birtok, kúria"]]
}}
```

**Never echo the sentences back.** They're ~40% of a naive response and carry no
information — the merge re-attaches them from disk. Sentences with nothing worth
glossing are simply absent from `entries`.

## 4. The entry contract

Three strings: `[term, de, hu]`.

### Which words

Gloss what a B1 learner would stumble on, in the sense **used in that sentence**.
Aim for roughly **2 entries per sentence** — the corpus average, and it reads well.

`new` is a candidate list, not a to-do list. Silently skip:

- **proper nouns** — characters, places, invented terms
- **interjections and fragments** — stray single letters, clipped colloquial forms
- **anything still below A2+** that slipped the filter
- **words already glossed earlier in the same batch**

### Which shape

| Kind | Shape | Example |
|---|---|---|
| Noun | `<der/die/das> <Noun>, <plural>` | `der Gärtner, -` · `die Hütte, -n` · `der Anlass, -̈e` |
| Verb | `<infinitiv>, <präteritum>, <hat/ist + partizip>` | `tratschen, tratschte, hat getratscht` |
| Separable | particle in its natural place | `vorfinden, fand vor, hat vorgefunden` |
| Reflexive | keep `sich` throughout | `sich lohnen, lohnte sich, hat sich gelohnt` |
| Phrase | citation form; verb forms if it has a verb | `zu dem Schluss kommen` |
| Adjective / adverb | bare | `löchrig` · `bitterernst` |

Two rules much of the older corpus breaks, so don't copy it:

- **always give the noun plural** — `-` for no change, `-̈er` / `-̈e` for umlaut.
  Omit only for genuine uncountables.
- **always give `hat` or `ist`** — only 39% of existing verb entries have it.

`de` — plain learner German, no article, not a sentence, ~25 chars. Don't reuse the
headword in its own definition.

`hu` — short, ~15 chars. Two or three senses comma-separated, not a thesaurus.
Mirror the German framing where it matters: `stolz (auf etw.)` → `büszke (vmire)`.

## 5. Check before reporting

```bash
python3 -c "import json;json.load(open('work/<Book>/ch05/glossed_001.json'))"
python3 scripts/merge_glossed.py --chapter "$BOOK/ch05.json"
```

The merge refuses to write in two cases, both of them things nothing else catches:

- **A letter from outside the Latin block.** Homoglyphs — a Cyrillic `а` for `a`,
  Arabic seen for `s` — look identical on screen and pass every other check;
  thirteen reached the corpus before this existed, and one corrupted a term's
  lemma so the word could never be matched again. Retype the field rather than
  pasting it back.
- **The same term glossed twice in one chapter.** The per-sentence de-duplication
  below it only looks inside a single sentence, so a repeat across two sentences
  used to pass straight through — one chapter shipped `anschauen` twice with
  competing definitions. Keep the better entry and delete the other.

The merge prints vocab-per-sentence. **Expect 1.7–2.3.** Outside that, investigate
rather than shipping:

- **too high** → over-glossing, or `--max-db-vocab` was raised
- **too low** → candidates skipped too aggressively, or a batch is missing

### The band is per sentence, but the real invariant is per character

1.7–2.3 is a *density*, and it only reads as a sentence count because the corpus
it was derived from averages 88 chars per sentence. The register-neutral figure is
**2.17 vocab per 100 chars** across all 64,208 sentences of that corpus — stable
book to book (2.72 down to 1.89) even as the per-sentence number drifts.

So **check a new author's chars/sentence before trusting the band.** Hoffmann
averages 183 — 2.07× — and the same reading density there is 3.95 per sentence,
i.e. a rescaled band of **3.5–4.75**. Applied literally, the 1.7–2.3 band would
have been satisfied by glossary re-use alone (1.85/sentence at the default cap) on
the hardest text in the corpus, discarding all 165 real candidates in ch01.

Rescale it as:

```
band_lo = 1.7 * (chars_per_sentence / 88.4)
band_hi = 2.3 * (chars_per_sentence / 88.4)
```

**Spend the budget on model entries, not DB re-use.** Raising `--max-db-vocab` is
the tempting way to hit a rescaled band, but the DB contributes rarest-first from
whatever the *previous* books happened to know — filler, on a new author. In
Hoffmann ch01 a cap of 4 would have supplied 3.53/sentence on its own and left
room for only 83 of the 165 candidates. Keep the cap low, gloss the candidates,
and use the cap only to trim the remainder.

### What moves the ratio

- **Chapter length predicts a second pass, not stoplist rate.** The floor scales
  with sentence count while DB re-use per sentence does not, so a 650-sentence
  chapter needs ~1,120 vocab items where a 150-sentence one needs 250. The fix is
  always the same: load the long descriptive sentences, never pad the short ones.
- **Over the ceiling, drop transparent compounds** — words a reader can decode
  from their parts — not opaque vocabulary.
- **Under the floor on a mature glossary,** reaching the band means deliberately
  re-glossing surface forms the DB holds under a different inflection: `höre`,
  `bat`, `komme`, `nehmt`, `flieh`, `darfst`.
- **Some prose shapes simply starve the ratio.** Fragment-heavy passages split
  into short sentences carrying nothing glossable; phonetically transcribed
  accents and dialect render candidates that aren't German words; proper-noun-dense
  scenes burn candidate slots on names. A low ratio with a stated reason beats an
  inflated one.
- **Re-use builds within a book as well as across.** Courtroom vocabulary glossed
  in one chapter cut the next chapter's distinct candidates from 310 to 192 at
  near-identical length.

Report chapter, sentences, entries from the glossary vs from the model, and
vocab/sentence. Flag anything odd rather than burying it.

## 6. Already decided — don't redo these

- **No OCR correction.** The epubs in use were audited and are clean. The standard
  fixes actively damage them: hyphen-plus-space hits are legitimate German
  suspended compounds that a dehyphenator would weld shut, and the "stray
  characters" are French dialogue. Audit a *new* book before assuming it needs
  anything.
- **Stoplist cutoff is 1200.** Every word it drops is one the early chapters
  already declined to gloss. Lower it only if real glosses start going missing.
- **Glossary re-use is capped at 2 per sentence,** rarest first. Uncapped it hits
  3.3 vocab/sentence and buries the page — cheaper in tokens, worse to read.
- **Morphological matching is on** (`bremste` → `bremsen`), but only to *skip* words —
  never to put an entry on the page. `german.py` is tuned for the skip decision,
  where its ~7% error rate costs a gloss; re-using those same hits as displayed
  entries turned that 7% into wrong glosses the reader sees. `Schlangen` stems to
  `schlang` and picked up a legacy entry for the preterite of `schlingen`, which
  shipped in Hoffmann ch01. A stemmer this cheap cannot tell that from the correct
  `Entsetzliches` → `entsetzlich`; both are two-character truncations. Costs ~5% of
  re-used entries, because the rarest-first cap drops most fuzzy hits anyway.
  `--morph-display` restores the old behaviour, `--exact-only` turns matching off
  entirely.
- **The same DB term may legitimately repeat across sentences** in one chapter — 33–42
  times in a long chapter, 22–29 per Hoffmann vigil. That is the word occurring twice
  and being glossed at both places, and it never carries competing definitions. The
  §5 duplicate guard deliberately checks only the model's entries, where a repeat
  *does* mean two competing definitions.
- **Sentence splitting is validated** at 99.1% against the 21,679 known-good
  sentences (`validate_splitter.py`). German uses `»…«` and a closing `«` stays
  welded to its sentence.

## 7. Adding a book

Drop the epub in `source/`, then:

```bash
python3 scripts/extract_epub.py --epub "source/<file>.epub" \
  --book "<Book Title>" --chapter-pattern "ch[0-9]+_" --dry-run
```

Check the count and titles, drop `--dry-run`, then start at §2. The spine often
carries teaser chapters from the *next* book — tighten `--chapter-pattern` until the
count is right. If the book might be a scan, audit it before adding any correction.

### When chapters aren't their own spine documents

`--chapter-pattern` can only pick whole spine entries, and Project Gutenberg epubs
chunk by byte size rather than by chapter — *Der goldene Topf* puts Vigilien 1–9 in
one document, 10–12 in the next, and the licence in a third. Use `--split-heading`
to cut *within* a document instead:

```bash
python3 scripts/extract_epub.py --epub "source/Der Goldene Topf.epub" \
  --book "E.T.A. Hoffmann - Der goldene Topf" \
  --chapter-pattern "17362-h-[01]\.htm" \
  --split-heading 'id="pgepubid000(0[1-9]|1[0-2])"' --dry-run
```

A section runs from its heading to the next match, and anything before the first
heading is dropped — which is what removes PG's boilerplate header. Match the
headings precisely: keying on `<h2` alone would also have caught PG's own
"The Project Gutenberg eBook of…" heading and made it ch01.

### When the epub has no headings at all

Gutenberg's plain-text conversions (`NNNN-8.txt` rendered to xhtml) carry no
`<h1>`–`<h6>` anywhere: every line, story titles included, is a bare `<p>`, and
the whole book sits in three or four byte-sized spine documents. `--split-tag p`
makes `--split-heading` cut on paragraphs instead, and `--stop-heading` drops the
matched section and everything after it — the licence, or the next story in an
anthology:

```bash
python3 scripts/extract_epub.py --epub "source/Nachtstuecke.epub" \
  --book "E.T.A. Hoffmann - Der Sandmann" \
  --chapter-pattern "6341-8-0\.txt\.xhtml" --split-tag p \
  --split-heading 'id="id000(08|24|31|34|38|40|44|45|47|52)"' \
  --stop-heading 'id="id00052"'
```

With `--split-tag p` a matched paragraph becomes the chapter title only if it is
at most 80 characters — otherwise it is the chapter's first paragraph and stays
in the body. Splitting on paragraph ids means the split points are positions in
one particular file, so re-run the dry run after any re-download.

Low paragraph counts are not necessarily a bug. This book extracts to 2–7
paragraphs per chapter because the source really does wrap thousands of characters
in a single `<p>` — 1814 prose keeps dialogue inline instead of breaking per
speaker. Check the `<p>` count in the source before assuming the parser dropped
something.

## 8. Why this is cheap

A word already in the glossary costs **zero tokens**: it never enters a prompt and
never comes back in a response — the merge attaches it from disk. In practice that
plus the stoplist keeps **~91% of the text away from the model**.

Per median chapter, estimated from character counts:

| | Input | Output | Total |
|---|---:|---:|---:|
| Naive (what the earliest chapters cost) | ~12k | ~32k | **~44k** |
| With the glossary + index-keyed output | ~12k | ~11k | **~23k** |

Input is a floor — the model must see the sentences to judge context. Output is
where it's won, via three levers in order of size: never ask for a word the
glossary has; never echo sentences back; never paste the 13MB glossary into a
prompt (query it, inject nothing).

Reuse compounds book over book — **0% → 38% → 50%** — because the corpus is
Zipf-shaped: the 1,000 commonest lemmas account for 17,752 of 43,319 entries, and
the stoplist head alone covers 84% of running words.

Batch by projected output, not sentence count: budget ~30 tokens per entry and
target ~250 new entries per batch.

## 9. Scripts

| Script | Does |
|---|---|
| `extract_epub.py` | epub → `chapters/<Book>/chNN.json`, spine order, sentence splitting |
| `make_batches.py` | resolves what the glossary knows; emits only unknown words |
| `merge_glossed.py` | model output + glossary hits → `results/<Book>/chNN_glossed.json` |
| `build_reader_glossary.py` | rebuilds `glossary.db` from every glossed chapter |
| `sentences.py` · `german.py` · `paths.py` | splitting, cheap morphology, path config |
| `validate_splitter.py` | round-trips the splitter against the known-good corpus |

The glossary is derived data — safe to delete, one command to restore. Three
tables: `entries` (every gloss, keyed by lemma), `occurrences` (which chapter each
came from), `stoplist` (word forms never once glossed — the empirical A2+ bar).

Known wrinkles in the existing corpus, so new output isn't benchmarked against
them: 39% of verb entries carry the auxiliary, 66% of nouns carry a plural, and
5,615 lemmas have more than one sense row (2.23 rows/lemma — when re-using, take
`ORDER BY freq DESC LIMIT 1`).

## 10. Status

217 chapters glossed, all finished; nothing outstanding. Glossary: 29,881 lemmas,
154,237 occurrences.

### Der Sandmann — 9 chapters, 555 sentences, 1,728 entries

The second Hoffmann, and the first book glossed against a glossary that already
held the author. Overall **3.11 per sentence / 2.20 per 100 chars** — the
per-character figure lands on *Der goldene Topf*'s 2.19 and the corpus's 2.17,
while the per-sentence figure differs by a quarter, which is the §5 point stated
as a measurement rather than an argument.

|  | Title | Sentences | Vocab | Per sentence | Band | Per 100 chars | cap |
|---|---|---:|---:|---:|---:|---:|---:|
| ch01 | Nathanael an Lothar | 159 | 443 | 2.79 | 2.25–3.05 | 2.38 | 3 |
| ch02 | Clara an Nathanael | 38 | 151 | 3.97 | 3.49–4.72 | 2.19 | 4 |
| ch03 | Nathanael an Lothar | 33 | 68 | 2.06 | 1.94–2.63 | 2.04 | 2 |
| ch04 | | 44 | 184 | 4.18 | 3.55–4.80 | 2.27 | 4 |
| ch05 | | 53 | 204 | 3.85 | 3.10–4.19 | 2.39 | 4 |
| ch06 | | 55 | 185 | 3.36 | 3.16–4.27 | 2.05 | 4 |
| ch07 | | 69 | 200 | 2.90 | 2.58–3.49 | 2.16 | 3 |
| ch08 | | 62 | 184 | 2.97 | 2.88–3.90 | 1.98 | 3 |
| ch09 | | 42 | 109 | 2.60 | 2.51–3.40 | 1.99 | 3 |

**The chapters are cuts, not the author's.** The novella has no chapter
divisions. ch01–ch03 are the three letters and carry their own headings; ch04–ch09
are splits made at paragraph breaks to keep chapters between 5.5k and 9.4k
characters, and they have no titles because inventing one would be worse than
leaving it empty.

**Sentence length swings even harder than in the first book** — 100.9 chars in
ch03 against 184.4 in ch04, a factor of 1.83 inside one novella — so the band was
computed per chapter throughout. The caps that followed ran 2–4, lower than *Der
goldene Topf*'s 3–5, because more of the text now resolves from the glossary.

**Same-author re-use is real but smaller than it looks.** Stoplist absorption
(54.9–61.9%) and candidate rates (8.6–12.5%) stayed in the range the first
Hoffmann set, against the modern corpus's 73–77% and 3–6%. What did change is the
DB's *usefulness* per hit: ch01 merged at 2.79 with a cap of 3 where *Der goldene
Topf* ch01 needed the same cap to reach 4.15 on much longer sentences.

**Coppola's accent is dead weight, as §5 predicts.** The Italian optician speaks
phonetically transcribed German — `sköne Oke`, `nix`, `Brill`, `Nas`, `su`,
`Peipendreher` — and every one of those surfaces as a candidate that cannot be
glossed. ch06 and ch09 carry most of them. Old orthography behaves as in the
first book: `daß`, `muß`, `häßlich`, `Hülfe` miss both stoplist and glossary and
are skipped as spelling variants.

**What this book needs glossed** is the optical and mechanical vocabulary the
plot turns on (`das Perspektiv`, `der Optikus`, `das Räderwerk`, `das Triebwerk`,
`der Automat`, `aufziehen`), Biedermeier dress and household (`das Beinkleid`,
`der Haarbeutel`, `die Lorgnette`, `die Magd`), and the musical terms of the ball
scene (`die Roulade`, `die Kadenz`, `der Triller`, `die Bravour`).

No chapter fired the duplicate-term or charset guard.

### Der goldene Topf — 12 Vigilien, 1,000 sentences, 3,996 entries

The only book in the repo that can be discussed by name, and the first by an
author the glossary had never seen. It moves every number the earlier chapters
established. Glossed against the **per-chapter rescaled band of §5**, not 1.7–2.3.
Overall **4.00 per sentence / 2.19 per 100 chars**, against the earlier corpus's
1.91 and 2.17: the same reading density in sentences twice as long.

|  | Title | Sentences | Vocab | Per sentence | Band | Per 100 chars | cap |
|---|---|---:|---:|---:|---:|---:|---:|
| ch01 | Erste Vigilie | 68 | 282 | 4.15 | 3.65–4.94 | 2.18 | 3 |
| ch02 | Zweite Vigilie | 96 | 365 | 3.80 | 3.70–5.00 | 1.98 | 3 |
| ch03 | Dritte Vigilie | 62 | 276 | 4.45 | 3.79–5.13 | 2.26 | 4 |
| ch04 | Vierte Vigilie | 64 | 326 | 5.09 | 4.74–6.41 | 2.07 | 5 |
| ch05 | Fünfte Vigilie | 108 | 445 | 4.12 | 3.56–4.81 | 2.23 | 4 |
| ch06 | Sechste Vigilie | 98 | 395 | 4.03 | 3.33–4.50 | 2.33 | 4 |
| ch07 | Siebente Vigilie | 89 | 334 | 3.75 | 3.17–4.28 | 2.28 | 4 |
| ch08 | Achte Vigilie | 92 | 447 | 4.86 | 4.21–5.69 | 2.22 | 5 |
| ch09 | Neunte Vigilie | 93 | 278 | 2.99 | 2.78–3.75 | 2.07 | 3 |
| ch10 | Zehnte Vigilie | 79 | 293 | 3.71 | 3.42–4.63 | 2.08 | 4 |
| ch11 | Elfte Vigilie | 79 | 295 | 3.73 | 3.02–4.09 | 2.38 | 4 |
| ch12 | Zwölfte Vigilie | 72 | 260 | 3.61 | 3.11–4.21 | 2.23 | 4 |

**The band moves per chapter here, so compute it per chapter.** Hoffmann's
sentence length swings far more — 144 chars in ch09 against 246 in ch04 — so a
single fixed band would have been wrong at both ends. `--max-db-vocab` is the
balancing knob: pick it *after* glossing, so the model's entries set the density
and DB re-use fills the remainder. It ranged 3–5 across the book.

**Re-use collapses on a new author.** Candidate rates run **7–12%** of running
words against the earlier corpus's 3–6%, and the stoplist absorbs only **58–63%**
against 73–77%. A 29,000-lemma glossary built entirely on one modern author covers
far less of 1814 prose than the raw lemma count suggests. Within the book re-use
did build: ch01 needed 165 distinct candidates, ch12 only 108, and the model's
share fell from 110 entries to 51.

**The old orthography inflates the candidate rate with non-vocabulary.** The text
is pre-1996 (`daß`, `muß`, `wußte`, `häßlich`), so these miss both the stoplist and
the glossary, which hold the modern spellings, and surface as candidates. They are
spelling variants rather than words — skip them under §4's sub-A2 rule. Watch also
for Hoffmann's nonce coinages (`rischelnd`, paired with `raschelnd`) and the split
fragments his spaced-out dashes produce (`Her u — u — u nter` yields `u`, `nter`).

**What this book actually needs glossed** is Romantic-era officialdom
(`Konrektor`, `Registrator`, `Speziestaler`, `Hofrätin`), alchemy and occultism
(`Salamander`, `Nekromant`, `Karfunkel`, `Auripigment`), and a steady run of
veraltet forms a B1 reader cannot decode (`unerachtet`, `daselbst`, `nimmermehr`,
`annoch`, `dorten`, `ohnedies`).

No chapter fired the duplicate-term or charset guard.
