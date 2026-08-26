# Format samples

Four consecutive sentences from a real run, showing each file the pipeline
passes along. The text is E.T.A. Hoffmann's *Der goldene Topf* (1814) — public
domain, and the same German the toolchain was built against.

These are the only book text in the repo; everything else (`source/`,
`chapters/`, `work/`, `results/`, `live/`, `glossary.db`) is gitignored and
never leaves the working machine.

| File | Who writes it | Who reads it |
|---|---|---|
| `batch_001.json` | `make_batches.py` | the model |
| `glossed_001.json` | the model | `merge_glossed.py` |
| `chNN_glossed.json` | `merge_glossed.py` | the Reader app |

The sentence indices start at 24 because a batch is a slice of a chapter, and
`i` is what the merge keys on — it is the chapter position, not a counter that
restarts per batch.

Four rules that are easy to get wrong:

- **Sentence 27 is absent from `glossed_001.json`.** Its `new` list is empty, so
  there was nothing to consider and it simply doesn't appear. Sentences with
  nothing worth glossing are omitted, not returned with an empty list.
- **`new` is a candidate list, not a to-do list.** Sentence 24 offers
  `Pappelallee` and `Koselschen`; only *die Pappel* comes back, because
  `Koselschen` is a Dresden place name and glossing it would teach nothing.
  Sentence 25 goes further: `Himmelswillen` and `solcher` are two separate
  candidates, but what a learner actually needs is the phrase *um Himmels
  willen*, so the two collapse into one entry. Sentence 26 does the same in
  reverse — the candidate is `gewurzelt`, and the entry is the idiom it belongs
  to, *wie angewurzelt*.
- **The model never echoes the sentences back.** `glossed_001.json` is keyed by
  the `i` from the batch and holds nothing else; the merge re-attaches the text
  from disk. Echoing it back would roughly double the output.
- **The merged output carries more vocab than the model wrote.** Every sentence
  here picks up entries the glossary already knew — *einbiegen*, *der Herr*,
  *rennen*, *überzeugt*, *das Unglück*. Those cost zero tokens and never entered
  a prompt. Sentence 27 is glossed in the final file despite being omitted by
  the model, entirely from re-use.

One caveat on the third file: entries re-used from the glossary take the most
frequent sense of a word, which is not always the sense in *this* sentence —
`Tausend` on sentence 25 is glossed as the number, though it is doing idiomatic
work in *um tausend Himmelswillen*. That is a known limit of re-use, not a
template to copy; entries the model writes are held to the sense actually used,
per §4 of the main README.
