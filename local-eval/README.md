# local-eval — can a local model do the glossing?

A bench for replacing the frontier model in `../README.md` §2 step 2 with
something that runs on the Mac. It answers one question: **what does each
candidate actually lose?**

Nothing here touches `results/`, `live/` or `glossary.db`. Only `harness/` is in
git; the gold set and runs contain book text, and `models/` is ~30 GB of
downloads.

## The bench in one paragraph

Every `batch_NNN.json` in `../work/` has a `glossed_NNN.json` beside it — the
exact input a frontier model saw and the output it produced, **210 paired
files**. That is a ready-made gold set, so a local model is measured
against real accepted output rather than a hand-written rubric.

## Glossing is three jobs, and they are scored apart

A single quality score would hide the distinction that decides the design:

| | Sub-task | Example | Could a lookup do it? |
|---|---|---|---|
| **A** | **Select** which candidates deserve an entry | skip `Ladenfronten`, keep `gellen` | no — judgment |
| **B** | **Citation form** | `der Vorhang, -̈e` · `zuziehen, zog zu, hat zugezogen` | **yes — it is morphology** |
| **C** | **Content**: the German definition and the Hungarian | `einen Vorhang schließen` / `behúz` | no — and C is where Hungarian lives |

If a model scores well on A and C but badly on B, the answer is not a bigger
model, it is taking B away from the model — hence `build_lexicon.py`. If it
fails C, no scaffolding helps.

## End-to-end result

`run_pipeline.py` runs all four stages over the same 60 tasks and writes a run
file `score.py` grades exactly like any model's:

| | entries | JSON ok | B:form | invented Hungarian | on-candidate |
|---|---:|---:|---:|---:|---:|
| frontier corpus | 424 | 100% | 88.0% | 0.9% | 69.3% |
| **pipeline** | **320** | **100%** | **100%** | **0.0%** | **100%** |
| monolithic 26B | 3 of 4 tasks | 50% | 100% | 31.6% | 100% |

Five fixes took it from a first draft of 168 entries at 4.2% non-words:

1. **Lemmatise before every lookup** (`lemma.py`). The pipeline was looking up
   `angenommen`, `abzugeben`, `aufgebrochen` — the dictionary keys on
   `annehmen`, `abgeben`, `aufbrechen`. This alone recovered 118 of 373 misses.
2. **Follow `ld.` redirects instead of emitting them.** `trieb → ld. treiben`
   is Hungarian for "see: treiben", and shipping it put the German words
   *treiben*, *waschen* and *wissen* into the Hungarian field.
3. **Strip domain labels** — `(átv.)` figurative, `(kat)` military, `(átv. is)`.
   Keyed on the abbreviation, not a trailing dot, which missed two forms of
   three. Government markers like `(vmire)` are kept: the house style wants them.
4. **Respect case on compound heads.** `Laden` (shop) and `laden` (to load)
   collapse under a lowercased key, which is how `Antiquitätenladen` came out
   as `hív` (to call). Now `ablaktábla, bolt`.
5. **Hunspell as a filter, not just a metric.** The source dictionary has its
   own typos — `probláma`, `aláterí`. Because it usually offers several senses,
   the misspelled one is dropped and the entry survives.

Compound splitting is now restricted to words the lexicon calls nouns.
Unrestricted it produced confident, real, wrong words that no checker can
catch: `bergeweise` → `weise` → *bölcs* (wise), `Artischocke` → `Schocke` →
*öt tucat* (five dozen).

**Speed.** Only 19% of definitions actually need the model — 24% of Wiktionary
glosses are already short enough and the first-clause trim handles another 47%.
Calling the model only on the remainder cuts a median chapter from ~139 min to
~50, and a 37-chapter book from ~86 h to ~31 h.

### Two obvious improvements, both tested, both rejected

**Generating the missing Hungarian and gating it with hunspell does not work.**
The gap is 204 words with no dictionary translation, so generation looks like
the way to close it. On 120 gap words TranslateGemma produced something every
time and **76.7% passed the spellchecker — but only 5% were correct**:

| | |
|---|---:|
| produced something | 100% |
| survives hunspell | 76.7% |
| agrees with the corpus | **5.0%** |

`kopfüber aufhängen` → *Szöveges zaklatás* (textual harassment). `hinabspähen`
→ *Idővesztés* (time loss). `auskommen mit` → *Okosan viszonozni* (wisely
reciprocate). All fluent, all real words, all wrong.

**hunspell catches non-words, not wrong meanings.** What makes the dictionary route safe is not that errors get
filtered downstream — it is that the dictionary never proposes a wrong meaning
in the first place. Generation plus a spellchecker is not a substitute, and
shipping it would put ~70% wrong entries into the glossary looking perfect.

**Picking the Wiktionary sense by sentence context does not work either.**
Scored against the corpus's own German definitions over 340 cases:

| choosing the gloss | agrees |
|---|---:|
| always sense 1 | 36.2% |
| picked by overlap with the sentence | **35.6%** |
| best sense available (ceiling) | 47.1% |

Context selection is a hair *worse* than doing nothing, and the ceiling is only
47% because the corpus writes its own concise learner German rather than
reusing Wiktionary's phrasing. Cheap sense disambiguation has no headroom here.

### What is still weak

`C:hu` sits at 54.6%: the dictionary covers 55% of new candidates, so 204 of
683 selected words are dropped for having no Hungarian at all. And composing
two independent sources can cross senses — `tagen` gets Wiktionary's "Tag
werden" beside the dictionary's *ülésezik* (to hold a session), which are
different words in the same entry. Volume is 320 against the corpus's 424, so a
chapter glossed this way still lands under the §5 band.

## Running it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mlx-lm huggingface_hub

python3 harness/build_goldset.py --tasks 60 --sentences 10
python3 harness/gold_as_run.py                 # the frontier baseline

.venv/bin/python -m mlx_lm server --model models/<model> --port 8080 &
.venv/bin/python harness/run_model.py --name <label> \
    --base http://127.0.0.1:8080/v1 --model models/<model>

python3 harness/score.py runs/*.jsonl --detail
```

`run_model.py` speaks the OpenAI chat shape, which is the one thing every
runtime here agrees on — mlx-lm, llama-server and TurboFieldfareServer all
expose it, so every model gets an identical prompt.

## One model cannot do the whole job, and does not have to

Every model that fits in 8 GB failed the whole job, and the one that runs by
streaming off SSD failed it differently — it invented Hungarian words in
**31.6%** of its entries, against the frontier corpus's 0.9%. So the bench
grew a second half: instead of asking which model, ask **which source** should
answer each field.

| Field | Best source | Covers | Wrong |
|---|---|---:|---:|
| `term` | Wiktionary + CharSplit + separable-prefix rule | **93.8%** of nouns/verbs | 0.8% |
| `hu` | `dictionary.sqlite` (Balázs's own de-hu, 22k headwords) | 57.5% | never invents |
| `de` | Wiktionary gloss (encyclopedic — usually needs rewriting) | 63.9% | — |
| JSON | assembled in Python | 100% | — |

Nothing above needs a model, and each removes a failure the monolithic run
actually made. What is left for a model is narrow: which candidates to gloss,
which dictionary sense fits this sentence, and a learner-style German
definition.

### Ranking beats generating

Asked to *choose* a Hungarian sense rather than write one:

| | correct |
|---|---:|
| always take the dictionary's first sense (no model) | 45.0% |
| Gemma 4 E2B — fits in RAM | **40.0%** — worse than no model |
| Gemma 4 26B-A4B via TurboFieldfare | **65.0%** |

More important than the 65%: **an invented word is now structurally
impossible**, because every option came from the dictionary. All
20 wrong answers are real Hungarian near-synonyms — `ártalmas` for `káros`,
`szitkozódik` for `káromkodik`, `keleties` for `keleti` — where the monolithic
run produced `megjámul` and `házasmertek`. And it costs 2 output tokens instead
of ~285, so it runs ~15× faster (7–16s against 156s per call).

E2B's result is the other half: a model too small to do the job
is also too small to do a *piece* of the job. Splitting the task does not
rescue a weak model, it rescues a strong model from a task shape it handles
badly.

### TranslateGemma, used the way an MT model wants to be

Scoring it on bare citation forms (13.3%) was the wrong test — a word with no
context is an MT model's worst case. `stage_mt_rank.py` instead translates the
whole sentence and looks for which dictionary sense turns up in the result:

| | correct |
|---|---:|
| baseline, always sense 1 | 46.7% |
| TranslateGemma-4B, whole-sentence | 60.0% overall — **65.2% where it locates a sense** |
| Gemma 4 26B-A4B, asked to choose | 65.0% |

**A 2.1 GB model matches the 13 GB one on this stage.** It only locates a sense
in 38% of cases and falls back to sense 1 otherwise, but where it fires it is
as good, and it fits in RAM with room to spare.

It still cannot be *the* model. It has no chat mode at all — the template
accepts only `{source_lang_code, target_lang_code, text}` — so it cannot choose
which candidates to gloss, cannot write a German definition, and cannot emit
JSON. And its raw output is not shippable: 12.4% of the words in its sentence
translations fail hunspell, including truncations (`anélkü`, `küz`,
`leereskenek`) and one Chinese character sequence that would trip README §5's
charset guard. That does not matter here only because its text is never
shipped — it is used to *locate* a dictionary sense, and the dictionary
supplies the word that reaches the reader.

*Correction: the run behind that table used a case window shifted by one
against every other run in this file — `einsammeln` at the front, the 60th case
missing — so its baseline reads 46.7% where the shared window gives 45.0%.
Re-run on the shared window TranslateGemma scores 56.7%, not 60.0%. The table
below supersedes it.*

### Which model actually speaks Hungarian

`stage_mt_rank.py` scores any German→Hungarian translator, so the question
"which model is best at Hungarian" has an answer measured on this corpus rather
than on a leaderboard. The candidates were picked by whether Hungarian is a
training target or a scrape byproduct — the EU-funded multilingual models
(EuroLLM, Teuken), and dedicated MT (NLLB, TranslateGemma). All 60 cases, one
matcher, one window:

| | resident | locates a sense | correct where located | correct |
|---|---:|---:|---:|---:|
| baseline, always sense 1 | — | — | — | 45.0% |
| Gemma 4 E2B | 3.3 GB | — | — | 40.0% |
| EuroLLM-1.7B, Q8_0 | 1.7 GB | 40.0% | 62.5% | 53.3% |
| TranslateGemma-4B, 4-bit | 2.1 GB | 36.7% | 63.6% | 56.7% |
| NLLB-200-distilled-600M, fp32 | 2.4 GB | 36.7% | 77.3% | 58.3% |
| Teuken-7B, 4-bit | 4.2 GB | 55.0% | 75.8% | 61.7% |
| EuroLLM-9B-2512, Q3_K_M | 4.3 GB | 48.3% | 79.3% | 63.3% |
| **NLLB-200-distilled-1.3B, fp16** | **2.6 GB** | **53.3%** | **81.2%** | **65.0%** |
| Gemma 4 26B-A4B, asked to choose | 1.35 GB + SSD | — | — | 65.0% |

**A 1.3B translation model ties the 26B.** Not by being a better model — by
being asked a question it was built for. The ordering is not by size: the two
NLLB checkpoints beat every general model of their weight class and the 600M
beats a 7B, while EuroLLM-1.7B and TranslateGemma-4B sit at the bottom. What
separates them is `correct where located` — the models trained to translate are
right ~80% of the time when they commit, against ~63% for the instruct models,
which locate a sense confidently and locate the wrong one.

**They are complementary, not interchangeable.** Over the same 60 cases NLLB
and the 26B agree on 28, and then each gets **11 the other misses**; 10 defeat
both. Either model alone is 65.0%, and something that could pick the right one
of the two every time would reach **83.3%** — a bigger gap than anything left
between the candidates. That is the next experiment worth running, and it is
affordable, because NLLB-1.3B is 2.6 GB and the 26B is already streamed.

Two practical limits found on the way:

- **EuroLLM-9B does not fit at a quantisation that flatters it.** IQ4_XS
  (5.05 GB) dies on the first decode — Metal's working-set limit here is about
  5.4 GB, so the weights fit and the compute buffers do not. On the CPU with
  `-ngl 0` it did not finish one sentence in 600 s. Its 63.3% is Q3_K_M, and a
  9B at Q3 is being scored with a handicap.
- **Teuken needs `trust_remote_code`** — it ships its own tokenizer class in
  the repo. Read `gptx_tokenizer.py` before enabling it, as the flag runs
  whatever the checkpoint contains.

Not worth testing on this machine: DeepSeek. The distills are Qwen weights with
reasoning SFT, and the reasoning is the problem — this stage wants two output
tokens, and the whole 15× speed win comes from that. The real DeepSeek
architectures are 671B MoE and never fit. Llama 3.x does not list Hungarian
among its supported languages at all.

### The `de` field

Wiktionary has a German gloss for 63.9% of entries, but they average 61
characters against the corpus's 26, and 54% exceed the 52-char ship limit.
Cutting at the first clause brings the mean to 28.0 with 13% over — right
length, but it over-trims meaning often enough (`das Zubehör` → `Gegenstände`)
that a model earns its place here.

| shortening the gloss | mean chars | within spec |
|---|---:|---:|
| gold (frontier) | 20.1 | 97% |
| first-clause trim, no model | 29.9 | 90% |
| **Gemma 4 E2B** | **20.3** | **97%** |

**E2B produced 0% valid JSON on the whole task and matches the frontier model
on this one** — `büßen` → `Strafe zahlen`, `loben` → `Kompliment machen`,
`Zubehör` → `Zusatzteile`. The 3.3 GB model was never the problem; the task
shape was.

One catch worth knowing: it spends ~370 reasoning tokens to emit two words, so
it needs `max_tokens` in the thousands. Cut short it returns its chain of
thought and no `content` at all, which reads as total failure and is really
truncation.

## What is in here

| File | Does |
|---|---|
| `harness/build_goldset.py` | `../work/` → `goldset/tasks.jsonl` |
| `harness/prompt.py` | the glossing contract, compressed to what a small model can hold |
| `harness/run_model.py` | runs the tasks against one OpenAI-compatible endpoint |
| `harness/gold_as_run.py` | emits the frontier output in run format, as the baseline |
| `harness/score.py` | grades A, B and C separately, plus the README §5 ship-blockers |
| `harness/build_lexicon.py` | German Wiktionary → `lexicon.db` |
| `harness/stage_form.py` | the `term` field with no model — lookup, CharSplit, prefix rule |
| `harness/stage_hu.py` | the `hu` field from `dictionary.sqlite` |
| `harness/stage_rank.py` | can a model pick the right sense? baseline: always take sense 1 |
| `harness/stage_mt_rank.py` | the same stage by translating the sentence — `--backend` picks the model |
| `harness/translators.py` | German→Hungarian behind one interface: MLX, seq2seq MT, or an endpoint |
| `harness/bench_mt_gguf.sh` | serves a GGUF with llama-server and scores it, for weights MLX cannot hold |
| `harness/hu_wordcheck.py` | hunspell `hu_HU` — is the Hungarian even a word? |
| `harness/pipeline.py` | how much of an entry the deterministic layer fills |
| `harness/chapter_cost.py` | projected wall-clock per chapter |

`hu_wordcheck.py` needs `brew install hunspell` and the LibreOffice `hu_HU`
dictionary in `models/hu-dict/`; `stage_form.py` uses `compound-split` from
PyPI and degrades gracefully without it. The `nllb` backend needs `torch` and
runs fp16 on the GPU — the 1.3B checkpoint is fp32 on disk and gets OOM-killed
at that size. `bench_mt_gguf.sh` needs `brew install llama.cpp`.

## Two things the bench design had to account for

**Sentences with no candidates are dropped from the prompt.** They are 62% of
the sentences and 47% of the prompt characters but carry only 0.9% of the gold
entries. Sending them was nearly free against a cloud model; locally, prefill is
the whole cost.

**The gold set excludes one style of entry.** Two books in the corpus put the
*inflected token* in the term field (`Ladenfronten`) and folded the citation
form into the definition (`die Front, -en: die Vorderseite der Läden`), against
`../README.md` §4. 27% of the corpus is written that way and 5,426 glossary
rows are keyed on inflected forms because of it. Benchmarking against those
would score a model on reproducing a defect, so `build_goldset.py` drops them —
`--keep-folded` puts them back.

## Reading the scorecard

- **A:prec / A:rec / A:F1** — did it pick the same words the frontier model did.
- **B:form** — share of entries whose citation form is structurally sound:
  article present and correct (checked against `glossary.db` genders), plural
  present, verbs with three parts and `hat`/`ist`.
- **C:hu** — share of Hungarian glosses that share a token with what the corpus
  already uses for that lemma. Scoreable only where the corpus knows the word,
  but that is 50k entries deep.
- **ship-blockers** — the two things `merge_glossed.py` refuses to write:
  non-Latin homoglyphs and a term glossed twice.

`GOLD-frontier` scores A=100% by construction, which is what proves the matcher
pairs entries correctly. Its **B=88%** is the ceiling — the frontier runs
were not perfect either, and the residual is the known "66% of nouns carry a
plural" wrinkle from `../README.md` §9.
