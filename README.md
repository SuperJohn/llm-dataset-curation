# llm-dataset-curation

[![ci](https://github.com/SuperJohn/llm-dataset-curation/actions/workflows/ci.yml/badge.svg)](https://github.com/SuperJohn/llm-dataset-curation/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue) ![polars](https://img.shields.io/badge/engine-polars%20%2B%20duckdb-cd792c)

A config-driven pipeline for cleaning instruction or pretraining text before it goes
anywhere near a model. It normalizes, removes **exact and near duplicates (MinHash
LSH)**, applies **Gopher/C4-style quality heuristics** and language ID, **redacts PII**,
filters **toxic** content, and writes an **eval report** that shows exactly what was
removed and why.

```bash
uv sync
uv run curate run --config configs/sample.yaml   # offline, ~5 s
open output/dolly-sample/report.html
```

📊 **[Example report on all of databricks-dolly-15k →](docs/example-report/report.md)**

## Why this exists

Data quality is the highest-leverage, least glamorous part of training a model. This
repo is a compact, testable version of the curation stages used in large-scale
pipelines, built the way I'd want a production data job to look:

- **Declarative:** every threshold lives in YAML, and any of them can be overridden from the CLI
  (`--set stages.near_dedupe.threshold=0.7`).
- **Auditable:** every removed record keeps its stage and a human-readable reason
  (`near duplicate of dolly-01279 (J≈0.95)`, `repetitive top 2-gram covers 36% of text`),
  and every kept record carries its quality signals, so thresholds get tuned from data,
  not vibes.
- **Safe by construction:** PII is redacted *immediately after normalization*, so no
  downstream artifact (dedupe clusters, removed samples, the report) ever holds raw PII.
- **Tested against labelled noise:** the sample includes 33 synthetic "bad" records
  (whitespace and Unicode duplicates, light paraphrases, fake PII, other languages, spam,
  boilerplate, insults). A test asserts that each one is caught by the *right* stage.

## Pipeline

```mermaid
flowchart LR
    IN[(JSONL / Parquet / CSV<br/>config: input.paths)] --> N[normalize<br/>NFKC · control chars · whitespace]
    N --> P[PII redaction<br/>email · phone · IPv4 · card+Luhn · SSN]
    P --> E[exact dedupe<br/>SHA-256 of key fields]
    E --> NN[near dedupe<br/>MinHash LSH · 5-gram shingles<br/>verified Jaccard · union-find clusters]
    NN --> Q[quality filters<br/>length · word length · symbols<br/>alpha ratio · dup lines · top n-gram<br/>boilerplate · language ID]
    Q --> T[toxicity<br/>lexicon + optional classifier hook]
    T --> OUT[(curated.parquet / .jsonl)]
    E & NN & Q & T -.removed + reason.-> RM[(removed.parquet)]
    NN -.-> CL[(dup_clusters.parquet)]
    OUT & RM & CL --> R[report.html / report.md<br/>stats.json]
```

| Stage | What it does | Key config |
|---|---|---|
| `normalize` | Unicode NFKC, strips zero-width and control characters, collapses whitespace, rebuilds `text` from `input.text_fields` | `unicode_form` |
| `pii` | Regex redaction to typed placeholders (`<EMAIL>`, `<PHONE>`, …). Card numbers must pass Luhn. Optionally drops PII-dense records. | `types`, `action: redact\|drop`, `max_hits_before_drop` |
| `exact_dedupe` | SHA-256 over the chosen fields (case-insensitive) and keeps the first occurrence | `fields` |
| `near_dedupe` | MinHash (128 perms) over word 5-grams, LSH candidates **verified** by estimated Jaccard, and union-find clusters that keep the earliest record | `threshold`, `ngram`, `num_perm`, `min_words`, `fields` |
| `quality` | Gopher-style rules: word count, mean word length, `#`/ellipsis ratio, alphabetic-word fraction, duplicate-line fraction (per field), top 2/3/4-gram character coverage, banned boilerplate phrases, and language ID (py3langid restricted to common languages) | `min_words`, `max_top_ngram_char_fraction`, `language.*` |
| `toxicity` | Whole-word lexicon matching (phrases allowed) with hit count and fraction thresholds. Any `module:function` scorer can be plugged in (e.g. Detoxify). | `lexicon_paths`, `max_hits`, `classifier` |

For instruction data, the **identity of an example is `instruction + response`**, so
several questions about the same context passage are *not* duplicates. Language ID runs
on the **prompt**, because responses are often lists of proper nouns, and translation
tasks legitimately answer in another language.

## Results

| Run | In | Kept | Exact dup | Near dup | Quality | Toxic | PII redacted | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `configs/sample.yaml` (1,500 Dolly + 33 synthetic) | 1,533 | 1,482 | 22 | 11 | 16 | 2 | 14 records | ~5 s |
| `configs/dolly15k.yaml` (full dataset) | 15,011 | 14,939 | 16 | 10 | 45 | 1 | 11 records (22 hits) | ~15 s |

The [example report](docs/example-report/report.md) includes the funnel, the length
distribution before and after, near-duplicate cluster sizes with examples, PII hit rates
by type, and samples of removed records for each stage.

<p>
<img src="docs/example-report/report_assets/funnel.png" width="62%">
<img src="docs/example-report/report_assets/clusters.png" width="34%">
</p>

## Usage

```bash
make setup                    # uv sync
make run                      # curate the checked-in sample
make full                     # download databricks-dolly-15k (13 MB) and curate all of it
make test                     # ruff + pytest

uv run curate run -c configs/sample.yaml \
    --set stages.near_dedupe.threshold=0.7 \
    --set stages.mitigation.pii.action=drop
uv run curate validate -c configs/sample.yaml   # print the fully-resolved config
```

Outputs, written to `output.dir`:

```
curated.parquet / curated.jsonl   # kept records + quality signals + pii_hits / pii_types
removed.parquet                   # every removed record with _stage and _reason
dup_clusters.parquet              # cluster_id, member, representative, est. Jaccard
stats.json                        # funnel + per-stage stats + timings
report.html / report.md           # eval report (HTML is self-contained)
```

Bringing your own data takes one config: point `input.paths` at JSONL, Parquet, or CSV, and
set `id_field` and `text_fields`.

## Layout

```
src/curate/
  cli.py            # curate run | download | validate
  config.py         # pydantic models for the YAML config
  io.py             # readers, combined-text builder, writers
  pipeline.py       # stage orchestration + funnel accounting
  report.py         # matplotlib charts, DuckDB summaries, HTML/Markdown report
  classifiers.py    # optional toxicity-classifier hooks
  stages/           # normalize, dedupe, quality, mitigation
configs/            # sample.yaml (offline), dolly15k.yaml (full)
lexicons/           # toxicity lexicon (extend or replace)
data/sample/        # 1,500-record Dolly subset + 33 labelled synthetic noise records
scripts/            # synthetic noise generator
tests/              # unit tests per stage + end-to-end + CLI + packaging
```

## Known limitations & next steps

- **Language ID on short or name-heavy prompts is noisy.** A handful of English prompts
  full of foreign proper nouns ("Who was Octávio Tarquínio de Sousa?") get flagged.
  Next step: fastText `lid.176` with per-language confidence thresholds.
- **The lexicon toxicity filter is deliberately crude.** On Dolly it flags one
  creative-writing prompt ("the cat should be hostile toward the dog"). The classifier hook
  is there for a real model such as Detoxify, or a small fine-tuned classifier.
- **Regex PII has gaps.** It doesn't detect names or street addresses. Next step: a
  Presidio/NER pass behind the same interface.
- **Scale.** Stages run in-process on Polars, which handles a few million records on a laptop.
  For billions: shard the input, run MinHash per shard, and do a global LSH join (e.g. Ray or
  Spark). The stage interfaces were kept pure-function to make that straightforward.
- **More quality signals:** perplexity filtering with a small LM, an instruction/response
  relevance score, and a decontamination check against eval benchmarks (n-gram overlap).


## Install

From a clone (development):

```bash
uv sync
uv run curate --version
uv run curate run --config configs/sample.yaml
```

From a built wheel:

```bash
uv build                                 # writes dist/*.whl and dist/*.tar.gz
uv pip install dist/llm_dataset_curation-*.whl
curate --version
```

GitHub Releases attach the same artifacts when a `v*` tag is pushed (see below).
PyPI publish is optional and not wired up yet — drop a token into the release
workflow later if you want `twine upload`.

## Cutting a release

Version is single-sourced in `src/curate/__init__.py` (`__version__`); hatchling
reads it for the wheel/sdist metadata.

1. Bump `__version__` (e.g. to `0.1.1`).
2. Commit on `main`, then tag and push:
   ```bash
   git tag v0.1.1
   git push origin main --tags
   ```
3. The **release** workflow (`.github/workflows/release.yml`) runs lint + tests,
   builds the sdist and wheel with `uv build`, uploads them as Actions artifacts,
   and attaches them to a GitHub Release for the tag.
4. You can also trigger a build-only run via **Actions → release → Run workflow**
   (no tag / no GitHub Release).

## Data & license

- [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
  by Databricks, licensed **CC BY-SA 3.0**. The in-repo sample and the example report
  contain excerpts of it.
- Synthetic noise records are generated by `scripts/make_synthetic_noise.py`. All "PII" in
  them is fake (example.com addresses, 555 numbers, documentation IP ranges, test card numbers).
- Code: MIT.
