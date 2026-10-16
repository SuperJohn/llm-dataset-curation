"""MinHash vs embedding near-dedupe on a labelled perturbation benchmark (issue #4).

Builds the benchmark from the checked-in Dolly sample (or any JSONL with instruction/response),
runs each method as its own stage over the same rows, and prints a markdown table plus JSON.

    uv run python scripts/compare_dedupe.py                         # offline: MinHash + hashing
    uv run python scripts/compare_dedupe.py --sentence-transformers # also MiniLM (semantic extra)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from curate.config import NearDedupeConfig, SemanticDedupeConfig
from curate.dedupe_compare import (
    PERTURBATIONS,
    false_positive_breakdown,
    make_benchmark,
    markdown_table,
    score_method,
)
from curate.stages.dedupe import near_dedupe, semantic_dedupe

ROOT = Path(__file__).resolve().parents[1]
ST = "curate.embeddings:sentence_transformer_embeddings"
HASH = "curate.embeddings:hashing_embeddings"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", default=str(ROOT / "data/sample/dolly_sample.jsonl"))
    p.add_argument("--per-kind", type=int, default=50)
    p.add_argument("--hard-negatives", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--sentence-transformers", action="store_true")
    p.add_argument("--json", help="also write scores to this JSON file")
    args = p.parse_args()

    records = pl.read_ndjson(args.input)
    bench = make_benchmark(
        records, per_kind=args.per_kind, hard_negatives=args.hard_negatives, seed=args.seed
    )
    fields = ["instruction", "response"]
    methods = {
        "MinHash J>=0.8 (pipeline default)": lambda df: near_dedupe(
            df, NearDedupeConfig(fields=fields, threshold=0.8, min_words=20)
        ),
        "MinHash J>=0.5": lambda df: near_dedupe(
            df, NearDedupeConfig(fields=fields, threshold=0.5, min_words=20)
        ),
        "hashing embedder cos>=0.9": lambda df: semantic_dedupe(
            df, SemanticDedupeConfig(enabled=True, fields=fields, embedder=HASH, threshold=0.9)
        ),
    }
    if args.sentence_transformers:
        for t in (0.95, 0.9):
            methods[f"MiniLM-L6 cos>={t}"] = lambda df, t=t: semantic_dedupe(
                df,
                SemanticDedupeConfig(
                    enabled=True, fields=fields, embedder=ST, threshold=t, index="faiss"
                ),
            )

    scores, breakdown = [], {}
    for name, run in methods.items():
        scores.append(score_method(bench, name, run))
        breakdown[name] = false_positive_breakdown(bench, run(bench))

    kinds = bench["kind"].value_counts().sort("kind")
    print(f"benchmark: {bench.height} rows from {args.input}")
    print(dict(zip(kinds["kind"].to_list(), kinds["count"].to_list(), strict=True)))
    print()
    print(markdown_table(scores))
    print()
    for name, fp in breakdown.items():
        print(f"false positives by row kind, {name}: {fp or 'none'}")
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "perturbations": PERTURBATIONS,
                    "rows": bench.height,
                    "scores": [s.as_dict() for s in scores],
                    "false_positives_by_kind": breakdown,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
