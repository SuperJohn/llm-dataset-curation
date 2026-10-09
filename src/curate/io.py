"""Load heterogeneous inputs into one Polars frame with `id` + `text` columns."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from curate.config import PipelineConfig


def read_input(cfg: PipelineConfig) -> pl.DataFrame:
    frames = []
    for p in cfg.input.paths:
        path = cfg.resolve(p)
        if cfg.input.format == "jsonl":
            df = pl.read_ndjson(path, infer_schema_length=None)
        elif cfg.input.format == "parquet":
            df = pl.read_parquet(path)
        else:
            df = pl.read_csv(path)
        frames.append(df.with_columns(pl.lit(Path(p).name).alias("_source")))
    df = pl.concat(frames, how="diagonal_relaxed")
    if cfg.input.limit:
        df = df.head(cfg.input.limit)

    id_field = cfg.input.id_field
    if id_field not in df.columns:
        df = df.with_row_index("id").with_columns(pl.col("id").cast(pl.Utf8))
    elif id_field != "id":
        df = df.rename({id_field: "id"})
    df = df.with_columns(pl.col("id").cast(pl.Utf8))

    missing = [f for f in cfg.input.text_fields if f not in df.columns]
    if missing:
        raise ValueError(f"text_fields not found in input: {missing}")
    df = df.with_columns([pl.col(f).cast(pl.Utf8).fill_null("") for f in cfg.input.text_fields])
    return with_combined_text(df, cfg.input.text_fields)


def with_combined_text(df: pl.DataFrame, fields: list[str]) -> pl.DataFrame:
    """`text` = non-empty text fields joined by a blank line (what a model would actually see)."""
    parts = [pl.when(pl.col(f).str.strip_chars() != "").then(pl.col(f)) for f in fields]
    return df.with_columns(pl.concat_list(parts).list.drop_nulls().list.join("\n\n").alias("text"))


def write_outputs(df: pl.DataFrame, out_dir: Path, name: str, jsonl: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / f"{name}.parquet")
    if jsonl:
        df.write_ndjson(out_dir / f"{name}.jsonl")
