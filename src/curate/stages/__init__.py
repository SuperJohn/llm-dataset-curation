"""Pipeline stages. Each stage returns a StageResult (kept, removed, stats)."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl


@dataclass
class StageResult:
    name: str
    kept: pl.DataFrame
    removed: pl.DataFrame  # same schema as input + `_stage`, `_reason`
    stats: dict = field(default_factory=dict)


def removed_frame(df: pl.DataFrame, stage: str, reason: pl.Expr | pl.Series | str) -> pl.DataFrame:
    if isinstance(reason, str):
        reason = pl.lit(reason, dtype=pl.Utf8)
    elif isinstance(reason, pl.Series):
        reason = pl.lit(reason)
    return df.with_columns(pl.lit(stage).alias("_stage"), reason.alias("_reason"))
