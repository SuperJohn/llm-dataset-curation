"""Orchestrates the stages and collects everything the report needs."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from curate.config import PipelineConfig
from curate.io import read_input, write_outputs
from curate.stages import StageResult, dedupe, mitigation, normalize, quality

log = logging.getLogger(__name__)

# PII redaction runs immediately after normalization so that no downstream artifact
# (dedupe clusters, removed samples, the report) ever contains raw PII.
STAGE_ORDER = ["normalize", "pii", "exact_dedupe", "near_dedupe", "quality", "toxicity"]


@dataclass
class RunResult:
    config: PipelineConfig
    input_df: pl.DataFrame
    curated: pl.DataFrame
    removed: pl.DataFrame
    stages: list[StageResult]
    funnel: list[dict] = field(default_factory=list)
    clusters: pl.DataFrame | None = None
    out_dir: Path | None = None


def _enabled(cfg: PipelineConfig, name: str) -> bool:
    s = cfg.stages
    return {
        "normalize": s.normalize.enabled,
        "pii": s.mitigation.pii.enabled,
        "exact_dedupe": s.exact_dedupe.enabled,
        "near_dedupe": s.near_dedupe.enabled,
        "quality": s.quality.enabled,
        "toxicity": s.mitigation.toxicity.enabled,
    }[name]


def _run_stage(name: str, df: pl.DataFrame, cfg: PipelineConfig) -> StageResult:
    s, fields = cfg.stages, cfg.input.text_fields
    if name == "normalize":
        return normalize.run(df, s.normalize, fields)
    if name == "pii":
        return mitigation.run_pii(df, s.mitigation.pii, fields)
    if name == "exact_dedupe":
        return dedupe.exact_dedupe(df, s.exact_dedupe)
    if name == "near_dedupe":
        return dedupe.near_dedupe(df, s.near_dedupe)
    if name == "quality":
        return quality.run(df, s.quality, fields)
    if name == "toxicity":
        return mitigation.run_toxicity(df, s.mitigation.toxicity, cfg.base_dir)
    raise KeyError(name)


def run_pipeline(cfg: PipelineConfig, write: bool = True) -> RunResult:
    df = read_input(cfg)
    input_df = df
    log.info("loaded %s records from %s file(s)", df.height, len(cfg.input.paths))
    results: list[StageResult] = []
    funnel = [{"stage": "input", "rows_in": df.height, "removed": 0, "rows_out": df.height}]
    for name in STAGE_ORDER:
        if not _enabled(cfg, name):
            continue
        t0 = time.perf_counter()
        res = _run_stage(name, df, cfg)
        res.stats["seconds"] = round(time.perf_counter() - t0, 3)
        funnel.append(
            {
                "stage": name,
                "rows_in": df.height,
                "removed": res.removed.height,
                "rows_out": res.kept.height,
            }
        )
        log.info("%-13s %6d -> %6d  (-%d)", name, df.height, res.kept.height, res.removed.height)
        results.append(res)
        df = res.kept

    removed = pl.concat([r.removed for r in results if r.removed.height], how="diagonal_relaxed")
    if removed.is_empty():
        removed = df.clear().with_columns(pl.lit("").alias("_stage"), pl.lit("").alias("_reason"))
    clusters = next((r.stats.pop("_clusters") for r in results if "_clusters" in r.stats), None)
    run = RunResult(cfg, input_df, df, removed, results, funnel, clusters)

    if write:
        out_dir = cfg.resolve(cfg.output.dir)
        write_outputs(df, out_dir, "curated", cfg.output.write_jsonl)
        removed.write_parquet(out_dir / "removed.parquet")
        if clusters is not None:
            clusters.write_parquet(out_dir / "dup_clusters.parquet")
        stats = {
            "name": cfg.name,
            "funnel": funnel,
            "stages": {r.name: r.stats for r in results},
            "input_rows": input_df.height,
            "output_rows": df.height,
            "retention": round(df.height / max(input_df.height, 1), 4),
        }
        (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
        (out_dir / "config.resolved.json").write_text(cfg.model_dump_json(indent=2))
        run.out_dir = out_dir
    return run
