"""Walkthrough notebook validity + the same offline path the notebook runs."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import polars as pl
import pytest
from nbformat.validator import NotebookValidationError

from curate.config import load_config
from curate.pipeline import run_pipeline
from curate.report import write_report

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "walkthrough.ipynb"


def test_walkthrough_notebook_exists_and_validates():
    assert NOTEBOOK.is_file(), f"missing {NOTEBOOK}"
    nb = nbformat.read(NOTEBOOK, as_version=4)
    try:
        nbformat.validate(nb)
    except NotebookValidationError as exc:
        pytest.fail(f"notebook failed nbformat validation: {exc}")
    assert len(nb.cells) >= 8
    sources = [c.source for c in nb.cells]
    joined = "\n".join(sources)
    assert "run_pipeline" in joined
    assert "write_report" in joined
    assert "configs/sample.yaml" in joined
    assert "synthetic-022" in joined
    assert any(c.cell_type == "markdown" for c in nb.cells)
    assert any(c.cell_type == "code" for c in nb.cells)


def test_walkthrough_path_produces_artifacts(tmp_path: Path):
    """Mirror the notebook's happy path without requiring Jupyter execution."""
    out = tmp_path / "notebook-walkthrough"
    cfg = load_config(ROOT / "configs" / "sample.yaml", {"output": {"dir": str(out)}})
    cfg.base_dir = ROOT

    run = run_pipeline(cfg, write=True)
    paths = write_report(run)

    assert run.out_dir == out
    assert (out / "curated.parquet").exists()
    assert (out / "removed.parquet").exists()
    assert (out / "stats.json").exists()
    assert paths["html"].exists() and paths["markdown"].exists()

    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert stats["input_rows"] == run.input_df.height
    assert stats["output_rows"] == run.curated.height
    assert stats["output_rows"] < stats["input_rows"]
    assert stats["funnel"][0]["stage"] == "input"
    assert any(f["stage"] == "near_dedupe" for f in stats["funnel"])

    curated = pl.read_parquet(out / "curated.parquet")
    removed = pl.read_parquet(out / "removed.parquet")
    assert curated.height == run.curated.height
    assert removed.height == run.removed.height
    assert {"_stage", "_reason"} <= set(removed.columns)

    # PII before/after on the labelled synthetic row the notebook highlights
    pii_id = "synthetic-022"
    before = run.input_df.filter(pl.col("id") == pii_id).row(0, named=True)
    after = curated.filter(pl.col("id") == pii_id).row(0, named=True)
    assert "@example.com" in before["response"]
    assert "@example.com" not in after["response"]
    assert "<EMAIL>" in after["response"] and "<PHONE>" in after["response"]
    assert after["pii_hits"] > 0

    near = removed.filter(pl.col("_stage") == "near_dedupe")
    assert near.height > 0
    reason = near.head(1).row(0, named=True)["_reason"]
    assert "near duplicate of" in reason

    md = paths["markdown"].read_text(encoding="utf-8")
    assert "## Funnel" in md
