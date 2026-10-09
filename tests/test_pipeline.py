import json

import polars as pl

from curate.cli import main
from curate.eval import NOISE_TO_STAGE
from curate.pipeline import run_pipeline
from curate.report import write_report


def test_sample_end_to_end(sample_config):
    run = run_pipeline(sample_config)
    out = run.out_dir
    assert (out / "curated.parquet").exists() and (out / "removed.parquet").exists()
    stats = json.loads((out / "stats.json").read_text())
    n_synthetic = pl.read_ndjson(sample_config.resolve("data/sample/synthetic_noise.jsonl")).height
    assert stats["input_rows"] == 1500 + n_synthetic
    assert stats["output_rows"] == run.curated.height
    # funnel is internally consistent
    for f in stats["funnel"][1:]:
        assert f["rows_in"] - f["removed"] == f["rows_out"]
    # ids are unique in the output and never appear in both kept and removed
    assert run.curated["id"].is_unique().all()
    assert not set(run.curated["id"]) & set(run.removed["id"])
    # eval block present with per-stage metrics
    assert "eval" in stats and "per_stage" in stats["eval"]
    for stage in ("exact_dedupe", "near_dedupe", "quality", "toxicity", "pii"):
        assert stage in stats["eval"]["per_stage"]
        m = stats["eval"]["per_stage"][stage]
        assert {"tp", "fp", "fn", "precision", "recall", "support"} <= set(m)


def test_every_synthetic_noise_record_is_handled(sample_config):
    run = run_pipeline(sample_config, write=False)
    removed = dict(zip(run.removed["id"], run.removed["_stage"], strict=True))
    curated = run.curated.filter(pl.col("id").str.starts_with("synthetic"))
    noise = pl.read_ndjson(sample_config.resolve("data/sample/synthetic_noise.jsonl"))
    for rec in noise.iter_rows(named=True):
        nt = rec["noise_type"]
        if nt == "pii":
            row = curated.filter(pl.col("id") == rec["id"]).row(0, named=True)
            assert row["pii_hits"] > 0
            assert "@example" not in row["text"] and "555" not in row["text"]
        elif nt == "clean":
            assert rec["id"] not in removed, f"clean control removed by {removed.get(rec['id'])}"
            row = curated.filter(pl.col("id") == rec["id"]).row(0, named=True)
            assert row.get("pii_hits", 0) == 0
        else:
            assert removed.get(rec["id"]) == NOISE_TO_STAGE[nt], rec["id"]


def test_report_written(sample_config):
    run = run_pipeline(sample_config)
    paths = write_report(run)
    html = paths["html"].read_text()
    assert "Curation report" in html and "data:image/png;base64" in html
    assert "Precision / Recall" in html
    md = paths["markdown"].read_text()
    assert "## Funnel" in md and (run.out_dir / "report_assets" / "funnel.png").exists()
    assert "## Precision / Recall" in md
    assert "| stage | kind | precision | recall |" in md


def test_cli_run_with_override(tmp_path, monkeypatch, sample_config):
    monkeypatch.chdir(sample_config.base_dir)
    rc = main(
        [
            "run",
            "-c",
            "configs/sample.yaml",
            "--set",
            f"output.dir={tmp_path / 'cli'}",
            "--set",
            "stages.near_dedupe.enabled=false",
        ]
    )
    assert rc == 0
    stats = json.loads((tmp_path / "cli" / "stats.json").read_text())
    assert "near_dedupe" not in [f["stage"] for f in stats["funnel"]]
