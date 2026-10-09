from __future__ import annotations

from pathlib import Path

import pytest

from curate.config import PipelineConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def sample_config(tmp_path: Path) -> PipelineConfig:
    cfg = load_config(ROOT / "configs" / "sample.yaml", {"output": {"dir": str(tmp_path / "out")}})
    cfg.base_dir = ROOT
    return cfg
