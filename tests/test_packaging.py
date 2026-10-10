"""Packaging, version, and CLI entry-point checks."""

from __future__ import annotations

import re
import tomllib
from importlib.metadata import PackageNotFoundError, entry_points, metadata, version
from pathlib import Path

import pytest

from curate import __version__
from curate.cli import main, package_version

ROOT = Path(__file__).resolve().parents[1]


def test_version_constant_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__), __version__


def test_package_version_matches_constant():
    assert package_version() == __version__


def test_cli_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out.strip()
    assert out == f"curate {__version__}"


def test_installed_distribution_metadata():
    try:
        meta = metadata("llm-dataset-curation")
    except PackageNotFoundError:
        pytest.skip("package not installed in this environment")
    assert meta["Name"] == "llm-dataset-curation"
    assert version("llm-dataset-curation") == __version__
    assert meta["License"] == "MIT"
    project_urls = meta.get_all("Project-URL") or []
    assert any(u.startswith("Homepage,") for u in project_urls)
    assert any(u.startswith("Repository,") for u in project_urls)
    assert any(u.startswith("Issues,") for u in project_urls)
    scripts = entry_points(group="console_scripts")
    assert any(ep.name == "curate" and ep.value == "curate.cli:main" for ep in scripts)


def test_pyproject_publish_metadata():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    assert project["name"] == "llm-dataset-curation"
    assert project["dynamic"] == ["version"]
    assert "authors" in project and project["authors"][0]["name"] == "John Houghton"
    assert "license" in project
    assert "keywords" in project and "llm" in project["keywords"]
    assert "classifiers" in project
    urls = project["urls"]
    assert urls["Homepage"].endswith("/llm-dataset-curation")
    assert urls["Repository"].endswith("/llm-dataset-curation")
    assert urls["Issues"].endswith("/issues")
    assert project["scripts"]["curate"] == "curate.cli:main"
    hatch = data["tool"]["hatch"]
    assert hatch["version"]["path"] == "src/curate/__init__.py"
    assert "src/curate" in hatch["build"]["targets"]["wheel"]["packages"]


def test_hatch_version_matches_init():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    path = ROOT / data["tool"]["hatch"]["version"]["path"]
    text = path.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.M)
    assert match, "expected __version__ assignment in version path"
    assert match.group(1) == __version__
