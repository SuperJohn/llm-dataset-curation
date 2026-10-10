"""`curate` command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import requests
import yaml

from curate import __version__ as _fallback_version
from curate.config import load_config

DATASETS = {
    # CC BY-SA 3.0 — https://huggingface.co/datasets/databricks/databricks-dolly-15k
    "dolly-15k": "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl",
}


def package_version() -> str:
    """Installed distribution version, falling back to the in-tree constant."""
    try:
        return version("llm-dataset-curation")
    except PackageNotFoundError:
        return _fallback_version


def _cmd_run(args: argparse.Namespace) -> int:
    from curate.pipeline import run_pipeline
    from curate.report import write_report

    overrides = {}
    for kv in args.set or []:
        key, _, raw = kv.partition("=")
        node = overrides
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    cfg = load_config(args.config, overrides)
    run = run_pipeline(cfg)
    paths = {} if args.no_report else write_report(run)
    print(f"\n{cfg.name}: {run.input_df.height:,} -> {run.curated.height:,} records")
    for f in run.funnel[1:]:
        print(f"  {f['stage']:<13} -{f['removed']:<6} -> {f['rows_out']:,}")
    print(f"outputs: {run.out_dir}")
    for k, p in paths.items():
        print(f"  {k} report: {p}")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    url = DATASETS[args.dataset]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with out.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    # dolly has no id column; add a stable one
    if args.dataset == "dolly-15k":
        lines = out.read_text(encoding="utf-8").splitlines()
        rows = []
        for i, line in enumerate(lines):
            rec = json.loads(line)
            rows.append(json.dumps({"id": f"dolly-{i:05d}", **rec}, ensure_ascii=False))
        out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"downloaded {args.dataset} -> {out}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(cfg.model_dump_json(indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="curate", description="LLM dataset curation pipeline")
    p.add_argument("--version", action="version", version=f"curate {package_version()}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the pipeline and write outputs + report")
    r.add_argument("--config", "-c", required=True)
    r.add_argument(
        "--set",
        action="append",
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set stages.near_dedupe.threshold=0.7",
    )
    r.add_argument("--no-report", action="store_true")
    r.set_defaults(func=_cmd_run)

    d = sub.add_parser("download", help="download a public dataset")
    d.add_argument("--dataset", choices=sorted(DATASETS), default="dolly-15k")
    d.add_argument("--out", default="data/raw/dolly-15k.jsonl")
    d.set_defaults(func=_cmd_download)

    v = sub.add_parser("validate", help="validate and print a config")
    v.add_argument("--config", "-c", required=True)
    v.set_defaults(func=_cmd_validate)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
