SHELL := /bin/bash
CONFIG ?= configs/sample.yaml
RUN := uv run

.PHONY: help setup run full test lint format build report-example clean

help:  ## Show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-15s %s\n", $$1, $$2}'

setup:  ## Install deps with uv
	uv sync

run:  ## Curate the offline sample (or CONFIG=...)
	$(RUN) curate run --config $(CONFIG)

full:  ## Download databricks-dolly-15k and curate all of it
	$(RUN) curate download --dataset dolly-15k --out data/raw/dolly-15k.jsonl
	$(RUN) curate run --config configs/dolly15k.yaml

test:  ## Lint + tests
	$(RUN) ruff check .
	$(RUN) ruff format --check .
	$(RUN) pytest -q

lint:  ## Ruff lint
	$(RUN) ruff check .

format:  ## Ruff autoformat
	$(RUN) ruff format . && $(RUN) ruff check --fix .

build:  ## Build sdist + wheel into dist/
	uv build

report-example: full  ## Refresh docs/example-report from a full dolly-15k run
	rm -rf docs/example-report && mkdir -p docs/example-report
	cp -r output/dolly-15k/report.md output/dolly-15k/report.html output/dolly-15k/report_assets docs/example-report/

clean:  ## Remove outputs and downloads
	rm -rf output data/raw .pytest_cache dist build *.egg-info
