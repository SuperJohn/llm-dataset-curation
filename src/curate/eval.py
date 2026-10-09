"""Per-stage precision / recall against labelled synthetic noise.

Ground truth comes from `noise_type` on synthetic rows (see scripts/make_synthetic_noise.py).
Dolly / unlabeled rows have null `noise_type`.

Definitions for each *removal* stage S in {exact_dedupe, near_dedupe, quality, toxicity}:

  TP  synthetic rows whose noise_type maps to S and that were removed by S
  FN  synthetic rows whose noise_type maps to S but were *not* removed by S
      (kept through the whole pipeline, or removed by a different stage)
  FP  rows removed by S that were *not* labelled for S:
        - unlabeled / Dolly rows (null noise_type)
        - clean-labelled controls
        - synthetic rows labelled for a different stage (wrong-stage removals)

  precision = TP / (TP + FP)   (None when TP+FP == 0)
  recall    = TP / (TP + FN)   (None when TP+FN == 0)
  support   = TP + FN          (labelled positives for S)

PII is a *detection* stage (redact-and-keep), so it is scored separately:

  TP  labelled pii synthetics that survive with pii_hits > 0
  FN  labelled pii synthetics that were removed entirely, or kept with pii_hits == 0
  FP  clean-labelled synthetics with pii_hits > 0
      (Dolly rows with genuine PII are excluded from FP so the metric stays
      synthetic-grounded; see `fp_scope` in the returned dict.)
"""

from __future__ import annotations

from typing import Any

import polars as pl

# noise_type -> expected removal stage (pii is detection-only; clean expects survival)
NOISE_TO_STAGE: dict[str, str] = {
    "exact_dup_after_normalize": "exact_dedupe",
    "near_dup": "near_dedupe",
    "non_english": "quality",
    "repetitive": "quality",
    "too_short": "quality",
    "boilerplate": "quality",
    "symbol_noise": "quality",
    "toxic": "toxicity",
}

REMOVAL_STAGES = ("exact_dedupe", "near_dedupe", "quality", "toxicity")
CLEAN_LABEL = "clean"
PII_LABEL = "pii"


def _rate(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return round(num / den, 4)


def _stage_counts(tp: int, fp: int, fn: int) -> dict[str, Any]:
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
        "precision": _rate(tp, tp + fp),
        "recall": _rate(tp, tp + fn),
    }


def per_stage_metrics(
    input_df: pl.DataFrame,
    removed: pl.DataFrame,
    curated: pl.DataFrame,
) -> dict[str, Any]:
    """Compute per-stage precision/recall for the labelled synthetic set.

    Returns a dict suitable for `stats.json` under `eval`, including `per_stage`.
    """
    if "noise_type" not in input_df.columns:
        return {
            "note": "no noise_type column; synthetic eval skipped",
            "per_stage": {},
        }

    # Map id -> noise_type for every input row (null for Dolly).
    labels = {r["id"]: r["noise_type"] for r in input_df.select("id", "noise_type").to_dicts()}
    removed_stage: dict[str, str] = {}
    if not removed.is_empty() and "_stage" in removed.columns:
        removed_stage = dict(zip(removed["id"].to_list(), removed["_stage"].to_list(), strict=True))

    per_stage: dict[str, Any] = {}

    for stage in REMOVAL_STAGES:
        labelled_ids = {i for i, n in labels.items() if NOISE_TO_STAGE.get(n) == stage}
        removed_ids = {i for i, s in removed_stage.items() if s == stage}
        tp_ids = labelled_ids & removed_ids
        fn_ids = labelled_ids - removed_ids
        fp_ids = removed_ids - labelled_ids
        per_stage[stage] = {
            **_stage_counts(len(tp_ids), len(fp_ids), len(fn_ids)),
            "kind": "removal",
            # FP includes Dolly, clean controls, and wrong-stage synthetics.
            "fp_scope": "unlabeled_or_clean_or_wrong_stage",
        }

    # PII detection (redact-and-keep).
    pii_ids = {i for i, n in labels.items() if n == PII_LABEL}
    curated_pii_hits: dict[str, int] = {}
    if not curated.is_empty() and "pii_hits" in curated.columns:
        curated_pii_hits = {
            r["id"]: int(r["pii_hits"] or 0) for r in curated.select("id", "pii_hits").to_dicts()
        }
    tp = sum(1 for i in pii_ids if curated_pii_hits.get(i, 0) > 0)
    fn = len(pii_ids) - tp
    # Only count clean-labelled FPs so Dolly's real PII does not dominate.
    clean_ids = {i for i, n in labels.items() if n == CLEAN_LABEL}
    fp = sum(1 for i in clean_ids if curated_pii_hits.get(i, 0) > 0)
    per_stage["pii"] = {
        **_stage_counts(tp, fp, fn),
        "kind": "detection",
        "fp_scope": "clean_labelled_only",
    }

    # Clean-control survival summary (useful sanity check in the report).
    clean_removed = [i for i in clean_ids if i in removed_stage]
    n_clean = len(clean_ids)

    return {
        "noise_to_stage": dict(NOISE_TO_STAGE),
        "n_synthetic": sum(1 for n in labels.values() if n is not None),
        "n_labelled_bad": sum(1 for n in labels.values() if n is not None and n != CLEAN_LABEL),
        "n_clean_controls": n_clean,
        "clean_controls_removed": len(clean_removed),
        "clean_control_survival": _rate(n_clean - len(clean_removed), n_clean),
        "per_stage": per_stage,
    }
