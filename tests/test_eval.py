"""Unit tests for per-stage precision/recall against labelled synthetic noise."""

from __future__ import annotations

import polars as pl

from curate.eval import NOISE_TO_STAGE, per_stage_metrics


def _frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_perfect_removal_stage_scores():
    input_df = _frame(
        [
            {"id": "s1", "noise_type": "toxic"},
            {"id": "s2", "noise_type": "toxic"},
            {"id": "d1", "noise_type": None},
            {"id": "c1", "noise_type": "clean"},
        ]
    )
    removed = _frame(
        [
            {"id": "s1", "_stage": "toxicity"},
            {"id": "s2", "_stage": "toxicity"},
        ]
    )
    curated = _frame(
        [
            {"id": "d1", "noise_type": None, "pii_hits": 0},
            {"id": "c1", "noise_type": "clean", "pii_hits": 0},
        ]
    )
    m = per_stage_metrics(input_df, removed, curated)["per_stage"]["toxicity"]
    assert m == {
        "tp": 2,
        "fp": 0,
        "fn": 0,
        "support": 2,
        "precision": 1.0,
        "recall": 1.0,
        "kind": "removal",
        "fp_scope": "unlabeled_or_clean_or_wrong_stage",
    }


def test_false_positives_include_dolly_and_wrong_stage():
    input_df = _frame(
        [
            {"id": "s1", "noise_type": "toxic"},
            {"id": "s2", "noise_type": "too_short"},  # expected quality
            {"id": "d1", "noise_type": None},
            {"id": "c1", "noise_type": "clean"},
        ]
    )
    # toxicity removes true toxic + Dolly + clean + quality-labelled synthetic
    removed = _frame(
        [
            {"id": "s1", "_stage": "toxicity"},
            {"id": "s2", "_stage": "toxicity"},
            {"id": "d1", "_stage": "toxicity"},
            {"id": "c1", "_stage": "toxicity"},
        ]
    )
    curated = _frame([])
    m = per_stage_metrics(input_df, removed, curated)["per_stage"]["toxicity"]
    assert m["tp"] == 1
    assert m["fn"] == 0
    assert m["fp"] == 3  # s2 wrong-stage + d1 + c1
    assert m["precision"] == round(1 / 4, 4)
    assert m["recall"] == 1.0


def test_false_negatives_when_labelled_row_kept_or_wrong_stage():
    input_df = _frame(
        [
            {"id": "s1", "noise_type": "toxic"},
            {"id": "s2", "noise_type": "toxic"},
        ]
    )
    removed = _frame([{"id": "s2", "_stage": "quality"}])  # wrong stage
    curated = _frame([{"id": "s1", "noise_type": "toxic", "pii_hits": 0}])
    m = per_stage_metrics(input_df, removed, curated)["per_stage"]["toxicity"]
    assert m["tp"] == 0
    assert m["fn"] == 2
    assert m["recall"] == 0.0


def test_pii_detection_scores_clean_fp_only():
    input_df = _frame(
        [
            {"id": "p1", "noise_type": "pii"},
            {"id": "p2", "noise_type": "pii"},
            {"id": "c1", "noise_type": "clean"},
            {"id": "d1", "noise_type": None},  # Dolly with real PII — excluded from FP
        ]
    )
    removed = _frame([])
    curated = _frame(
        [
            {"id": "p1", "noise_type": "pii", "pii_hits": 2},
            {"id": "p2", "noise_type": "pii", "pii_hits": 0},  # FN
            {"id": "c1", "noise_type": "clean", "pii_hits": 1},  # FP
            {"id": "d1", "noise_type": None, "pii_hits": 3},  # not counted as FP
        ]
    )
    m = per_stage_metrics(input_df, removed, curated)["per_stage"]["pii"]
    assert m["tp"] == 1
    assert m["fn"] == 1
    assert m["fp"] == 1
    assert m["kind"] == "detection"
    assert m["fp_scope"] == "clean_labelled_only"
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5


def test_noise_to_stage_covers_known_bad_types():
    assert set(NOISE_TO_STAGE) >= {
        "exact_dup_after_normalize",
        "near_dup",
        "non_english",
        "repetitive",
        "too_short",
        "boilerplate",
        "symbol_noise",
        "toxic",
    }
    assert "pii" not in NOISE_TO_STAGE
    assert "clean" not in NOISE_TO_STAGE
