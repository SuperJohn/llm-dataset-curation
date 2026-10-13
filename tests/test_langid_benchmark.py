"""Issue #2 benchmark: English prompts dense with foreign names must not be flagged.

Uses the labelled prompts in data/benchmarks/langid_prompts.jsonl, scored the way the quality
stage scores them (default candidate list; flagged = non-English at >= threshold).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest

from curate.config import LanguageConfig, QualityConfig, load_config
from curate.langid import detect_language
from curate.stages import quality

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_langid as bench  # noqa: E402

CANDIDATES = LanguageConfig().candidates
ROWS = bench.load_prompts()


@pytest.mark.parametrize(
    "prompt",
    [
        "Who was Octávio Tarquínio de Sousa?",
        "Where is Château de Chenonceau located?",
        "Which club does Cristiano Ronaldo dos Santos Aveiro play for?",
        "Who was Hô Chí Minh?",
        "Who was Juscelino Kubitschek de Oliveira?",
        "What is the plot of Les Misérables by Victor Hugo?",
    ],
)
def test_name_heavy_english_prompts_detect_as_english(prompt):
    lang, conf = detect_language(prompt, CANDIDATES)
    assert lang == "en" and conf >= 0.6, (lang, conf)


@pytest.mark.parametrize(
    "prompt",
    [
        "Explain the career of Sor Juana Inés de la Cruz.",
        "Tell me about Jean-Paul Sartre and Simone de Beauvoir in Paris.",
        "Who is Gabriel García Márquez?",
        "Name three dishes from Emilia-Romagna: tagliatelle al ragù, tortellini in brodo, piadina.",
    ],
)
def test_ambiguous_english_prompts_stay_below_the_language_gate(prompt):
    # fastText may still lean towards the names' language, but with low confidence.
    lang, conf = detect_language(prompt, CANDIDATES)
    assert not bench.flagged(lang, conf, threshold=0.5), (lang, conf)


@pytest.mark.parametrize("row", [r for r in ROWS if r["lang"] != "en"], ids=lambda r: r["lang"])
def test_non_english_prompts_detected(row):
    lang, conf = detect_language(row["text"], CANDIDATES)
    assert lang == row["lang"] and conf >= 0.8, (lang, conf)


def test_benchmark_false_flags_and_recall():
    res = bench.score(ROWS, detect_language, threshold=0.5, candidates=CANDIDATES)
    # One known miss: "How do I make good pour over coffee?" -> fr 0.97 ('pour').
    assert res["false_flag_texts"] == ["How do I make good pour over coffee?"]
    assert res["non_english_recall"] == 1.0


def test_fasttext_beats_py3langid_on_benchmark():
    if importlib.util.find_spec("py3langid") is None:
        pytest.skip("py3langid (dev-only, previous detector) not installed")
    for threshold in (0.5, 0.9):
        new = bench.score(ROWS, detect_language, threshold, CANDIDATES)
        old = bench.score(ROWS, bench.py3langid_detect, threshold, CANDIDATES)
        assert new["english_false_flags"] * 4 <= old["english_false_flags"], (threshold, new, old)


def test_quality_rule_keeps_names_and_drops_foreign_prompts():
    """End to end through the quality stage with the sample config's language settings."""
    cfg = load_config(ROOT / "configs" / "sample.yaml").stages.quality
    cfg = QualityConfig(**{**cfg.model_dump(), "required_fields": [], "min_words": 1})
    prompts = {
        "en-names": "Tell me about Jean-Paul Sartre and Simone de Beauvoir in Paris.",
        "en-dolly": "who is a feminist writer : simone de beauvoir, simone veil, virginia woolf, "
        "Gisèle halimi, Voltaire, Marquis de Sade",
        "es": "¿Cómo se prepara una paella valenciana tradicional paso a paso?",
        "de": "Warum ist die Berliner Mauer historisch so bedeutend für Europa?",
    }
    df = pl.DataFrame(
        {
            "id": list(prompts),
            "instruction": list(prompts.values()),
            "response": ["A short answer in the same language as the question."] * len(prompts),
        }
    ).with_columns(text=pl.col("instruction") + "\n" + pl.col("response"))
    res = quality.run(df, cfg)
    assert set(res.kept["id"]) == {"en-names", "en-dolly"}
    removed = dict(zip(res.removed["id"], res.removed["_reason"], strict=True))
    assert set(removed) == {"es", "de"}
    assert all(
        r.startswith("language 'es'") or r.startswith("language 'de'") for r in removed.values()
    )
