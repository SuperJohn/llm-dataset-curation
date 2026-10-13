import polars as pl

from curate.config import LanguageConfig, QualityConfig
from curate.stages import quality
from curate.stages.quality import detect_language, top_ngram_char_fraction

GOOD = (
    "Photosynthesis is the process plants use to turn sunlight, water and carbon dioxide "
    "into glucose and oxygen. It happens mostly in the leaves, inside chloroplasts."
)


def _run(texts, **overrides):
    cfg = QualityConfig(**overrides)
    df = pl.DataFrame({"id": [f"r{i}" for i in range(len(texts))], "text": texts})
    return quality.run(df, cfg)


def test_good_text_passes():
    res = _run([GOOD])
    assert res.kept.height == 1 and res.removed.height == 0


def test_rules_fire_with_expected_codes():
    texts = [
        "ok",  # too_short
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit sed do.",  # boilerplate
        "#### ### ## # ok words here now",  # symbol ratio
        "\n".join(["same line here"] * 6),  # duplicate lines
        "buy now " * 40,  # repetitive n-grams
        "La capital de Francia es París, una ciudad famosa por su arte y su comida.",  # language
    ]
    res = _run(texts)
    assert res.kept.height == 0
    assert set(res.stats["by_rule"]) == {
        "too_short",
        "boilerplate",
        "symbol_ratio",
        "duplicate_lines",
        "repetitive_ngrams",
        "language",
    }


def test_language_can_be_disabled():
    res = _run(
        ["La capital de Francia es París, una ciudad famosa por su arte y su comida."],
        language=LanguageConfig(enabled=False),
    )
    assert res.kept.height == 1


def test_signals_are_attached_to_kept_rows():
    res = _run([GOOD])
    assert {"q_n_words", "q_lang", "q_alpha_word_frac"} <= set(res.kept.columns)


def test_detect_language():
    lang, conf = detect_language(GOOD)
    assert lang == "en" and conf > 0.9
    lang, conf = detect_language("Die Hauptstadt von Deutschland ist Berlin und sehr schön.")
    assert lang == "de" and conf > 0.9


def test_signals_record_fasttext_language_and_confidence():
    cfg = QualityConfig()
    sig = quality.signals(GOOD, cfg)
    assert sig["lang"] == "en" and 0.9 < sig["lang_conf"] <= 1.0


def test_top_ngram_fraction():
    assert top_ngram_char_fraction(["a", "b", "c", "d"], 2) == 0.0
    assert top_ngram_char_fraction(("spam eggs " * 10).split(), 2) > 0.9
