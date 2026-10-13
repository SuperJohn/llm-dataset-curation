"""Heuristic quality filters (Gopher / C4 style) + fastText lid.176 language ID.

Every row gets a set of quality signals; a row is dropped with the *first* failing rule as its
reason, and all signals are kept for the report so thresholds can be tuned from data.
"""

from __future__ import annotations

import re
from collections import Counter

import polars as pl

from curate.config import QualityConfig
from curate.langid import detect_language
from curate.stages import StageResult, removed_frame

_WORD = re.compile(r"\S+")
_ALPHA = re.compile(r"[^\W\d_]", re.UNICODE)
# Gopher counts hash symbols and ellipses (not list bullets) relative to word count.
_SYMBOL = re.compile(r"#|\.\.\.|\u2026")

__all__ = ["detect_language", "first_failure", "run", "signals", "top_ngram_char_fraction"]


def top_ngram_char_fraction(words: list[str], n: int) -> float:
    """Gopher: share of characters covered by the most frequent word n-gram."""
    if len(words) < n:
        return 0.0
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    gram, count = grams.most_common(1)[0]
    if count < 2:
        return 0.0
    total_chars = sum(len(w) for w in words)
    return (sum(len(w) for w in gram) * count) / max(total_chars, 1)


def _dup_line_frac(text: str) -> float:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return 1 - len(set(lines)) / len(lines) if len(lines) >= 3 else 0.0


def signals(
    text: str, cfg: QualityConfig, row: dict | None = None, fields: list[str] | None = None
) -> dict:
    row = row or {}
    fields = fields or []
    words = _WORD.findall(text)
    n = len(words)
    lower_words = [w.lower() for w in words]
    sig = {
        "n_words": n,
        "n_chars": len(text),
        "mean_word_len": (sum(len(w) for w in words) / n) if n else 0.0,
        "symbol_word_ratio": (len(_SYMBOL.findall(text)) / n) if n else 0.0,
        "alpha_word_frac": (sum(bool(_ALPHA.search(w)) for w in words) / n) if n else 0.0,
        # computed per field so a response that legitimately quotes its context isn't penalised
        "dup_line_frac": max(
            (_dup_line_frac(row[f]) for f in fields if isinstance(row.get(f), str)),
            default=_dup_line_frac(text),
        ),
        "top_ngram_frac": max(
            (top_ngram_char_fraction(lower_words, k) for k in cfg.max_top_ngram_char_fraction),
            default=0.0,
        )
        if n >= cfg.repetition_min_words
        else 0.0,
        "lang": None,
        "lang_conf": None,
    }
    if cfg.language.enabled:
        lang_text = (
            "\n".join(row.get(f) or "" for f in cfg.language.fields)
            if cfg.language.fields
            else text
        )
        if len(_WORD.findall(lang_text)) >= cfg.language.min_words:
            sig["lang"], sig["lang_conf"] = detect_language(
                lang_text.replace("\n", " "), cfg.language.candidates
            )
    return sig


def first_failure(row: dict, sig: dict, cfg: QualityConfig) -> tuple[str, str] | None:
    for f in cfg.required_fields:
        if not (row.get(f) or "").strip():
            return "required_field", f"empty required field '{f}'"
    if sig["n_words"] < cfg.min_words:
        return "too_short", f"too short ({sig['n_words']} words < {cfg.min_words})"
    if sig["n_words"] > cfg.max_words:
        return "too_long", f"too long ({sig['n_words']} words)"
    text_l = row["text"].lower()
    for phrase in cfg.banned_phrases:
        if phrase in text_l:
            return "boilerplate", f"boilerplate phrase '{phrase}'"
    if not cfg.min_mean_word_length <= sig["mean_word_len"] <= cfg.max_mean_word_length:
        return "mean_word_length", f"mean word length {sig['mean_word_len']:.1f} out of range"
    if sig["symbol_word_ratio"] > cfg.max_symbol_word_ratio:
        return "symbol_ratio", f"symbol/word ratio {sig['symbol_word_ratio']:.2f}"
    if sig["alpha_word_frac"] < cfg.min_alpha_word_fraction:
        return "alpha_fraction", f"alphabetic word fraction {sig['alpha_word_frac']:.2f}"
    if sig["dup_line_frac"] > cfg.max_duplicate_line_fraction:
        return "duplicate_lines", f"duplicate lines {sig['dup_line_frac']:.0%}"
    if sig["n_words"] >= cfg.repetition_min_words:
        words = [w.lower() for w in _WORD.findall(row["text"])]
        for k, limit in cfg.max_top_ngram_char_fraction.items():
            frac = top_ngram_char_fraction(words, k)
            if frac > limit:
                return "repetitive_ngrams", f"repetitive top {k}-gram covers {frac:.0%} of text"
    lang_cfg = cfg.language
    if (
        sig["lang"] is not None
        and sig["lang"] not in lang_cfg.allowed
        and sig["lang_conf"] >= lang_cfg.min_confidence
    ):
        return (
            "language",
            f"language '{sig['lang']}' ({sig['lang_conf']:.2f}) not in {lang_cfg.allowed}",
        )
    return None


def run(df: pl.DataFrame, cfg: QualityConfig, text_fields: list[str] | None = None) -> StageResult:
    sigs, rules, reasons = [], [], []
    for row in df.to_dicts():
        sig = signals(row["text"], cfg, row, text_fields)
        failure = first_failure(row, sig, cfg)
        sigs.append(sig)
        rules.append(failure[0] if failure else None)
        reasons.append(failure[1] if failure else None)
    sig_df = pl.DataFrame(sigs, schema_overrides={"lang": pl.Utf8, "lang_conf": pl.Float64})
    scored = pl.concat(
        [df, sig_df.rename({c: f"q_{c}" for c in sig_df.columns})], how="horizontal"
    ).with_columns(
        pl.Series("_rule", rules, dtype=pl.Utf8), pl.Series("_why", reasons, dtype=pl.Utf8)
    )
    kept = scored.filter(pl.col("_rule").is_null()).drop("_rule", "_why")
    failed = scored.filter(pl.col("_rule").is_not_null())
    removed = removed_frame(failed, "quality", pl.col("_why")).drop("_rule", "_why")
    by_rule = failed.group_by("_rule").len().sort("len", descending=True)
    return StageResult(
        name="quality",
        kept=kept,
        removed=removed,
        stats={"by_rule": dict(zip(by_rule["_rule"], by_rule["len"], strict=True))},
    )
