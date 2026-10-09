"""Mitigation: PII redaction (regex + Luhn) and toxicity filtering (lexicon + classifier hook)."""

from __future__ import annotations

import importlib
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import polars as pl

from curate.config import PIIConfig, ToxicityConfig
from curate.io import with_combined_text
from curate.stages import StageResult, removed_frame

# --------------------------------------------------------------------------- PII
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "us_ssn": re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "phone": re.compile(
        r"(?<![\w-])(?:\+?1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?![\w-])"
    ),
    "ipv4": re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
}
# Order matters: SSN and card numbers before phone so digits aren't double-counted.
PII_ORDER = ["email", "us_ssn", "credit_card", "phone", "ipv4"]


def luhn_ok(digits: str) -> bool:
    nums = [int(d) for d in digits if d.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(nums)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact_pii(text: str, types: list[str]) -> tuple[str, Counter]:
    hits: Counter = Counter()
    for t in PII_ORDER:
        if t not in types or not text:
            continue
        pat = PII_PATTERNS[t]

        def _sub(m: re.Match[str], t: str = t) -> str:
            if t == "credit_card" and not luhn_ok(m.group(0)):
                return m.group(0)
            hits[t] += 1
            return f"<{t.upper()}>"

        text = pat.sub(_sub, text)
    return text, hits


def run_pii(df: pl.DataFrame, cfg: PIIConfig, text_fields: list[str]) -> StageResult:
    rows = df.to_dicts()
    per_row_hits: list[dict] = []
    for row in rows:
        total: Counter = Counter()
        for f in text_fields:
            row[f], h = redact_pii(row[f], cfg.types)
            total += h
        per_row_hits.append(dict(total))
    out = pl.DataFrame(rows, schema=df.schema)
    out = with_combined_text(out, text_fields).with_columns(
        pl.Series("pii_hits", [sum(h.values()) for h in per_row_hits], dtype=pl.Int64),
        pl.Series(
            "pii_types",
            [",".join(sorted(h)) for h in per_row_hits],
            dtype=pl.Utf8,
        ),
    )
    by_type: Counter = Counter()
    for h in per_row_hits:
        by_type.update(h)

    drop_mask = pl.lit(False)
    if cfg.action == "drop":
        drop_mask = pl.col("pii_hits") > 0
    elif cfg.max_hits_before_drop is not None:
        drop_mask = pl.col("pii_hits") > cfg.max_hits_before_drop
    kept = out.filter(~drop_mask)
    # PII runs right after normalization, so no later stage (or removed.parquet) sees raw PII.
    removed = removed_frame(
        out.filter(drop_mask), "pii", pl.format("PII-dense ({} hits)", "pii_hits")
    )
    return StageResult(
        name="pii",
        kept=kept,
        removed=removed,
        stats={
            "docs_with_pii": int((out["pii_hits"] > 0).sum()),
            "hit_rate": float((out["pii_hits"] > 0).mean() or 0.0),
            "hits_by_type": dict(by_type),
            "action": cfg.action,
        },
    )


# --------------------------------------------------------------------------- toxicity
def load_lexicon(paths: list[Path]) -> list[str]:
    terms: set[str] = set()
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip().lower()
            if line and not line.startswith("#"):
                terms.add(line)
    return sorted(terms, key=len, reverse=True)


def compile_lexicon(terms: list[str]) -> re.Pattern[str] | None:
    if not terms:
        return None
    alt = "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in terms)
    return re.compile(rf"(?<!\w)(?:{alt})(?!\w)", re.IGNORECASE)


def load_classifier(spec: str) -> Callable[[list[str]], list[float]]:
    module, _, fn = spec.partition(":")
    return getattr(importlib.import_module(module), fn)


def run_toxicity(df: pl.DataFrame, cfg: ToxicityConfig, base_dir: Path) -> StageResult:
    paths = [p if Path(p).is_absolute() else base_dir / p for p in map(Path, cfg.lexicon_paths)]
    pattern = compile_lexicon(load_lexicon(paths))
    texts = df["text"].to_list()
    hits = [len(pattern.findall(t)) if pattern else 0 for t in texts]
    n_words = [max(len(t.split()), 1) for t in texts]
    out = df.with_columns(
        pl.Series("tox_lexicon_hits", hits, dtype=pl.Int64),
        pl.Series("tox_hit_frac", [h / n for h, n in zip(hits, n_words, strict=True)]),
    )
    lex_flag = (pl.col("tox_lexicon_hits") >= cfg.max_hits) | (
        (pl.col("tox_hit_frac") > cfg.max_hit_fraction) & (pl.col("tox_lexicon_hits") > 0)
    )
    reason = pl.format("toxic lexicon: {} hits", "tox_lexicon_hits")
    if cfg.classifier:
        scores = load_classifier(cfg.classifier)(texts)
        out = out.with_columns(pl.Series("tox_classifier_score", scores, dtype=pl.Float64))
        clf_flag = pl.col("tox_classifier_score") >= cfg.classifier_threshold
        reason = (
            pl.when(lex_flag)
            .then(reason)
            .otherwise(
                pl.format("toxicity classifier score {}", pl.col("tox_classifier_score").round(2))
            )
        )
        flag = lex_flag | clf_flag
    else:
        flag = lex_flag
    kept = out.filter(~flag)
    removed = removed_frame(out.filter(flag), "toxicity", reason)
    return StageResult(
        name="toxicity",
        kept=kept,
        removed=removed,
        stats={
            "docs_with_lexicon_hits": int(sum(h > 0 for h in hits)),
            "classifier": cfg.classifier,
        },
    )
