"""Mitigation: PII redaction (regex + Luhn + optional NER hook) and toxicity filtering
(lexicon + classifier hook)."""

from __future__ import annotations

import importlib
import re
import time
import warnings
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import polars as pl

from curate.config import PIIConfig, ToxicityConfig
from curate.io import with_combined_text
from curate.ner import NER_LABELS, NerHook, load_ner
from curate.stages import StageResult, removed_frame

# --------------------------------------------------------------------------- PII
_STREET_SUFFIX = (
    r"Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl|"
    r"Circle|Cir|Terrace|Ter|Parkway|Pkwy"
)
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "us_ssn": re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "phone": re.compile(
        r"(?<![\w-])(?:\+?1[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?![\w-])"
    ),
    "ipv4": re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
    # US street address: house number, optional direction, 1-4 capitalised or ordinal street
    # words, then a street suffix ("123 Main Street", "350 5th Ave", "10 N. Elm St").
    "street_address": re.compile(
        r"(?<![\w.,/-])\d{1,6}[A-Za-z]?"
        r"(?:\s+(?:[NSEW]|N[EW]|S[EW]|North|South|East|West)\.?)?"
        r"(?:\s+(?:[A-Z][A-Za-z'-]*\.?|\d{1,3}(?:st|nd|rd|th))){1,4}?"
        rf"\s+(?:{_STREET_SUFFIX})\b"
    ),
}
# Order matters: SSN and card numbers before phone so digits aren't double-counted.
# person_name has no regex: it comes only from the NER hook (PIIConfig.ner).
PII_ORDER = ["email", "us_ssn", "credit_card", "phone", "ipv4", "street_address"]
_PLACEHOLDER = re.compile("<(?:" + "|".join(t.upper() for t in {*PII_ORDER, *NER_LABELS}) + ")>")


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


def redact_pii(text: str, types: list[str], ner: NerHook | None = None) -> tuple[str, Counter]:
    """Replace PII with typed placeholders. Regex types run first, then the optional NER hook."""
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
    if ner is not None and text:
        text = _apply_ner(text, types, ner, hits)
    return text, hits


def _apply_ner(text: str, types: list[str], ner: NerHook, hits: Counter) -> str:
    """Redact NER spans for enabled labels, skipping overlaps with each other and with
    placeholders already inserted by the regex pass so nothing is counted twice."""
    taken = [(m.start(), m.end()) for m in _PLACEHOLDER.finditer(text)]
    spans: list[tuple[int, int, str]] = []
    for start, end, label in sorted(ner(text), key=lambda s: (s[0], -s[1])):
        if label not in types or label not in NER_LABELS or not 0 <= start < end <= len(text):
            continue
        if any(start < e and s < end for s, e in taken):
            continue
        taken.append((start, end))
        spans.append((start, end, label))
    # Right to left so earlier offsets stay valid.
    for start, end, label in sorted(spans, reverse=True):
        text = f"{text[:start]}<{label.upper()}>{text[end:]}"
        hits[label] += 1
    return text


def run_pii(df: pl.DataFrame, cfg: PIIConfig, text_fields: list[str]) -> StageResult:
    ner: NerHook | None = None
    if cfg.ner and any(t in cfg.types for t in NER_LABELS):
        ner = load_ner(cfg.ner)
    elif "person_name" in cfg.types:
        warnings.warn(
            "pii.types includes person_name but pii.ner is not set; names will not be redacted "
            "(e.g. ner: 'curate.ner:spacy_person_spans')",
            stacklevel=2,
        )
    rows = df.to_dicts()
    per_row_hits: list[dict] = []
    for row in rows:
        total: Counter = Counter()
        for f in text_fields:
            row[f], h = redact_pii(row[f], cfg.types, ner)
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
            "ner": cfg.ner,
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


def load_classifier(spec: str) -> Callable[..., list[float]]:
    module, _, fn = spec.partition(":")
    if not module or not fn:
        raise ValueError(
            f"toxicity.classifier must look like 'package.module:function', got {spec!r}"
        )
    return getattr(importlib.import_module(module), fn)


def score_in_batches(
    scorer: Callable[..., list[float]],
    texts: list[str],
    batch_size: int,
    options: dict | None = None,
) -> list[float]:
    """Call `scorer` on fixed-size batches and check it returns one probability per text."""
    options = options or {}
    scores: list[float] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        out = [float(s) for s in scorer(batch, **options)]
        if len(out) != len(batch):
            raise ValueError(
                f"toxicity classifier returned {len(out)} scores for a batch of {len(batch)}"
            )
        bad = [s for s in out if not 0.0 <= s <= 1.0]
        if bad:
            raise ValueError(f"toxicity classifier scores must be in [0, 1], got {bad[0]!r}")
        scores.extend(out)
    return scores


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
    clf_stats: dict = {}
    if cfg.classifier:
        t0 = time.perf_counter()
        scores = score_in_batches(
            load_classifier(cfg.classifier),
            texts,
            cfg.classifier_batch_size,
            cfg.classifier_options,
        )
        clf_stats = {
            "classifier_threshold": cfg.classifier_threshold,
            "classifier_batch_size": cfg.classifier_batch_size,
            "classifier_flagged": int(sum(s >= cfg.classifier_threshold for s in scores)),
            "classifier_seconds": round(time.perf_counter() - t0, 3),
        }
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
            **clf_stats,
        },
    )
