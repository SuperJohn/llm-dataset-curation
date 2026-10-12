"""Optional NER hooks for name / address PII.

Point `stages.mitigation.pii.ner` at any "module:function" that takes a string and returns a
list of `(start, end, label)` character spans, where `label` is `person_name` or
`street_address`. Spans are only redacted for labels listed in `pii.types`. Examples:

    ner: "curate.ner:spacy_person_spans"  # pip install 'llm-dataset-curation[ner]' + a model
    ner: "curate.ner:demo_person_spans"   # dependency-free demo used in tests
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from functools import lru_cache

Span = tuple[int, int, str]
NerHook = Callable[[str], list[Span]]

NER_LABELS = ("person_name", "street_address")

# Obviously fake names used in docs and tests.
_DEMO_NAMES = re.compile(r"\b(?:Jane Q\. Public|John Q\. Public|John Doe|Jane Doe|Alice Example)\b")


def demo_person_spans(text: str) -> list[Span]:
    """Toy NER for tests and demos: tags a handful of well-known placeholder names."""
    return [(m.start(), m.end(), "person_name") for m in _DEMO_NAMES.finditer(text)]


@lru_cache(maxsize=4)
def _spacy_pipeline(model: str):
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "spaCy is not installed: pip install 'llm-dataset-curation[ner]' "
            f"&& python -m spacy download {model}"
        ) from exc
    try:
        # Only the NER component is needed; skipping the rest keeps it fast.
        return spacy.load(model, disable=["parser", "lemmatizer", "textcat"])
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model {model!r} not found: python -m spacy download {model}"
        ) from exc


def spacy_person_spans(text: str, model: str = "en_core_web_sm") -> list[Span]:
    """spaCy PERSON entities mapped to `person_name` (optional dependency)."""
    if not text:
        return []
    doc = _spacy_pipeline(model)(text)
    return [(e.start_char, e.end_char, "person_name") for e in doc.ents if e.label_ == "PERSON"]


def load_ner(spec: str) -> NerHook:
    """Resolve a "package.module:function" spec to a callable."""
    module, _, fn = spec.partition(":")
    if not module or not fn:
        raise ValueError(f"ner hook must look like 'package.module:function', got {spec!r}")
    return getattr(importlib.import_module(module), fn)
