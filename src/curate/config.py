"""Typed pipeline configuration (YAML -> pydantic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class InputConfig(BaseModel):
    paths: list[str]
    format: Literal["jsonl", "parquet", "csv"] = "jsonl"
    id_field: str = "id"
    text_fields: list[str] = Field(default_factory=lambda: ["text"])
    limit: int | None = None


class OutputConfig(BaseModel):
    dir: str = "output/run"
    write_jsonl: bool = True
    max_removed_examples: int = 5


class NormalizeConfig(BaseModel):
    enabled: bool = True
    unicode_form: Literal["NFC", "NFKC", "NFD", "NFKD"] = "NFKC"
    strip_control_chars: bool = True
    collapse_whitespace: bool = True


class ExactDedupeConfig(BaseModel):
    enabled: bool = True
    # Fields that define "the same example". Excluding shared context passages avoids
    # collapsing different questions asked about the same document.
    fields: list[str] | None = None
    case_insensitive: bool = True


class NearDedupeConfig(BaseModel):
    enabled: bool = True
    fields: list[str] | None = None
    threshold: float = 0.8
    num_perm: int = 128
    ngram: int = 5
    seed: int = 42
    min_words: int = 0  # skip very short records (templated prompts look alike but aren't dups)


class LanguageConfig(BaseModel):
    """fastText `lid.176` language ID (see `curate.langid` for model download and caching)."""

    enabled: bool = True
    allowed: list[str] = Field(default_factory=lambda: ["en"])
    # A row is dropped only when a disallowed language scores at least this fastText
    # probability. English prompts dense with foreign names usually score well below 0.9 for
    # the other language, so a high threshold is what keeps them.
    min_confidence: float = 0.5
    min_words: int = 6  # language ID is unreliable on very short strings
    # Which fields to run language ID on (None = combined text). For instruction data the
    # prompt is the best signal: responses are often lists of proper nouns, which confuse
    # character n-gram models, and translation tasks legitimately answer in another language.
    fields: list[str] | None = None
    # lid.176 knows 176 labels. Restricting the pick to widely used languages avoids obscure
    # calls such as an English name list labelled Tagalog ('tl') or Ilocano ('ilo'). The
    # chosen label keeps its raw probability (no renormalisation over the candidates).
    candidates: list[str] | None = Field(
        default_factory=lambda: [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "nl",
            "ru",
            "zh",
            "ja",
            "ko",
            "ar",
            "hi",
            "tr",
            "pl",
            "sv",
            "da",
            "no",
            "fi",
            "cs",
            "el",
            "he",
            "id",
            "vi",
            "th",
            "uk",
            "ro",
            "hu",
        ]
    )


class QualityConfig(BaseModel):
    enabled: bool = True
    required_fields: list[str] = Field(default_factory=list)
    min_words: int = 5
    max_words: int = 10_000
    min_mean_word_length: float = 2.0
    max_mean_word_length: float = 12.0
    max_symbol_word_ratio: float = 0.1
    min_alpha_word_fraction: float = 0.6
    max_duplicate_line_fraction: float = 0.3
    max_top_ngram_char_fraction: dict[int, float] = Field(
        default_factory=lambda: {2: 0.20, 3: 0.18, 4: 0.16}
    )
    repetition_min_words: int = 30
    banned_phrases: list[str] = Field(
        default_factory=lambda: ["lorem ipsum", "enable javascript", "cookie policy"]
    )
    language: LanguageConfig = Field(default_factory=LanguageConfig)


PIIType = Literal[
    "email", "phone", "ipv4", "credit_card", "us_ssn", "street_address", "person_name"
]


class PIIConfig(BaseModel):
    """PII redaction.

    `street_address` and `person_name` are opt-in: they only run when listed in `types`.
    Street addresses use a built-in US regex (no extra dependencies); an NER hook may add more.
    Person names need an NER hook: set `ner` (e.g. "curate.ner:spacy_person_spans" with the
    `ner` extra installed). Without one, `person_name` is skipped with a warning.
    """

    enabled: bool = True
    types: list[PIIType] = Field(
        default_factory=lambda: ["email", "phone", "ipv4", "credit_card", "us_ssn"]
    )
    action: Literal["redact", "drop"] = "redact"
    max_hits_before_drop: int | None = None  # with action=redact, drop docs that are PII-dense
    # Optional "package.module:function" taking str -> list[(start, end, label)] where label is
    # "person_name" or "street_address". See curate/ner.py.
    ner: str | None = None


class ToxicityConfig(BaseModel):
    enabled: bool = True
    lexicon_paths: list[str] = Field(default_factory=lambda: ["lexicons/toxicity_en.txt"])
    max_hits: int = 2
    max_hit_fraction: float = 0.05
    # Optional "package.module:function" taking list[str] -> list[float] (prob. toxic).
    classifier: str | None = None
    classifier_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    # The stage feeds the scorer this many texts per call, so memory stays flat on big runs.
    classifier_batch_size: int = Field(default=32, ge=1)
    # Extra keyword arguments for the scorer, e.g. {model: unitary/toxic-bert, device: cpu}.
    classifier_options: dict[str, Any] = Field(default_factory=dict)


class MitigationConfig(BaseModel):
    pii: PIIConfig = Field(default_factory=PIIConfig)
    toxicity: ToxicityConfig = Field(default_factory=ToxicityConfig)


class StagesConfig(BaseModel):
    normalize: NormalizeConfig = Field(default_factory=NormalizeConfig)
    exact_dedupe: ExactDedupeConfig = Field(default_factory=ExactDedupeConfig)
    near_dedupe: NearDedupeConfig = Field(default_factory=NearDedupeConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    mitigation: MitigationConfig = Field(default_factory=MitigationConfig)


class PipelineConfig(BaseModel):
    name: str = "run"
    input: InputConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    stages: StagesConfig = Field(default_factory=StagesConfig)
    base_dir: Path = Path(".")

    def resolve(self, p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (self.base_dir / path)


def load_config(path: str | Path, overrides: dict | None = None) -> PipelineConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    if overrides:
        data = _deep_merge(data, overrides)
    # Relative paths in a config resolve against the current working directory (repo root).
    data.setdefault("base_dir", str(Path.cwd()))
    return PipelineConfig.model_validate(data)


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = (
            _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        )
    return out
