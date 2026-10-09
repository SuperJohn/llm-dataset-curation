"""Unicode normalization + whitespace/control-character cleanup. Never removes rows."""

from __future__ import annotations

import re
import unicodedata

import polars as pl

from curate.config import NormalizeConfig
from curate.io import with_combined_text
from curate.stages import StageResult, removed_frame

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b-\u200d\ufeff]")
_SPACES = re.compile(r"[ \t\u00a0\u2000-\u200a\u3000]+")
_BLANKLINES = re.compile(r"\n{3,}")


def normalize_text(s: str, cfg: NormalizeConfig) -> str:
    if not s:
        return s
    s = unicodedata.normalize(cfg.unicode_form, s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    if cfg.strip_control_chars:
        s = _CONTROL.sub("", s)
    if cfg.collapse_whitespace:
        s = _SPACES.sub(" ", s)
        s = "\n".join(line.strip() for line in s.split("\n"))
        s = _BLANKLINES.sub("\n\n", s)
    return s.strip()


def run(df: pl.DataFrame, cfg: NormalizeConfig, text_fields: list[str]) -> StageResult:
    out = df.with_columns(
        [
            pl.col(f).map_elements(lambda s: normalize_text(s, cfg), return_dtype=pl.Utf8)
            for f in text_fields
        ]
    )
    out = with_combined_text(out, text_fields)
    rows_modified = int((df["text"] != out["text"]).sum())
    return StageResult(
        name="normalize",
        kept=out,
        removed=removed_frame(out.clear(), "normalize", ""),
        stats={"rows_modified": rows_modified},
    )
