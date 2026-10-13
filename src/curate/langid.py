"""Language ID with Facebook's fastText `lid.176` model.

The compressed model (`lid.176.ftz`, ~0.9 MB, CC BY-SA 3.0) is downloaded once from
Facebook's CDN into the user cache and reused offline after that:

- `$CURATE_LID_MODEL` points at a local model file (`.ftz` or `.bin`) and skips the download,
  for air-gapped machines or to use the larger, slightly more accurate `lid.176.bin`.
- Otherwise the model lives at `$XDG_CACHE_HOME/curate/lid.176.ftz` (default `~/.cache`).

Confidences are fastText's own softmax probabilities over all 176 labels. Restricting to
`candidates` picks the best-scoring allowed label but keeps its raw probability, so a prompt
that fastText finds ambiguous (English text full of foreign names) stays low-confidence
instead of being renormalised into a confident non-English call.
"""

from __future__ import annotations

import hashlib
import os
import threading
from functools import lru_cache
from pathlib import Path

LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
LID_SHA256 = "8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83"
_LABEL_PREFIX = "__label__"
_lock = threading.Lock()


def cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "curate"


def model_path() -> Path:
    """Local path of the lid.176 model, downloading it on first use."""
    override = os.environ.get("CURATE_LID_MODEL")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"CURATE_LID_MODEL={override} does not exist")
        return path
    path = cache_dir() / "lid.176.ftz"
    with _lock:
        if not path.is_file():
            _download(LID_URL, path, LID_SHA256)
    return path


def _download(url: str, dest: Path, sha256: str) -> None:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(f"{dest.suffix}.part{os.getpid()}")
    digest = hashlib.sha256()
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in r.iter_content(1 << 16):
                    digest.update(chunk)
                    fh.write(chunk)
        if digest.hexdigest() != sha256:
            raise OSError(f"checksum mismatch for {url}: got {digest.hexdigest()}")
        tmp.replace(dest)  # atomic, so a concurrent reader never sees a partial file
    finally:
        tmp.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def _model():
    import fasttext

    path = model_path()
    with _lock:
        return fasttext.load_model(str(path))


def predict(text: str, k: int = -1) -> list[tuple[str, float]]:
    """Top-k (label, probability) pairs, best first. `k=-1` returns all 176 labels."""
    text = " ".join(text.split())  # fastText predicts one line at a time
    if not text:
        return []
    labels, probs = _model().predict(text, k=k, threshold=0.0)
    return [
        (label.removeprefix(_LABEL_PREFIX), float(p))
        for label, p in zip(labels, probs, strict=True)
    ]


def detect_language(text: str, candidates: list[str] | None = None) -> tuple[str, float]:
    """Return (ISO 639 code, fastText probability) for the most likely language.

    With `candidates`, the best label among them wins (falling back to the overall top label
    if none of them scored). Returns ("und", 0.0) for empty text.
    """
    ranked = predict(text)
    if not ranked:
        return "und", 0.0
    if candidates:
        allowed = set(candidates)
        for label, prob in ranked:
            if label in allowed:
                return label, prob
    return ranked[0]
