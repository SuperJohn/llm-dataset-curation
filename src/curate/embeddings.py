"""Embedding hooks and similarity search for semantic near-duplicate detection.

Point `stages.semantic_dedupe.embedder` at any "module:function" that takes a list of
strings and returns one vector per string (a 2-D array or a list of lists). The stage calls
it in batches of `batch_size`, passes `embedder_options` as keyword arguments, and
L2-normalises the vectors so that inner product equals cosine similarity. Examples:

    embedder: "curate.embeddings:hashing_embeddings"               # dependency-free default
    embedder: "curate.embeddings:sentence_transformer_embeddings"  # uv sync --extra semantic
    embedder_options: {model: sentence-transformers/all-MiniLM-L6-v2}
"""

from __future__ import annotations

import hashlib
import importlib
import re
from collections.abc import Callable, Iterator
from functools import lru_cache
from typing import Literal

import numpy as np

DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

Embedder = Callable[..., "np.ndarray | list[list[float]]"]

_WORD = re.compile(r"\w+", re.UNICODE)


def load_embedder(spec: str) -> Embedder:
    module, _, fn = spec.partition(":")
    if not module or not fn:
        raise ValueError(
            f"semantic_dedupe.embedder must look like 'package.module:function', got {spec!r}"
        )
    return getattr(importlib.import_module(module), fn)


def hashing_embeddings(texts: list[str], dim: int = 512, ngram: int = 3) -> np.ndarray:
    """Feature-hashed bag of word unigrams and character n-grams.

    No model download and fully deterministic (hashes come from BLAKE2b, not Python's
    salted `hash`). It catches reworded near-copies that share vocabulary, but not true
    paraphrases; use a sentence-transformers model for those.
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        words = _WORD.findall(text.lower())
        joined = " ".join(words)
        feats = words + [joined[i : i + ngram] for i in range(max(len(joined) - ngram + 1, 0))]
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode("utf-8"), digest_size=8).digest(), "little")
            # the top bit picks a sign so colliding features tend to cancel, not pile up
            out[row, h % dim] += 1.0 if (h >> 63) & 1 else -1.0
    return out


@lru_cache(maxsize=2)
def _load_st(model: str, device: str):  # pragma: no cover - heavy
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "sentence_transformer_embeddings needs the 'semantic' extra: uv sync --extra semantic"
        ) from e
    return SentenceTransformer(model, device=device)


def sentence_transformer_embeddings(  # pragma: no cover - needs a model download
    texts: list[str], model: str = DEFAULT_ST_MODEL, device: str = "cpu"
) -> np.ndarray:
    """Dense sentence embeddings from a sentence-transformers checkpoint."""
    return _load_st(model, device).encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )


def embed_in_batches(
    embedder: Embedder, texts: list[str], batch_size: int, options: dict | None = None
) -> np.ndarray:
    """Embed `texts` in fixed-size batches and return L2-normalised float32 rows."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    options = options or {}
    chunks: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = np.asarray(embedder(batch, **options), dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[0] != len(batch):
            raise ValueError(
                f"embedder returned shape {vecs.shape} for a batch of {len(batch)} texts"
            )
        if chunks and vecs.shape[1] != chunks[0].shape[1]:
            raise ValueError(
                f"embedder changed dimension from {chunks[0].shape[1]} to {vecs.shape[1]}"
            )
        chunks.append(vecs)
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32)
    emb = np.vstack(chunks)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    # all-zero rows (empty text) stay zero, so they never match anything
    return np.divide(emb, norms, out=np.zeros_like(emb), where=norms > 0)


def similar_pairs(
    emb: np.ndarray,
    threshold: float,
    index: Literal["exact", "faiss"] = "exact",
    block_size: int = 1024,
) -> Iterator[tuple[int, int, float]]:
    """Yield (i, j, cosine) for every pair with j < i and cosine >= threshold.

    `exact` is a blocked brute-force matmul in numpy: O(n^2) work but only
    O(block_size * n) memory, fine for ~100k rows. `faiss` uses an exact inner-product
    index with range search, which is the same answer computed faster.
    """
    n = emb.shape[0]
    if n < 2:
        return
    if index == "faiss":
        yield from _faiss_pairs(emb, threshold)
        return
    if index != "exact":
        raise ValueError(f"unknown index {index!r}; expected 'exact' or 'faiss'")
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        sims = emb[start:stop] @ emb[:stop].T
        for local, i in enumerate(range(start, stop)):
            row = sims[local, :i]
            for j in np.flatnonzero(row >= threshold):
                yield i, int(j), float(row[j])


def _faiss_pairs(emb: np.ndarray, threshold: float) -> Iterator[tuple[int, int, float]]:
    try:
        import faiss
    except ImportError as e:
        raise ImportError(
            "semantic_dedupe.index='faiss' needs faiss-cpu: uv sync --extra semantic"
        ) from e
    x = np.ascontiguousarray(emb, dtype=np.float32)
    idx = faiss.IndexFlatIP(x.shape[1])
    idx.add(x)
    # range_search keeps scores strictly above the radius; nudge it so >= threshold holds
    lims, dists, labels = idx.range_search(x, float(np.nextafter(np.float32(threshold), -1)))
    for i in range(x.shape[0]):
        for k in range(lims[i], lims[i + 1]):
            j = int(labels[k])
            if j < i and dists[k] >= threshold:
                yield i, j, float(dists[k])
