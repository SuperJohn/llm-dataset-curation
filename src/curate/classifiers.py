"""Optional toxicity-classifier hooks.

Point `stages.mitigation.toxicity.classifier` at any "module:function" that takes a list of
strings and returns a list of probabilities in [0, 1]. Examples:

    classifier: "curate.classifiers:detoxify_scores"      # pip install detoxify (downloads weights)
    classifier: "curate.classifiers:keyword_demo_scores"  # dependency-free demo used in tests
"""

from __future__ import annotations


def keyword_demo_scores(texts: list[str]) -> list[float]:
    """Toy scorer for tests and demos: 0.9 if the text contains 'flag me', else 0.0."""
    return [0.9 if "flag me" in t.lower() else 0.0 for t in texts]


def detoxify_scores(texts: list[str], batch_size: int = 64) -> list[float]:
    """Unitary's Detoxify 'original' model (optional dependency)."""
    try:
        from detoxify import Detoxify
    except ImportError as exc:  # pragma: no cover - optional
        raise RuntimeError("pip install detoxify to use this classifier") from exc
    model = Detoxify("original")
    scores: list[float] = []
    for i in range(0, len(texts), batch_size):
        scores.extend(float(s) for s in model.predict(texts[i : i + batch_size])["toxicity"])
    return scores
