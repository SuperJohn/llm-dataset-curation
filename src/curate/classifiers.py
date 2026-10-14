"""Optional toxicity-classifier hooks.

Point `stages.mitigation.toxicity.classifier` at any "module:function" that takes a list of
strings and returns a list of probabilities in [0, 1]. The stage calls it in batches of
`classifier_batch_size` and passes `classifier_options` as keyword arguments. Examples:

    classifier: "curate.classifiers:hf_toxicity_scores"   # uv sync --extra toxicity
    classifier_options: {model: unitary/toxic-bert, label: toxic}
    classifier: "curate.classifiers:detoxify_scores"      # pip install detoxify
    classifier: "curate.classifiers:keyword_demo_scores"  # dependency-free demo used in tests
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import numpy as np

DEFAULT_HF_MODEL = "unitary/toxic-bert"

# (texts) -> (logits [n, num_labels], id2label, multi_label)
LogitsFn = Callable[[list[str]], tuple[np.ndarray, dict[int, str], bool]]


def keyword_demo_scores(texts: list[str]) -> list[float]:
    """Toy scorer for tests and demos: 0.9 if the text contains 'flag me', else 0.0."""
    return [0.9 if "flag me" in t.lower() else 0.0 for t in texts]


def label_probabilities(
    logits: np.ndarray, id2label: dict[int, str], label: str, multi_label: bool
) -> list[float]:
    """Probability of `label` for each row of logits.

    Multi-label heads (toxic-bert's toxic / insult / threat ...) are independent sigmoids.
    Single-label heads (toxic vs. neutral) are a softmax across classes. A one-logit head
    is read as a sigmoid on the positive class.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"expected 2-D logits, got shape {logits.shape}")
    if logits.shape[1] == 1:
        return (1.0 / (1.0 + np.exp(-logits[:, 0]))).tolist()
    by_name = {str(v).lower(): int(k) for k, v in id2label.items()}
    if label.lower() not in by_name:
        raise ValueError(f"label {label!r} not in model labels {sorted(by_name)}")
    idx = by_name[label.lower()]
    if multi_label:
        probs = 1.0 / (1.0 + np.exp(-logits[:, idx]))
    else:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp[:, idx] / exp.sum(axis=1)
    return probs.tolist()


@lru_cache(maxsize=4)
def _load_hf(model: str, device: str, max_length: int) -> LogitsFn:  # pragma: no cover - heavy
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "hf_toxicity_scores needs transformers + torch: uv sync --extra toxicity"
        ) from exc
    tok = AutoTokenizer.from_pretrained(model)
    net = AutoModelForSequenceClassification.from_pretrained(model).to(device).eval()
    id2label = {int(k): v for k, v in net.config.id2label.items()}
    multi_label = net.config.problem_type == "multi_label_classification"

    def logits_fn(texts: list[str]) -> tuple[np.ndarray, dict[int, str], bool]:
        enc = tok(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.inference_mode():
            out = net(**enc.to(device)).logits
        return out.float().cpu().numpy(), id2label, multi_label

    return logits_fn


def hf_toxicity_scores(
    texts: list[str],
    model: str = DEFAULT_HF_MODEL,
    label: str = "toxic",
    device: str = "cpu",
    max_length: int = 256,
    multi_label: bool | None = None,
) -> list[float]:
    """Any Hugging Face sequence-classification model, scored on one batch of texts.

    The model loads once per process (cached) and runs under `torch.inference_mode()`.
    `multi_label` defaults to the model's own `problem_type`; set it to force sigmoid
    (True) or softmax (False) for checkpoints that don't record one.
    """
    if not texts:
        return []
    logits, id2label, model_multi = _load_hf(model, device, max_length)(texts)
    use_multi = model_multi if multi_label is None else multi_label
    return label_probabilities(logits, id2label, label, use_multi)


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
