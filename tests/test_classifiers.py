import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from curate import classifiers
from curate.classifiers import hf_toxicity_scores, label_probabilities
from curate.config import ToxicityConfig
from curate.stages.mitigation import run_toxicity

TOXIC_BERT_LABELS = {
    0: "toxic",
    1: "severe_toxic",
    2: "obscene",
    3: "threat",
    4: "insult",
    5: "identity_hate",
}


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def test_multi_label_head_uses_sigmoid_on_the_named_label():
    logits = np.array([[2.0, -3.0, 0, 0, 0, 0], [-4.0, 5.0, 0, 0, 0, 0]])
    probs = label_probabilities(logits, TOXIC_BERT_LABELS, "toxic", multi_label=True)
    assert probs == pytest.approx([_sigmoid(2.0), _sigmoid(-4.0)])


def test_single_label_head_uses_softmax_and_label_case_is_ignored():
    logits = np.array([[0.0, 0.0], [0.0, math.log(3)]])
    probs = label_probabilities(logits, {0: "neutral", 1: "Toxic"}, "toxic", multi_label=False)
    assert probs == pytest.approx([0.5, 0.75])


def test_one_logit_head_is_a_sigmoid():
    probs = label_probabilities(np.array([[0.0], [3.0]]), {0: "LABEL_0"}, "x", False)
    assert probs == pytest.approx([0.5, _sigmoid(3.0)])


def test_unknown_label_lists_the_model_labels():
    with pytest.raises(ValueError, match="not in model labels"):
        label_probabilities(np.zeros((1, 2)), {0: "neutral", 1: "toxic"}, "hate", False)


class FakeHF:
    """Stands in for a loaded checkpoint: logit +4 on 'toxic' when a text says 'awful'."""

    def __init__(self) -> None:
        self.loads: list[tuple[str, str, int]] = []
        self.batches: list[int] = []

    def load(self, model: str, device: str, max_length: int):
        self.loads.append((model, device, max_length))

        def logits_fn(texts):
            self.batches.append(len(texts))
            rows = [[4.0 if "awful" in t else -4.0, -6.0] for t in texts]
            return np.array(rows), {0: "toxic", 1: "threat"}, True

        return logits_fn


@pytest.fixture
def fake_hf(monkeypatch):
    fake = FakeHF()
    monkeypatch.setattr(classifiers, "_load_hf", fake.load)
    return fake


def test_hf_scores_pass_model_options_through(fake_hf):
    scores = hf_toxicity_scores(["awful", "kind"], model="org/tiny", device="cpu", max_length=64)
    assert scores == pytest.approx([_sigmoid(4.0), _sigmoid(-4.0)])
    assert fake_hf.loads == [("org/tiny", "cpu", 64)]
    assert hf_toxicity_scores([]) == []


def test_hf_classifier_end_to_end_through_the_stage(tmp_path: Path, fake_hf):
    lex = tmp_path / "lex.txt"
    lex.write_text("")
    texts = ["what an awful take", "lovely garden", "awful awful", "tomatoes", "fine"]
    df = pl.DataFrame({"id": list("abcde"), "text": texts})
    cfg = ToxicityConfig(
        lexicon_paths=[str(lex)],
        classifier="curate.classifiers:hf_toxicity_scores",
        classifier_threshold=0.9,
        classifier_batch_size=2,
        classifier_options={"model": "org/tiny", "label": "toxic"},
    )
    res = run_toxicity(df, cfg, tmp_path)
    assert res.removed["id"].to_list() == ["a", "c"]
    assert fake_hf.batches == [2, 2, 1]
    assert res.kept["tox_classifier_score"].max() < 0.1
    assert res.stats["classifier_flagged"] == 2
