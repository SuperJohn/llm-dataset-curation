import sys

import polars as pl
import pytest

from curate.config import PIIConfig
from curate.ner import demo_person_spans, load_ner, spacy_person_spans
from curate.stages.mitigation import redact_pii, run_pii

DEMO = "curate.ner:demo_person_spans"


@pytest.mark.parametrize(
    "text",
    [
        "Ship it to 123 Main Street today.",
        "HQ is 1600 Amphitheatre Parkway, Mountain View.",
        "Meet at 350 5th Avenue",
        "Send to 10 N. Elm St, Springfield.",
        "Mail 742 Evergreen Terrace please",
        "221B Baker St is fiction.",
    ],
)
def test_street_address_regex(text):
    out, hits = redact_pii(text, ["street_address"])
    assert "<STREET_ADDRESS>" in out
    assert hits == {"street_address": 1}


@pytest.mark.parametrize(
    "text",
    [
        "Upgrade to version 2.0 and restart.",
        "In 1999 there were 3 apples on the table.",
        "Abbey Road came out in 1969 and Main Street is a common name.",
        "I ran 5 Miles before breakfast.",
    ],
)
def test_street_address_no_false_positives(text):
    out, hits = redact_pii(text, ["street_address"])
    assert out == text and not hits


def test_street_address_is_opt_in():
    text = "Ship it to 123 Main Street today."
    out, hits = redact_pii(text, PIIConfig().types)
    assert out == text and not hits


def test_demo_hook_spans():
    text = "Ask John Doe or Alice Example, not Bob."
    assert [text[s:e] for s, e, _ in demo_person_spans(text)] == ["John Doe", "Alice Example"]
    assert load_ner(DEMO) is demo_person_spans


def test_load_ner_rejects_bad_spec():
    with pytest.raises(ValueError):
        load_ner("curate.ner")


def test_name_and_address_redacted_with_ner_hook():
    df = pl.DataFrame({"id": ["1"], "text": ["Contact Jane Q. Public at 123 Main Street."]})
    cfg = PIIConfig(types=["person_name", "street_address"], ner=DEMO)
    res = run_pii(df, cfg, ["text"])
    assert res.kept["text"][0] == "Contact <PERSON_NAME> at <STREET_ADDRESS>."
    assert res.stats["hits_by_type"] == {"person_name": 1, "street_address": 1}
    assert res.kept["pii_types"][0] == "person_name,street_address"


def test_person_name_without_ner_leaves_names_alone():
    df = pl.DataFrame({"id": ["1"], "text": ["Contact Jane Q. Public at 123 Main Street."]})
    cfg = PIIConfig(types=["person_name", "street_address"])
    with pytest.warns(UserWarning, match="ner is not set"):
        res = run_pii(df, cfg, ["text"])
    assert res.kept["text"][0] == "Contact Jane Q. Public at <STREET_ADDRESS>."
    assert res.stats["hits_by_type"] == {"street_address": 1}


def test_ner_labels_respect_types():
    # Hook is set but person_name isn't enabled: names stay, regex types still run.
    text = "Email John Doe at jd@example.com"
    out, hits = redact_pii(text, ["email"], demo_person_spans)
    assert out == "Email John Doe at <EMAIL>"
    assert hits == {"email": 1}


def test_ner_overlap_with_regex_is_not_double_counted():
    def greedy(text):  # tags a span covering an existing placeholder, plus a duplicate
        i = text.index("Jane")
        return [(i, len(text), "person_name"), (i, i + 4, "person_name"), (0, 4, "person_name")]

    out, hits = redact_pii(
        "Call Jane at 123 Main Street", ["person_name", "street_address"], greedy
    )
    assert out == "<PERSON_NAME> <PERSON_NAME> at <STREET_ADDRESS>"
    assert hits == {"street_address": 1, "person_name": 2}


def test_spacy_hook_errors_clearly_without_spacy(monkeypatch):
    from curate import ner

    ner._spacy_pipeline.cache_clear()
    monkeypatch.setitem(sys.modules, "spacy", None)  # simulate "not installed"
    with pytest.raises(RuntimeError, match="spaCy is not installed"):
        spacy_person_spans("Jane Q. Public lives here.")
    ner._spacy_pipeline.cache_clear()
