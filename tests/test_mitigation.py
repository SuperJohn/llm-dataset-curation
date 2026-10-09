from pathlib import Path

import polars as pl

from curate.config import PIIConfig, ToxicityConfig
from curate.stages.mitigation import luhn_ok, redact_pii, run_pii, run_toxicity

ALL = ["email", "phone", "ipv4", "credit_card", "us_ssn"]


def test_redacts_each_pii_type():
    text = (
        "Mail a.b@example.com, call (555) 867-5309, host 192.0.2.10, "
        "card 4111 1111 1111 1111, ssn 078-05-1120."
    )
    out, hits = redact_pii(text, ALL)
    assert "<EMAIL>" in out and "<PHONE>" in out and "<IPV4>" in out
    assert "<CREDIT_CARD>" in out and "<US_SSN>" in out
    assert hits == {"email": 1, "phone": 1, "ipv4": 1, "credit_card": 1, "us_ssn": 1}


def test_no_false_positives_on_years_and_versions():
    text = "In 1999 there were 3.14 apples and version 2.0.1 shipped on 2023-05-01."
    out, hits = redact_pii(text, ALL)
    assert out == text and not hits


def test_luhn():
    assert luhn_ok("4111 1111 1111 1111")
    assert not luhn_ok("4111 1111 1111 1112")


def test_run_pii_redacts_all_fields_and_rebuilds_text():
    df = pl.DataFrame(
        {"id": ["1"], "instruction": ["email me"], "response": ["x@example.com"], "text": [""]}
    )
    res = run_pii(df, PIIConfig(), ["instruction", "response"])
    row = res.kept.row(0, named=True)
    assert row["response"] == "<EMAIL>"
    assert row["text"] == "email me\n\n<EMAIL>"
    assert res.stats["hits_by_type"] == {"email": 1}


def test_pii_drop_mode():
    df = pl.DataFrame({"id": ["1", "2"], "text": ["x@example.com", "clean"]})
    res = run_pii(df, PIIConfig(action="drop"), ["text"])
    assert res.kept["id"].to_list() == ["2"]


def test_toxicity_lexicon(tmp_path: Path):
    lex = tmp_path / "lex.txt"
    lex.write_text("# comment\nidiot\nshut up\n")
    df = pl.DataFrame(
        {
            "id": ["a", "b", "c"],
            "text": [
                "You idiot, shut up already.",
                "A perfectly nice sentence about gardening and tomatoes in summer.",
                "Shutup is not a match, nor is idiotic.",
            ],
        }
    )
    res = run_toxicity(df, ToxicityConfig(lexicon_paths=[str(lex)], max_hits=2), tmp_path)
    assert res.removed["id"].to_list() == ["a"]


def test_toxicity_classifier_hook(tmp_path: Path):
    lex = tmp_path / "lex.txt"
    lex.write_text("")
    df = pl.DataFrame({"id": ["a", "b"], "text": ["fine", "flag me please"]})
    cfg = ToxicityConfig(
        lexicon_paths=[str(lex)],
        classifier="curate.classifiers:keyword_demo_scores",
        classifier_threshold=0.5,
    )
    res = run_toxicity(df, cfg, tmp_path)
    assert res.removed["id"].to_list() == ["b"]
    assert "tox_classifier_score" in res.kept.columns
