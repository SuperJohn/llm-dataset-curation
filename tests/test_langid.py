"""fastText lid.176 wrapper: model resolution, label parsing, candidate restriction."""

from __future__ import annotations

import pytest

from curate import langid
from curate.langid import detect_language, predict


def test_predict_strips_label_prefix_and_sorts():
    ranked = predict("The weather today is sunny with a light breeze from the west.", k=5)
    assert len(ranked) == 5
    assert all(not label.startswith("__label__") for label, _ in ranked)
    probs = [p for _, p in ranked]
    assert probs == sorted(probs, reverse=True)
    assert ranked[0][0] == "en" and ranked[0][1] > 0.9


def test_predict_handles_newlines_and_empty_text():
    assert predict("Die Hauptstadt\nvon Deutschland\nist Berlin.", k=1)[0][0] == "de"
    assert predict("   \n ") == []
    assert detect_language("") == ("und", 0.0)


def test_candidates_pick_best_allowed_label_without_renormalising():
    # Tagalog: lid.176 calls it 'tl', which is not in the candidate list below.
    text = "Magandang umaga po, kumusta na kayo? Salamat po sa inyong tulong kahapon."
    assert predict(text, k=1)[0][0] == "tl"
    full = dict(predict(text))
    lang, conf = detect_language(text, candidates=["en", "es", "id"])
    assert lang == max(["en", "es", "id"], key=full.__getitem__)
    assert conf == pytest.approx(full[lang])  # raw probability, not renormalised
    assert conf < 0.5  # so the language rule's confidence gate will not fire on it


def test_env_override_must_exist(monkeypatch, tmp_path):
    monkeypatch.setenv("CURATE_LID_MODEL", str(tmp_path / "missing.ftz"))
    with pytest.raises(FileNotFoundError):
        langid.model_path()


def test_env_override_used_verbatim(monkeypatch, tmp_path):
    fake = tmp_path / "lid.176.bin"
    fake.write_bytes(b"x")
    monkeypatch.setenv("CURATE_LID_MODEL", str(fake))
    assert langid.model_path() == fake


def test_cached_model_is_not_redownloaded(monkeypatch, tmp_path):
    monkeypatch.delenv("CURATE_LID_MODEL", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    cached = tmp_path / "curate" / "lid.176.ftz"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"cached")

    def boom(*a, **k):
        raise AssertionError("should not download when the model is cached")

    monkeypatch.setattr(langid, "_download", boom)
    assert langid.model_path() == cached


def test_download_rejects_checksum_mismatch(monkeypatch, tmp_path):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield b"not a model"

    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse())
    dest = tmp_path / "lid.176.ftz"
    with pytest.raises(OSError, match="checksum mismatch"):
        langid._download("https://example.invalid/lid.176.ftz", dest, langid.LID_SHA256)
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []  # partial file cleaned up
