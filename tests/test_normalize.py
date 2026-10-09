from curate.config import NormalizeConfig
from curate.stages.normalize import normalize_text


def test_nfkc_and_whitespace():
    cfg = NormalizeConfig()
    assert normalize_text("ﬁne\u00a0\u00a0 day \u200b!\r\n\r\n\r\nok  ", cfg) == "fine day !\n\nok"


def test_control_chars_removed():
    assert normalize_text("a\x00b\x07c", NormalizeConfig()) == "abc"


def test_empty_passthrough():
    assert normalize_text("", NormalizeConfig()) == ""
