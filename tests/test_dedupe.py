import polars as pl

from curate.config import ExactDedupeConfig, NearDedupeConfig
from curate.stages.dedupe import exact_dedupe, near_dedupe, shingles

BASE = (
    "The quick brown fox jumps over the lazy dog while the farmer watches from the porch "
    "and the sun sets slowly behind the distant rolling hills of the quiet valley. Later "
    "that evening the farmer walks down to the barn, checks on the horses, fills the water "
    "troughs, locks the gate and returns to the house where dinner is waiting on the table "
    "and the radio plays an old song about rivers, trains and long summer nights."
)


def _df(texts):
    return pl.DataFrame({"id": [f"r{i}" for i in range(len(texts))], "text": texts})


def test_exact_keeps_first_case_insensitive():
    res = exact_dedupe(_df(["Hello world", "hello WORLD", "other"]), ExactDedupeConfig())
    assert res.kept["id"].to_list() == ["r0", "r2"]
    assert res.removed["_reason"].to_list() == ["exact duplicate of r0"]
    assert res.stats["duplicate_groups"] == 1


def test_exact_on_selected_fields():
    df = pl.DataFrame(
        {
            "id": ["a", "b"],
            "instruction": ["Q1", "Q2"],
            "context": ["same passage", "same passage"],
            "response": ["A", "A"],
        }
    ).with_columns(text=pl.lit("x"))
    res = exact_dedupe(df, ExactDedupeConfig(fields=["instruction", "response"]))
    assert res.kept.height == 2  # same context, different question -> not a duplicate


def test_near_dedupe_clusters_light_edits():
    edited = BASE.replace("quick", "quick,").replace("Later", "Much later")
    unrelated = "Completely different text about databases, indexes and query planning " * 2
    res = near_dedupe(_df([BASE, edited, unrelated]), NearDedupeConfig(threshold=0.7))
    assert res.kept["id"].to_list() == ["r0", "r2"]
    assert res.removed["id"].to_list() == ["r1"]
    clusters = res.stats["_clusters"]
    assert set(clusters["id"]) == {"r0", "r1"}
    assert clusters.filter(pl.col("is_representative"))["id"].to_list() == ["r0"]


def test_near_dedupe_is_deterministic():
    texts = [BASE, BASE + " extra", "something else entirely here"]
    a = near_dedupe(_df(texts), NearDedupeConfig()).kept["id"].to_list()
    b = near_dedupe(_df(texts), NearDedupeConfig()).kept["id"].to_list()
    assert a == b


def test_shingles_short_text():
    assert shingles("two words", 5) == {"two words"}
    assert shingles("", 5) == set()
