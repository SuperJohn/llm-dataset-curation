import random

import polars as pl
import pytest

from curate.config import NearDedupeConfig, SemanticDedupeConfig
from curate.dedupe_compare import (
    HARD_NEGATIVE,
    ORIGINAL,
    PERTURBATIONS,
    MethodScore,
    false_positive_breakdown,
    make_benchmark,
    markdown_table,
    perturb,
    score_method,
)
from curate.stages import StageResult, removed_frame
from curate.stages.dedupe import near_dedupe, semantic_dedupe

FIELDS = ["instruction", "response"]


def _records(n: int = 60, seed: int = 0) -> pl.DataFrame:
    """Distinct long multi-sentence responses built from a random vocabulary."""
    rng = random.Random(seed)
    vocab = [f"{a}{b}" for a in ("ka", "lo", "mi", "ne", "su", "te", "vo") for b in "aeiouxyz"]
    rows = []
    for i in range(n):
        sents = [" ".join(rng.choice(vocab) for _ in range(12)) + "." for _ in range(5)]
        rows.append(
            {
                "id": f"r{i:03d}",
                "instruction": f"Question number {i} about {rng.choice(vocab)}?",
                "response": " ".join(sents),
            }
        )
    return pl.DataFrame(rows)


TEXT = "One two three four five six seven eight. Nine ten eleven twelve. Thirteen fourteen."


@pytest.mark.parametrize("kind", sorted(PERTURBATIONS))
def test_perturb_changes_text_in_the_documented_way(kind):
    rng = random.Random(1)
    out = perturb(TEXT, kind, rng, ["zzz"])
    assert out != TEXT
    words, new = TEXT.split(), out.split()
    if kind == "light_edit":
        assert len(new) == len(words) + 1 and "really" in new and out.count(",") == 1
    elif kind == "word_drop_15":
        assert len(new) == len(words) - round(0.15 * len(words))
        assert all(w in words for w in new)
    elif kind == "sentence_shuffle":
        assert sorted(out.replace(".", "").split()) == sorted(TEXT.replace(".", "").split())
    else:
        assert new == ["zzz" if i % 4 == 3 else w for i, w in enumerate(words)]


def test_perturb_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown perturbation"):
        perturb(TEXT, "paraphrase", random.Random(0), ["x"])


def test_make_benchmark_labels_copies_and_hard_negatives():
    records = _records()
    bench = make_benchmark(records, per_kind=5, hard_negatives=4, seed=3)
    counts = dict(bench.group_by("kind").len().iter_rows())
    assert counts == {**dict.fromkeys(PERTURBATIONS, 5), HARD_NEGATIVE: 4, ORIGINAL: 60 - 4}
    assert "text" in bench.columns

    originals = bench.filter(pl.col("kind") == ORIGINAL)
    copies = bench.filter(pl.col("dup_of").is_not_null())
    # every copy points at a kept original, each source is used once, copies come last
    assert set(copies["dup_of"]) <= set(originals["id"])
    assert copies["dup_of"].n_unique() == copies.height
    assert bench.head(originals.height)["kind"].unique().to_list() == [ORIGINAL]
    # response donors are gone, so a hard negative's response matches nothing left in the set
    negs = bench.filter(pl.col("kind") == HARD_NEGATIVE)
    assert not set(negs["response"]) & set(originals["response"])
    assert set(negs["instruction"]) <= set(originals["instruction"])
    # same seed, same benchmark
    assert bench.equals(make_benchmark(records, per_kind=5, hard_negatives=4, seed=3))


def test_make_benchmark_errors_when_too_few_sources():
    with pytest.raises(ValueError, match="eligible sources"):
        make_benchmark(_records(10), per_kind=5, hard_negatives=0)
    with pytest.raises(ValueError, match="hard negatives"):
        make_benchmark(_records(25), per_kind=5, hard_negatives=5)


def _oracle(bench: pl.DataFrame) -> StageResult:
    """Removes exactly the labelled copies, each into its source's cluster."""
    copies = bench.filter(pl.col("dup_of").is_not_null())
    clusters = pl.concat(
        [
            copies.select(pl.col("dup_of").alias("cluster_id"), "id"),
            copies.select(pl.col("dup_of").alias("cluster_id"), pl.col("dup_of").alias("id")),
        ]
    )
    return StageResult(
        name="oracle",
        kept=bench.filter(pl.col("dup_of").is_null()),
        removed=removed_frame(copies, "oracle", "copy"),
        stats={"_clusters": clusters},
    )


def test_score_method_perfect_and_empty():
    bench = make_benchmark(_records(), per_kind=5, hard_negatives=4)
    perfect = score_method(bench, "oracle", _oracle)
    assert perfect.precision == 1.0 and perfect.false_positives == 0
    assert perfect.removed == perfect.correct == 20
    assert perfect.recall == dict.fromkeys(PERTURBATIONS, 1.0)
    assert false_positive_breakdown(bench, _oracle(bench)) == {}

    def nothing(df):
        return StageResult("none", df, df.clear(), {})

    empty = score_method(bench, "none", nothing)
    assert empty.precision is None and empty.removed == 0
    assert empty.recall == dict.fromkeys(PERTURBATIONS, 0.0)


def test_wrong_cluster_counts_as_false_positive():
    bench = make_benchmark(_records(), per_kind=5, hard_negatives=4)

    def misfiled(df):
        res = _oracle(df)
        res.stats["_clusters"] = res.stats["_clusters"].with_columns(
            pl.lit("r999").alias("cluster_id")
        )
        return res

    score = score_method(bench, "misfiled", misfiled)
    assert score.correct == 0 and score.false_positives == 20 and score.precision == 0.0
    assert sum(false_positive_breakdown(bench, misfiled(bench)).values()) == 20


def test_minhash_misses_reworded_copies_that_embeddings_catch():
    bench = make_benchmark(_records(), per_kind=8, hard_negatives=4)
    minhash = score_method(
        bench,
        "minhash",
        lambda df: near_dedupe(df, NearDedupeConfig(fields=FIELDS, threshold=0.8)),
    )
    hashed = score_method(
        bench,
        "hashing",
        lambda df: semantic_dedupe(
            df, SemanticDedupeConfig(enabled=True, fields=FIELDS, threshold=0.9)
        ),
    )
    assert minhash.recall["light_edit"] == 1.0
    assert minhash.recall["word_swap_25"] == 0.0  # every 5-gram in the response is broken
    assert hashed.recall["word_drop_15"] > minhash.recall["word_drop_15"]
    assert hashed.recall["sentence_shuffle"] >= minhash.recall["sentence_shuffle"]
    assert hashed.precision == 1.0  # synthetic vocabulary has no real duplicates


def test_markdown_table_and_as_dict():
    scores = [
        MethodScore("a", 4, 3, 1, 0.75, {"light_edit": 1.0, "word_swap_25": 0.0}, 0.1),
        MethodScore("b", 0, 0, 0, None, {"light_edit": None}),
    ]
    table = markdown_table(scores).splitlines()
    assert table[0] == (
        "| method | removed | false pos. | precision | recall: light_edit | recall: word_swap_25 |"
    )
    assert table[2] == "| a | 4 | 1 | 0.75 | 1.00 | 0.00 |"
    assert table[3] == "| b | 0 | 0 | n/a | n/a | n/a |"
    assert scores[0].as_dict()["recall"] == {"light_edit": 1.0, "word_swap_25": 0.0}
