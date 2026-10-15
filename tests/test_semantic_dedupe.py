import json
import sys
import textwrap

import numpy as np
import polars as pl
import pytest

from curate.config import NearDedupeConfig, SemanticDedupeConfig
from curate.embeddings import embed_in_batches, hashing_embeddings, load_embedder, similar_pairs
from curate.pipeline import run_pipeline
from curate.stages.dedupe import near_dedupe, semantic_dedupe

BASE = (
    "The quick brown fox jumps over the lazy dog while the farmer watches from the porch "
    "and the sun sets slowly behind the distant rolling hills of the quiet valley. Later "
    "that evening the farmer walks down to the barn, checks on the horses, fills the water "
    "troughs, locks the gate and returns to the house where dinner is waiting on the table."
)


def _df(texts):
    return pl.DataFrame({"id": [f"r{i}" for i in range(len(texts))], "text": texts})


def _reword(text: str, every: int = 4) -> str:
    """Change every `every`-th word: breaks most 5-gram shingles, keeps the vocabulary."""
    words = text.split()
    return " ".join(w.upper() + "s" if i % every == 0 else w for i, w in enumerate(words))


@pytest.fixture()
def vector_embedder(tmp_path, monkeypatch):
    """A module whose embedder maps a text like 'v:1,0' straight to the vector [1, 0]."""
    (tmp_path / "fake_vectors.py").write_text(
        textwrap.dedent(
            """
            def embed(texts, scale=1.0):
                out = []
                for t in texts:
                    head = t.split()[0]
                    out.append([scale * float(x) for x in head.removeprefix("v:").split(",")])
                return out
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "fake_vectors:embed"
    sys.modules.pop("fake_vectors", None)


# ---------------------------------------------------------------- embeddings
def test_hashing_embeddings_are_deterministic_and_sized():
    a = hashing_embeddings(["hello world", "other text"], dim=64)
    b = hashing_embeddings(["hello world", "other text"], dim=64)
    assert a.shape == (2, 64) and a.dtype == np.float32
    assert np.array_equal(a, b)
    with pytest.raises(ValueError, match="dim"):
        hashing_embeddings(["x"], dim=0)


def test_embed_in_batches_normalises_and_validates():
    emb = embed_in_batches(lambda b: [[3.0, 4.0]] * len(b), ["a", "b", "c"], batch_size=2)
    assert emb.shape == (3, 2)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0)
    zero = embed_in_batches(lambda b: [[0.0, 0.0]] * len(b), ["a"], batch_size=4)
    assert np.array_equal(zero, np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        embed_in_batches(lambda b: [[1.0, 0.0]], ["a", "b"], batch_size=2)
    with pytest.raises(ValueError, match="batch_size"):
        embed_in_batches(lambda b: [], ["a"], batch_size=0)


def test_load_embedder_rejects_bad_spec():
    with pytest.raises(ValueError, match="module:function"):
        load_embedder("curate.embeddings")
    assert load_embedder("curate.embeddings:hashing_embeddings") is hashing_embeddings


def test_similar_pairs_exact_matches_brute_force_across_blocks():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(37, 8)).astype(np.float32)
    emb[5] = emb[2] + 0.01  # a guaranteed close pair
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    sims = emb @ emb.T
    want = {(i, j) for i in range(37) for j in range(i) if sims[i, j] >= 0.6}
    got = {(i, j) for i, j, _ in similar_pairs(emb, 0.6, block_size=5)}
    assert (5, 2) in got
    assert got == want


def test_similar_pairs_faiss_matches_exact():
    pytest.importorskip("faiss")
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(50, 16)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    exact = {(i, j) for i, j, _ in similar_pairs(emb, 0.4)}
    fast = {(i, j) for i, j, _ in similar_pairs(emb, 0.4, index="faiss")}
    assert fast == exact


def test_similar_pairs_rejects_unknown_index():
    with pytest.raises(ValueError, match="unknown index"):
        list(similar_pairs(np.eye(3, dtype=np.float32), 0.5, index="hnsw"))


# ---------------------------------------------------------------- stage
def test_semantic_catches_rewording_that_minhash_misses():
    reworded = _reword(BASE)
    unrelated = "Completely different text about databases, indexes and query planning " * 3
    df = _df([BASE, reworded, unrelated])
    assert near_dedupe(df, NearDedupeConfig(threshold=0.8)).kept.height == 3
    res = semantic_dedupe(df, SemanticDedupeConfig(enabled=True, threshold=0.8))
    assert res.kept["id"].to_list() == ["r0", "r2"]
    assert res.removed["id"].to_list() == ["r1"]
    assert res.removed["_reason"][0].startswith("semantic duplicate of r0 (cos≈")
    clusters = res.stats["_semantic_clusters"]
    assert set(clusters["id"]) == {"r0", "r1"}
    assert clusters.filter(pl.col("is_representative"))["id"].to_list() == ["r0"]


def test_semantic_clusters_are_transitive_and_keep_earliest(vector_embedder):
    # r0~r1 and r1~r2 clear 0.9, r0~r2 does not; union-find still makes one cluster
    a, b = np.cos(np.radians(20)), np.sin(np.radians(20))
    c, d = np.cos(np.radians(40)), np.sin(np.radians(40))
    df = _df(["v:1,0 x", f"v:{a},{b} y", f"v:{c},{d} z", "v:0,1 w"])
    cfg = SemanticDedupeConfig(enabled=True, embedder=vector_embedder, threshold=0.9)
    res = semantic_dedupe(df, cfg)
    assert res.kept["id"].to_list() == ["r0", "r3"]
    assert res.stats["clusters"] == 1 and res.stats["largest_cluster"] == 3
    assert res.stats["similar_pairs"] == 2 and res.stats["dim"] == 2


def test_semantic_min_words_and_options(vector_embedder):
    df = _df(["v:1,0 short", "v:1,0 short", "v:1,0 one two three four", "v:1,0 a b c d"])
    cfg = SemanticDedupeConfig(
        enabled=True,
        embedder=vector_embedder,
        embedder_options={"scale": 2.0},
        threshold=0.99,
        min_words=6,  # "v:1,0 short" is 4 regex words
        batch_size=3,
    )
    res = semantic_dedupe(df, cfg)
    assert res.kept["id"].to_list() == ["r0", "r1", "r2"]
    assert res.stats["skipped_short"] == 2


def test_semantic_is_off_by_default_and_runs_in_pipeline(sample_config):
    assert "semantic_dedupe" not in [f["stage"] for f in run_pipeline(sample_config, False).funnel]
    sample_config.stages.semantic_dedupe = SemanticDedupeConfig(
        enabled=True, fields=["instruction", "response"], threshold=0.95, min_words=20
    )
    run = run_pipeline(sample_config)
    stages = [f["stage"] for f in run.funnel]
    assert stages.index("near_dedupe") < stages.index("semantic_dedupe") < stages.index("quality")
    stats = json.loads((run.out_dir / "stats.json").read_text())
    sem = stats["stages"]["semantic_dedupe"]
    assert "_semantic_clusters" not in sem
    assert sem["embedder"] == "curate.embeddings:hashing_embeddings"
    if sem["clusters"]:
        assert (run.out_dir / "semantic_clusters.parquet").exists()
    assert not set(run.curated["id"]) & set(run.removed["id"])
