"""Exact (hash) and near (MinHash LSH) deduplication."""

from __future__ import annotations

import hashlib
import re

import polars as pl
from datasketch import MinHash, MinHashLSH

from curate.config import ExactDedupeConfig, NearDedupeConfig
from curate.stages import StageResult, removed_frame

_WORD = re.compile(r"\w+", re.UNICODE)


def _key_text(df: pl.DataFrame, fields: list[str] | None) -> pl.Series:
    if not fields:
        return df["text"]
    return df.select(pl.concat_str([pl.col(f) for f in fields], separator="\n\n")).to_series()


# ---------------------------------------------------------------- exact
def exact_dedupe(df: pl.DataFrame, cfg: ExactDedupeConfig) -> StageResult:
    key = _key_text(df, cfg.fields)
    if cfg.case_insensitive:
        key = key.str.to_lowercase()
    hashes = key.map_elements(
        lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest(), return_dtype=pl.Utf8
    )
    df = df.with_columns(hashes.alias("_hash"))
    first = (
        df.with_row_index("_row")
        .group_by("_hash")
        .agg(
            pl.col("_row").min().alias("_first_row"),
            pl.col("id").sort_by("_row").first().alias("_kept_id"),
            pl.len().alias("_n"),
        )
    )
    df = df.with_row_index("_row").join(first, on="_hash", how="left")
    kept = df.filter(pl.col("_row") == pl.col("_first_row"))
    dropped = df.filter(pl.col("_row") != pl.col("_first_row"))
    removed = removed_frame(dropped, "exact_dedupe", pl.format("exact duplicate of {}", "_kept_id"))
    drop_cols = ["_row", "_first_row", "_kept_id", "_n", "_hash"]
    return StageResult(
        name="exact_dedupe",
        kept=kept.drop(drop_cols),
        removed=removed.drop(drop_cols),
        stats={
            "duplicate_groups": int(first.filter(pl.col("_n") > 1).height),
            "largest_group": int(first["_n"].max() or 0),
        },
    )


# ---------------------------------------------------------------- near
def shingles(text: str, n: int) -> set[str]:
    words = _WORD.findall(text.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # keep the earliest row as the cluster representative
            self.parent[max(ra, rb)] = min(ra, rb)


def near_dedupe(df: pl.DataFrame, cfg: NearDedupeConfig) -> StageResult:
    texts = _key_text(df, cfg.fields).to_list()
    lsh = MinHashLSH(threshold=cfg.threshold, num_perm=cfg.num_perm)
    sigs: list[MinHash] = []
    uf = _UnionFind()
    skipped = 0
    for i, t in enumerate(texts):
        m = MinHash(num_perm=cfg.num_perm, seed=cfg.seed)
        m.update_batch([s.encode("utf-8") for s in shingles(t, cfg.ngram)])
        sigs.append(m)
        if cfg.min_words and len(_WORD.findall(t)) < cfg.min_words:
            skipped += 1
            continue
        for j in lsh.query(m):
            # LSH buckets are approximate; confirm with the signature Jaccard estimate
            if m.jaccard(sigs[j]) >= cfg.threshold:
                uf.union(i, j)
        lsh.insert(i, m)

    rep = [uf.find(i) for i in range(len(texts))]
    ids = df["id"].to_list()
    jacc = [
        None if rep[i] == i else float(sigs[i].jaccard(sigs[rep[i]])) for i in range(len(texts))
    ]
    df = df.with_columns(
        pl.Series("_rep", rep, dtype=pl.Int64),
        pl.Series("_rep_id", [ids[r] for r in rep], dtype=pl.Utf8),
        pl.Series("_jaccard", jacc, dtype=pl.Float64),
    ).with_row_index("_row")

    sizes = df.group_by("_rep").agg(pl.len().alias("cluster_size"))
    clusters = (
        df.join(sizes, on="_rep")
        .filter(pl.col("cluster_size") > 1)
        .select(
            pl.col("_rep_id").alias("cluster_id"),
            "id",
            (pl.col("_row") == pl.col("_rep")).alias("is_representative"),
            pl.col("_jaccard").alias("est_jaccard_to_rep"),
            "cluster_size",
            pl.col("text").str.slice(0, 300).alias("text_preview"),
        )
        .sort(["cluster_size", "cluster_id", "is_representative"], descending=[True, False, True])
    )
    kept = df.filter(pl.col("_row") == pl.col("_rep"))
    dropped = df.filter(pl.col("_row") != pl.col("_rep"))
    removed = removed_frame(
        dropped,
        "near_dedupe",
        pl.format("near duplicate of {} (J≈{})", "_rep_id", pl.col("_jaccard").round(2)),
    )
    tmp = ["_row", "_rep", "_rep_id", "_jaccard"]
    return StageResult(
        name="near_dedupe",
        kept=kept.drop(tmp),
        removed=removed.drop(tmp),
        stats={
            "clusters": int(clusters["cluster_id"].n_unique()) if clusters.height else 0,
            "largest_cluster": int(clusters["cluster_size"].max() or 0) if clusters.height else 0,
            "threshold": cfg.threshold,
            "skipped_short": skipped,
            "num_perm": cfg.num_perm,
            "_clusters": clusters,
        },
    )
