"""Compare MinHash and embedding near-dedupe on a labelled perturbation benchmark.

`make_benchmark` takes real records, picks some as *sources*, and appends a perturbed copy of
each one with `dup_of` pointing at its source and `kind` naming the perturbation. It also adds
*hard negatives*: a source's instruction paired with an unrelated record's response, which look
alike on the prompt but are not duplicates (the record that donated the response is dropped from
the benchmark so the pair has no true match). Original rows keep `kind="original"`.

`score_method` runs one dedupe stage over the benchmark and reports, per kind:

  recall     copies removed into their own source's cluster / copies of that kind
  precision  removed copies in the right cluster / all removed rows (overall)

Removed originals count against precision too. Some of those can be genuine duplicates that
were already in the real data, so precision here is a lower bound.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import polars as pl

from curate.io import with_combined_text
from curate.stages import StageResult

PERTURBATIONS: dict[str, str] = {
    "light_edit": "one inserted word and one punctuation tweak",
    "word_drop_15": "15% of response words deleted at random",
    "sentence_shuffle": "response sentences put in a different order",
    "word_swap_25": "every 4th response word replaced by a random corpus word",
}
HARD_NEGATIVE = "hard_negative"
ORIGINAL = "original"

_SENT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z]{3,}")


def _sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(text.strip()) if s]


def perturb(text: str, kind: str, rng: random.Random, vocab: Sequence[str]) -> str:
    """Return a perturbed copy of `text` (a response) for one of `PERTURBATIONS`."""
    words = text.split()
    if kind == "light_edit":
        j = rng.randrange(1, len(words) - 1)
        words[j] = words[j] + ","
        words.insert(rng.randrange(1, len(words) - 1), "really")
        return " ".join(words)
    if kind == "word_drop_15":
        drop = set(rng.sample(range(len(words)), max(1, round(0.15 * len(words)))))
        return " ".join(w for i, w in enumerate(words) if i not in drop)
    if kind == "sentence_shuffle":
        sents = _sentences(text)
        order = list(range(len(sents)))
        while order == sorted(order):
            rng.shuffle(order)
        return " ".join(sents[i] for i in order)
    if kind == "word_swap_25":
        return " ".join(rng.choice(vocab) if i % 4 == 3 else w for i, w in enumerate(words))
    raise ValueError(f"unknown perturbation {kind!r}; expected one of {sorted(PERTURBATIONS)}")


def _eligible(kind: str, response: str, min_words: int) -> bool:
    if len(response.split()) < min_words:
        return False
    return kind != "sentence_shuffle" or len(_sentences(response)) >= 3


def make_benchmark(
    records: pl.DataFrame,
    per_kind: int = 50,
    hard_negatives: int = 50,
    min_words: int = 40,
    seed: int = 7,
    kinds: Sequence[str] = tuple(PERTURBATIONS),
) -> pl.DataFrame:
    """Originals followed by labelled perturbed copies and hard negatives.

    `records` needs `id`, `instruction` and `response`. Each source is used at most once, so
    every copy has exactly one true match. Copies come after all originals, so a correct
    dedupe keeps the original as the cluster representative.
    """
    rng = random.Random(seed)
    base = records.select("id", "instruction", "response").with_columns(
        pl.col("instruction").fill_null(""), pl.col("response").fill_null("")
    )
    rows = base.to_dicts()
    vocab = sorted({w.lower() for r in rows for w in _WORD.findall(r["response"])})
    if not vocab:
        raise ValueError("records have no response words to build a vocabulary from")
    order = list(range(len(rows)))
    rng.shuffle(order)
    used: set[int] = set()
    extra: list[dict] = []

    for kind in kinds:
        picks = [
            i for i in order if i not in used and _eligible(kind, rows[i]["response"], min_words)
        ][:per_kind]
        if len(picks) < per_kind:
            raise ValueError(f"only {len(picks)} eligible sources for {kind}, need {per_kind}")
        for i in picks:
            used.add(i)
            src = rows[i]
            extra.append(
                {
                    "id": f"{src['id']}~{kind}",
                    "instruction": src["instruction"],
                    "response": perturb(src["response"], kind, rng, vocab),
                    "dup_of": src["id"],
                    "kind": kind,
                }
            )

    free = [i for i in order if i not in used and len(rows[i]["response"].split()) >= min_words]
    if len(free) < 2 * hard_negatives:
        raise ValueError(f"need {2 * hard_negatives} unused sources for hard negatives")
    donors: set[str] = set()
    for n in range(hard_negatives):
        a, b = rows[free[2 * n]], rows[free[2 * n + 1]]
        donors.add(b["id"])
        extra.append(
            {
                "id": f"{a['id']}~{HARD_NEGATIVE}",
                "instruction": a["instruction"],
                "response": b["response"],
                "dup_of": None,
                "kind": HARD_NEGATIVE,
            }
        )

    # the response donors leave the benchmark, or each hard negative would be a true copy
    originals = base.filter(~pl.col("id").is_in(sorted(donors))).with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("dup_of"), pl.lit(ORIGINAL).alias("kind")
    )
    out = pl.concat([originals, pl.DataFrame(extra, schema=originals.schema)])
    return with_combined_text(out, ["instruction", "response"])


@dataclass
class MethodScore:
    method: str
    removed: int
    correct: int
    false_positives: int
    precision: float | None
    recall: dict[str, float | None]
    seconds: float | None = None

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "removed": self.removed,
            "correct": self.correct,
            "false_positives": self.false_positives,
            "precision": self.precision,
            "recall": dict(self.recall),
            "seconds": self.seconds,
        }


def _rate(num: int, den: int) -> float | None:
    return None if den == 0 else round(num / den, 4)


def _cluster_of(result: StageResult) -> dict[str, str]:
    clusters = next(
        (v for k, v in result.stats.items() if k.endswith("clusters") and k.startswith("_")),
        None,
    )
    if clusters is None or clusters.is_empty():
        return {}
    return dict(zip(clusters["id"].to_list(), clusters["cluster_id"].to_list(), strict=True))


def score_method(
    bench: pl.DataFrame, name: str, run: Callable[[pl.DataFrame], StageResult]
) -> MethodScore:
    """Run one dedupe stage on `bench` and score its removals against the labels."""
    t0 = time.perf_counter()
    result = run(bench)
    seconds = round(time.perf_counter() - t0, 2)
    cluster = _cluster_of(result)
    removed_ids = set(result.removed["id"].to_list()) if result.removed.height else set()
    labels = {
        r["id"]: (r["kind"], r["dup_of"]) for r in bench.select("id", "kind", "dup_of").to_dicts()
    }

    correct = {
        i for i in removed_ids if labels[i][1] is not None and cluster.get(i) == labels[i][1]
    }
    kinds = [k for k in dict.fromkeys(bench["kind"].to_list()) if k in PERTURBATIONS]
    recall = {}
    for kind in kinds:
        ids = [i for i, (k, _) in labels.items() if k == kind]
        recall[kind] = _rate(sum(1 for i in ids if i in correct), len(ids))
    fps = len(removed_ids) - len(correct)
    return MethodScore(
        method=name,
        removed=len(removed_ids),
        correct=len(correct),
        false_positives=fps,
        precision=_rate(len(correct), len(removed_ids)),
        recall=recall,
        seconds=seconds,
    )


def false_positive_breakdown(bench: pl.DataFrame, run_result: StageResult) -> dict[str, int]:
    """Count wrong removals by the `kind` of the removed row."""
    cluster = _cluster_of(run_result)
    labels = {
        r["id"]: (r["kind"], r["dup_of"]) for r in bench.select("id", "kind", "dup_of").to_dicts()
    }
    out: dict[str, int] = {}
    for i in run_result.removed["id"].to_list():
        kind, dup_of = labels[i]
        if dup_of is None or cluster.get(i) != dup_of:
            out[kind] = out.get(kind, 0) + 1
    return out


def markdown_table(scores: Sequence[MethodScore]) -> str:
    """One row per method: precision, then recall for each perturbation kind."""
    kinds = list(dict.fromkeys(k for s in scores for k in s.recall))

    def fmt(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.2f}"

    head = ["method", "removed", "false pos.", "precision", *[f"recall: {k}" for k in kinds]]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in scores:
        cells = [
            s.method,
            str(s.removed),
            str(s.false_positives),
            fmt(s.precision),
            *[fmt(s.recall.get(k)) for k in kinds],
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
