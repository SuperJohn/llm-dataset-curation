"""Benchmark fastText lid.176 (current) against py3langid (previous) for the language rule.

Scores both detectors the way the quality stage uses them: same candidate languages, and a row
is "flagged" when the detected language is not English and its confidence is at least the
threshold. Two inputs:

- data/benchmarks/langid_prompts.jsonl: labelled prompts, mostly English questions that are
  dense with foreign proper nouns (the false-flag case from issue #2), plus short non-English
  prompts that must still be caught.
- --dolly PATH: optionally, every `instruction` in databricks-dolly-15k (unlabelled; the
  flagged rows are listed so they can be read by hand).

py3langid is a dev-only dependency (`uv sync` installs it); the pipeline itself never imports it.

    uv run python scripts/benchmark_langid.py
    uv run python scripts/benchmark_langid.py --dolly data/raw/dolly-15k.jsonl --threshold 0.9
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from curate.config import LanguageConfig
from curate.langid import detect_language

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "data" / "benchmarks" / "langid_prompts.jsonl"

Detector = Callable[[str, list[str]], tuple[str, float]]


@lru_cache(maxsize=1)
def _py3langid(candidates: tuple[str, ...]):
    from py3langid.langid import MODEL_FILE, LanguageIdentifier

    ident = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=False)
    ident.set_languages(list(candidates))
    return ident


def py3langid_detect(text: str, candidates: list[str]) -> tuple[str, float]:
    """The pre-#2 detector: py3langid restricted to candidates, softmax over them."""
    ranked = _py3langid(tuple(candidates)).rank(" ".join(text.split()))
    best_lang, best = ranked[0]
    return best_lang, 1.0 / sum(math.exp(s - best) for _, s in ranked)


DETECTORS: dict[str, Detector] = {
    "fasttext-lid.176": detect_language,
    "py3langid": py3langid_detect,
}


def load_prompts(path: Path = PROMPTS) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def flagged(lang: str, conf: float, threshold: float, allowed: tuple[str, ...] = ("en",)) -> bool:
    return lang not in allowed and conf >= threshold


def score(rows: list[dict], detect: Detector, threshold: float, candidates: list[str]) -> dict:
    """English false flags and non-English recall for one detector on labelled rows."""
    en_false, non_en_caught, details = [], 0, []
    n_en = sum(r["lang"] == "en" for r in rows)
    for r in rows:
        lang, conf = detect(r["text"], candidates)
        is_flagged = flagged(lang, conf, threshold)
        details.append({**r, "pred": lang, "conf": round(conf, 3), "flagged": is_flagged})
        if r["lang"] == "en" and is_flagged:
            en_false.append(r["text"])
        elif r["lang"] != "en" and is_flagged:
            non_en_caught += 1
    n_non_en = len(rows) - n_en
    return {
        "n_english": n_en,
        "english_false_flags": len(en_false),
        "english_false_flag_rate": len(en_false) / max(n_en, 1),
        "n_non_english": n_non_en,
        "non_english_recall": non_en_caught / max(n_non_en, 1),
        "false_flag_texts": en_false,
        "rows": details,
    }


def _short(text: str, n: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threshold", type=float, nargs="+", default=[0.5, 0.9])
    ap.add_argument("--dolly", type=Path, help="dolly-15k JSONL to scan (instruction field)")
    args = ap.parse_args(argv)
    candidates = LanguageConfig().candidates or []
    rows = load_prompts()

    print(f"Labelled prompts: {PROMPTS.relative_to(ROOT)} ({len(rows)} rows)\n")
    print("| detector | threshold | English false flags | non-English recall |")
    print("|---|---|---|---|")
    results = {}
    for t in args.threshold:
        for name, det in DETECTORS.items():
            res = results[(name, t)] = score(rows, det, t, candidates)
            print(
                f"| {name} | {t} | {res['english_false_flags']}/{res['n_english']} "
                f"| {res['non_english_recall']:.0%} |"
            )
    print("\nPer-prompt predictions (code, confidence):\n")
    print("| gold | fastText lid.176 | py3langid | prompt |")
    print("|---|---|---|---|")
    ft, old = (results[(n, args.threshold[0])]["rows"] for n in DETECTORS)
    for a, b in zip(ft, old, strict=True):
        print(
            f"| {a['lang']} | {a['pred']} {a['conf']:.2f} | {b['pred']} {b['conf']:.2f} "
            f"| {_short(a['text'])} |"
        )

    if args.dolly:
        instructions = [
            json.loads(line)["instruction"]
            for line in args.dolly.read_text(encoding="utf-8").splitlines()
            if line
        ]
        cfg = LanguageConfig()
        print(f"\n{args.dolly}: {len(instructions):,} instructions\n")
        for t in args.threshold:
            for name, det in DETECTORS.items():
                hits = []
                for text in instructions:
                    if len(text.split()) < 8:  # configs/dolly15k.yaml language.min_words
                        continue
                    lang, conf = det(text, candidates)
                    if flagged(lang, conf, t, tuple(cfg.allowed)):
                        hits.append((lang, conf, text))
                print(f"- {name} @ {t}: {len(hits)} flagged")
                for lang, conf, text in hits:
                    print(f"    {lang} {conf:.2f}  {_short(text)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
