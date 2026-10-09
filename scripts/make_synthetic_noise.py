"""Generate data/sample/synthetic_noise.jsonl: labelled, fake "bad" records mixed into the sample.

Real instruction datasets are fairly clean, so a small synthetic slice makes every stage of the
pipeline observable in the eval report. All PII below is fake (example.com, 555 numbers,
documentation IP ranges). Each record carries `noise_type` so tests can assert the right stage
caught it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "dolly_sample.jsonl"
OUT = ROOT / "data" / "sample" / "synthetic_noise.jsonl"


def rec(i: int, noise: str, instruction: str, response: str, context: str = "") -> dict:
    return {
        "id": f"synthetic-{i:03d}",
        "instruction": instruction,
        "context": context,
        "response": response,
        "category": "synthetic",
        "noise_type": noise,
    }


def main() -> None:
    rng = random.Random(1234)
    base = [json.loads(line) for line in SAMPLE.read_text().splitlines()]
    long_ones = [b for b in base if len(b["response"].split()) > 60]
    picks = rng.sample(long_ones, 16)
    out: list[dict] = []
    i = 0

    # exact duplicates hidden behind whitespace / unicode / case differences
    for b in picks[:6]:
        out.append(
            rec(
                i,
                "exact_dup_after_normalize",
                "  " + b["instruction"].upper() + "\u00a0",
                b["response"].replace(" ", "  ", 3) + "\n\n",
                b["context"],
            )
        )
        i += 1

    # near duplicates: light edits of an existing record
    for b in picks[6:14]:
        words = b["response"].split()
        j = rng.randrange(5, len(words) - 5)
        words[j] = words[j] + ","  # punctuation tweak
        words.insert(rng.randrange(5, len(words) - 5), "really")  # one inserted word
        out.append(rec(i, "near_dup", b["instruction"], " ".join(words), b["context"]))
        i += 1

    pii = [
        (
            "How do I reach support?",
            "Email jane.doe@example.com or call (555) 867-5309 any weekday.",
        ),
        (
            "Who should I contact about the invoice?",
            "Send it to billing@example.org and cc accounts.team@example.net. "
            "If urgent, phone +1 555 010 4477.",
        ),
        (
            "What is the server address?",
            "The staging box lives at 192.0.2.44 and the backup at 198.51.100.7.",
        ),
        (
            "Can you store my card?",
            "Sure, the card on file is 4111 1111 1111 1111 with expiry 12/29.",
        ),
        ("What's my SSN on record?", "The record shows 078-05-1120 for the account holder."),
        (
            "Give me the team roster",
            "Alex: alex@example.com, 555-123-4567. Sam: sam@example.com, 555-987-6543. "
            "Both are on call this week and can be paged through the usual rota.",
        ),
    ]
    for q, a in pii:
        out.append(rec(i, "pii", q, a))
        i += 1

    non_english = [
        (
            "¿Cuál es la capital de Francia y por qué es tan famosa en el mundo?",
            "La capital de Francia es París, una ciudad conocida por su arte, su gastronomía "
            "y monumentos como la Torre Eiffel.",
        ),
        (
            "Was ist die Hauptstadt von Deutschland und wofür ist die Stadt bekannt?",
            "Die Hauptstadt von Deutschland ist Berlin. Die Stadt ist bekannt für ihre Geschichte, "
            "ihre Museen und ihr lebendiges Kulturleben.",
        ),
        (
            "Quelle est la meilleure saison pour visiter le Japon et pourquoi ?",
            "Le printemps est souvent recommandé, car les cerisiers sont en fleurs et le climat "
            "est agréable pour se promener dans les jardins.",
        ),
        (
            "Qual é o maior rio do Brasil e por que ele é tão importante?",
            "O maior rio do Brasil é o Amazonas, que atravessa a floresta e tem enorme importância "
            "para o clima e a biodiversidade da região.",
        ),
    ]
    for q, a in non_english:
        out.append(rec(i, "non_english", q, a))
        i += 1

    repetitive = [
        ("Write an ad", "Buy now! " * 40),
        ("List fruits", "\n".join(["apple banana apple banana"] * 25)),
        ("Describe the product", " ".join(["best price best quality best service"] * 15)),
    ]
    for q, a in repetitive:
        out.append(rec(i, "repetitive", q, a))
        i += 1

    out.append(rec(i, "too_short", "Hi", "ok"))
    i += 1  # noqa: E702
    out.append(rec(i, "too_short", "?", ""))
    i += 1  # noqa: E702
    out.append(
        rec(
            i,
            "boilerplate",
            "Summarize the page",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor "
            "incididunt ut labore et dolore magna aliqua. Lorem ipsum dolor sit amet.",
        )
    )
    i += 1  # noqa: E702
    out.append(
        rec(
            i,
            "symbol_noise",
            "Parse this",
            "#### $$$ @@@ ### %%% ^^^ &&& *** ((( ))) ### $$$ @@@ ### %%% ^^^ &&& *** {{{ }}} "
            ">>> <<< ### ||| ~~~ ### $$$ ok",
        )
    )
    i += 1  # noqa: E702

    toxic = [
        (
            "What do you think of my essay?",
            "Honestly you are a complete idiot and a moron, this is garbage written by a loser. "
            "Shut up and stop writing.",
        ),
        (
            "Reply to the customer",
            "Tell that stupid idiot to go away, nobody cares about his pathetic loser complaints.",
        ),
    ]
    for q, a in toxic:
        out.append(rec(i, "toxic", q, a))
        i += 1

    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    print(f"wrote {len(out)} synthetic records to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
