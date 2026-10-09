"""Generate data/sample/synthetic_noise.jsonl: labelled fake "bad" records + clean controls.

Real instruction datasets are fairly clean, so a labelled synthetic slice makes every stage of
the pipeline observable in the eval report (precision/recall). All PII below is fake
(example.com, 555 numbers, documentation IP ranges). Each record carries `noise_type` so tests
and the eval helper can assert the right stage caught it.

noise_type values:
  exact_dup_after_normalize, near_dup, pii, non_english, repetitive, too_short,
  boilerplate, symbol_noise, toxic  — expected to be caught by the mapped stage
  clean                             — hard-negative / control; must survive every stage
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
    # Need enough long records for exact + near duplicates (10 + 12).
    picks = rng.sample(long_ones, 24)
    out: list[dict] = []
    i = 0

    # exact duplicates hidden behind whitespace / unicode / case differences
    for b in picks[:10]:
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
    for b in picks[10:22]:
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
        (
            "Where do I mail the refund form?",
            "Mail it to refunds@example.com. For status updates text 555-222-0199.",
        ),
        (
            "List the lab machines",
            "Primary is 203.0.113.9 and the spare is 203.0.113.10 on the documentation net.",
        ),
        (
            "Confirm the payment method",
            "We charged 5500 0000 0000 0004 last Tuesday; receipt went to finance@example.net.",
        ),
        (
            "Who is the account owner?",
            "Owner SSN on file is 219-09-9999. Contact owner@example.org if that looks wrong.",
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
        (
            "¿Cómo se prepara una paella valenciana tradicional paso a paso?",
            "Primero se sofríe el pollo y el conejo, luego se añade el arroz y el caldo hasta "
            "que quede en su punto, con azafrán y judías verdes.",
        ),
        (
            "Warum ist die Berliner Mauer historisch so bedeutend für Europa?",
            "Die Mauer teilte die Stadt und wurde zum Symbol des Kalten Krieges, bis sie 1989 "
            "friedlich fiel und die Wiedervereinigung ermöglichte.",
        ),
        (
            "Comment fonctionne le métro de Paris pour un voyageur débutant ?",
            "Achetez un ticket, validez-le aux tourniquets, et suivez les lignes numérotées "
            "jusqu'à votre correspondance ou votre station d'arrivée.",
        ),
        (
            "Quali sono i piatti tipici della cucina italiana più conosciuti?",
            "Tra i più famosi ci sono la pizza napoletana, la pasta al pesto, il risotto "
            "alla milanese e il tiramisù come dessert classico.",
        ),
    ]
    for q, a in non_english:
        out.append(rec(i, "non_english", q, a))
        i += 1

    repetitive = [
        ("Write an ad", "Buy now! " * 40),
        ("List fruits", "\n".join(["apple banana apple banana"] * 25)),
        ("Describe the product", " ".join(["best price best quality best service"] * 15)),
        ("Repeat the slogan", "Amazing deal! " * 35),
        ("Fill the page", "\n".join(["lorem line lorem line"] * 30)),
    ]
    for q, a in repetitive:
        out.append(rec(i, "repetitive", q, a))
        i += 1

    too_short = [
        ("Hi", "ok"),
        ("?", ""),
        ("x", "y"),
        ("Go", "no"),
    ]
    for q, a in too_short:
        out.append(rec(i, "too_short", q, a))
        i += 1

    boilerplate = [
        (
            "Summarize the page",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor "
            "incididunt ut labore et dolore magna aliqua. Lorem ipsum dolor sit amet.",
        ),
        (
            "Extract the main idea",
            "Please enable javascript to continue browsing this site. Cookie policy applies "
            "to all visitors and must be accepted before content loads.",
        ),
        (
            "What does the article say?",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut enim ad minim veniam, "
            "quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
        ),
    ]
    for q, a in boilerplate:
        out.append(rec(i, "boilerplate", q, a))
        i += 1

    symbol_noise = [
        (
            "Parse this",
            "#### $$$ @@@ ### %%% ^^^ &&& *** ((( ))) ### $$$ @@@ ### %%% ^^^ &&& *** {{{ }}} "
            ">>> <<< ### ||| ~~~ ### $$$ ok",
        ),
        (
            "Decode the blob",
            "!!! ??? *** ### $$$ %%% ^^^ &&& *** ((( ))) [[[ ]]] {{{ }}} ||| ~~~ ``` ::: ;;; "
            "@@@ ### $$$ %%% ^^^ &&& *** end",
        ),
        (
            "Clean the tokens",
            "@@@ ### $$$ %%% ^^^ &&& *** ((( ))) ### $$$ @@@ ### %%% ^^^ &&& *** {{{ }}} >>> "
            "<<< ### ||| ~~~ ### $$$ ###",
        ),
    ]
    for q, a in symbol_noise:
        out.append(rec(i, "symbol_noise", q, a))
        i += 1

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
        (
            "Rate this submission",
            "This is trash from a dumb idiot. What a moron. Delete this garbage immediately.",
        ),
        (
            "Moderate the thread",
            "Ban that loser. He is a stupid idiot and a pathetic troll. Nobody wants him here.",
        ),
        (
            "Give honest feedback",
            "You sound like an idiot. Only a moron would write this garbage. Shut up already.",
        ),
    ]
    for q, a in toxic:
        out.append(rec(i, "toxic", q, a))
        i += 1

    # Clean controls / hard negatives: should survive every removal stage and not trip PII.
    # Mild opinions, short-but-valid Q&A, numeric content, and a single mild swear below the
    # toxicity lexicon threshold (max_hits=2) so precision has labelled negatives.
    clean = [
        (
            "What is the capital of France?",
            "Paris is the capital of France. It is known for museums, cuisine, and landmarks "
            "such as the Eiffel Tower and the Louvre.",
        ),
        (
            "Explain photosynthesis in simple terms.",
            "Plants convert light energy into chemical energy. Chlorophyll absorbs sunlight, "
            "and the plant uses water and carbon dioxide to make sugars and oxygen.",
        ),
        (
            "List three primenumbers between 10 and 20.",
            "11, 13, and 17 are prime. Each has no positive divisors other than one and itself.",
        ),
        (
            "How do I reset a forgotten password?",
            "Use the account recovery link on the sign-in page, verify your email, then choose "
            "a new password that meets the length and complexity rules.",
        ),
        (
            "Summarize why bees matter to agriculture.",
            "Bees pollinate many food crops. Without them, yields of fruits, nuts, and some "
            "vegetables would drop sharply, raising prices and reducing food diversity.",
        ),
        (
            "What does HTTP status 404 mean?",
            "It means the server could not find the requested resource. Check the URL spelling "
            "or whether the page was moved or deleted.",
        ),
        (
            # Near-miss: one mild lexicon hit should still be under max_hits=2.
            "Is the movie worth watching?",
            "It is pretty damn good if you like slow mysteries. The cast is strong and the "
            "ending pays off without feeling cheap or confusing.",
        ),
        (
            # Near-miss: symbols present but below max_symbol_word_ratio.
            "Format a short checklist",
            "1) Unpack the kit. 2) Charge the battery. 3) Pair the device. 4) Run the self-test. "
            "5) Log the serial number in the inventory sheet.",
        ),
    ]
    for q, a in clean:
        out.append(rec(i, "clean", q, a))
        i += 1

    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    from collections import Counter

    counts = Counter(r["noise_type"] for r in out)
    print(f"wrote {len(out)} synthetic records to {OUT.relative_to(ROOT)}")
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
