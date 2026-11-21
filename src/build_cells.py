"""Build candidate answers per stimulus cell (Section 4).

Nothing here is LLM-generated: every candidate is dataset text or a fixed
template, and every cell carries a correct verdict fixed by construction.

Cells built here are the ones this spec defines as new or re-derivable:
  C, C_v  unanswerable + refusal (correct verdict: accept)
  D_u     unanswerable + adversarial_answer (reject)
  E, E_v  answerable + refusal (reject iff support_retrieved; see Section 4.2)

Cells A, B and D_c come from the legacy builder (Section 5.1) and are pending
open questions 3, 4 and 7; see memory/README.md.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROCESSED = REPO / "data" / "processed"
STIMULI = REPO / "stimuli"

SAMPLE_SEED = 20261103

IDK = "I don't know."

# Section 4.1. Deliberately mirror a self-justifying refusal without echoing the
# question, which would add topical content and confound the comparison.
VERBOSE_TEMPLATES = [
    "I've looked through our previous conversations and I can't find anything "
    "that answers this, so I don't know.",
    "Based on what I remember from our chats, there's no information about "
    "that. I'd rather not guess.",
    "I don't have any record of that in our past conversations, so I can't "
    "answer it reliably.",
]


def template_index(item_id: str) -> int:
    """Stable stand-in for the spec's `hash(item_id) % 3`.

    Python's builtin hash() is salted per process, so it would reassign
    templates on every run and break byte-identical prompts (ground rule 1).
    """
    digest = hashlib.sha256(item_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") % len(VERBOSE_TEMPLATES)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def stratified_sample(items: list[dict], n: int, seed: int) -> list[dict]:
    """Sample n items, allocating by category in proportion to the pool."""
    by_cat: dict[int, list[dict]] = collections.defaultdict(list)
    for it in items:
        by_cat[it["category"]].append(it)

    total = len(items)
    quotas: dict[int, int] = {}
    for cat, pool in by_cat.items():
        quotas[cat] = min(len(pool), int(n * len(pool) / total))

    # Hand out the rounding remainder to the largest categories first.
    short = n - sum(quotas.values())
    for cat in sorted(by_cat, key=lambda c: -len(by_cat[c])):
        if short <= 0:
            break
        room = len(by_cat[cat]) - quotas[cat]
        take = min(room, short)
        quotas[cat] += take
        short -= take

    rng = random.Random(seed)
    picked: list[dict] = []
    for cat in sorted(by_cat):
        pool = sorted(by_cat[cat], key=lambda it: it["item_id"])
        picked.extend(rng.sample(pool, quotas[cat]))
    return sorted(picked, key=lambda it: it["item_id"])


def stimulus(item: dict, cell: str, answer: str, verdict: str, **extra) -> dict:
    row = {
        "stimulus_id": f"{item['item_id']}::{cell}",
        "item_id": item["item_id"],
        "conv_id": item["conv_id"],
        "cell": cell,
        "question": item["question"],
        "category": item["category"],
        "candidate_answer": answer,
        "correct_verdict": verdict,
        "answerable": item["answerable"],
        "evidence_units": item["evidence_units"],
        "t_min": item["t_min"],
    }
    row.update(extra)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=PROCESSED)
    ap.add_argument("--out", type=Path, default=STIMULI)
    args = ap.parse_args()

    qa = read_jsonl(args.processed / "locomo_qa.jsonl")

    unanswerable = [r for r in qa if not r["answerable"] and r["excluded"] is None]
    answerable = [r for r in qa if r["answerable"] and r["evidence_units"]]

    # Cell E is sized to match cell C (Section 4.2). Cell C is 444, not the
    # spec's 446, because two category-5 items carry a gold answer.
    e_items = stratified_sample(answerable, len(unanswerable), SAMPLE_SEED)

    rows: list[dict] = []
    for it in unanswerable:
        tpl = template_index(it["item_id"])
        rows.append(stimulus(it, "C", IDK, "accept"))
        rows.append(
            stimulus(it, "C_v", VERBOSE_TEMPLATES[tpl], "accept", template=tpl)
        )
        adv = it["adversarial_answer"]
        if adv:
            rows.append(stimulus(it, "D_u", adv, "reject"))

    for it in e_items:
        tpl = template_index(it["item_id"])
        # Correct verdict is conditional: reject only where support was actually
        # retrieved. Resolved per (item, t, retriever, k) in Section 5.4.
        rows.append(
            stimulus(it, "E", IDK, "reject_if_support_retrieved")
        )
        rows.append(
            stimulus(
                it, "E_v", VERBOSE_TEMPLATES[tpl],
                "reject_if_support_retrieved", template=tpl,
            )
        )

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "locomo_cells.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    (args.out / "cell_E_items.json").write_text(
        json.dumps(
            {
                "seed": SAMPLE_SEED,
                "n": len(e_items),
                "item_ids": [it["item_id"] for it in e_items],
            },
            indent=2,
        )
    )

    counts = collections.Counter(r["cell"] for r in rows)
    tpl_counts = collections.Counter(
        r["template"] for r in rows if "template" in r
    )
    manifest = {
        "n_stimuli": len(rows),
        "cells": dict(sorted(counts.items())),
        "template_balance": dict(sorted(tpl_counts.items())),
        "cell_E_sample_seed": SAMPLE_SEED,
        "cell_E_categories": dict(
            sorted(collections.Counter(it["category"] for it in e_items).items())
        ),
        "notes": [
            "D_u is short of C by the number of unanswerable items with no "
            "adversarial_answer.",
            "E/E_v verdicts resolve to reject only where support_retrieved is "
            "true (Section 4.2).",
        ],
    }
    (args.out / "cells_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
