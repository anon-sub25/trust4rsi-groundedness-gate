"""Normalize LoCoMo into data/processed/ (Section 3.1).

Ground truth comes from dataset labels only; no LLM touches this file.
Token counts use a single pinned tokenizer (ground rule 5).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "locomo" / "locomo10.json"
PROCESSED = REPO / "data" / "processed"

# VERIFY (open question 5): this is the checkpoint cached from the NeurIPS runs.
# The "median Qwen tokens" axis must use the same one.
TOKENIZER_ID = "Qwen/Qwen2.5-3B-Instruct"
TOKENIZER_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"

ADVERSARIAL_CATEGORY = 5  # verified empirically: the 446-item category
DIA_RE = re.compile(r"^D(\d+):(\d+)$")


@dataclass
class Unit:
    conv_id: str
    unit_id: str
    session: int
    turn: int
    speaker: str
    date: str
    text: str
    n_tokens: int


@dataclass
class QA:
    conv_id: str
    item_id: str
    question: str
    category: int
    answerable: bool
    gold_answer: str | None
    adversarial_answer: str | None
    evidence_units: list[str]
    t_min: int | None
    excluded: str | None  # reason, or None


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        TOKENIZER_ID, revision=TOKENIZER_REVISION, local_files_only=True
    )


def session_index(unit_id: str) -> int | None:
    m = DIA_RE.match(unit_id)
    return int(m.group(1)) if m else None


def build_units(data, tok) -> list[Unit]:
    units: list[Unit] = []
    for conv in data:
        conv_id = conv["sample_id"]
        talk = conv["conversation"]
        sessions = sorted(
            (k for k in talk if k.startswith("session_") and not k.endswith("date_time")),
            key=lambda k: int(k.split("_")[1]),
        )
        for key in sessions:
            s = int(key.split("_")[1])
            date = talk.get(f"{key}_date_time", "")
            for i, turn in enumerate(talk[key], start=1):
                uid = turn["dia_id"]
                text = turn["text"]
                units.append(
                    Unit(
                        conv_id=conv_id,
                        unit_id=uid,
                        session=session_index(uid) or s,
                        turn=int(uid.split(":")[1]) if ":" in uid else i,
                        speaker=turn["speaker"],
                        date=date,
                        text=text,
                        n_tokens=len(tok.encode(text, add_special_tokens=False)),
                    )
                )
    return units


def build_qa(data) -> list[QA]:
    rows: list[QA] = []
    for conv in data:
        conv_id = conv["sample_id"]
        for i, q in enumerate(conv.get("qa", []), start=1):
            cat = q.get("category")
            gold = q.get("answer")
            gold = None if gold in (None, "") else str(gold)
            adv = q.get("adversarial_answer")
            evidence = list(q.get("evidence") or [])

            # Cells C/C_v/D_u require the item to be genuinely unanswerable.
            # Two category-5 items in conv-26 carry BOTH a real answer and an
            # adversarial_answer (yes/no questions), so "I don't know." is not a
            # valid refusal for them. Excluded from the unanswerable pool and
            # reported; cell C is therefore 444, not 446.
            excluded = None
            if cat == ADVERSARIAL_CATEGORY and gold is not None:
                excluded = "cat5_has_gold_answer"

            answerable = cat != ADVERSARIAL_CATEGORY
            sessions = [s for s in (session_index(u) for u in evidence) if s is not None]
            t_min = max(sessions) if (answerable and sessions) else None

            rows.append(
                QA(
                    conv_id=conv_id,
                    item_id=f"{conv_id}_q{i}",
                    question=q["question"],
                    category=cat,
                    answerable=answerable,
                    gold_answer=gold if answerable else None,
                    adversarial_answer=adv,
                    evidence_units=evidence,
                    t_min=t_min,
                    excluded=excluded,
                )
            )
    return rows


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r)) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=PROCESSED)
    args = ap.parse_args()

    data = json.loads(args.raw.read_text())
    tok = load_tokenizer()

    units = build_units(data, tok)
    qa = build_qa(data)

    write_jsonl(args.out / "locomo_units.jsonl", units)
    write_jsonl(args.out / "locomo_qa.jsonl", qa)

    unanswerable = [r for r in qa if not r.answerable]
    usable_c = [r for r in unanswerable if r.excluded is None]
    manifest = {
        "tokenizer": {"id": TOKENIZER_ID, "revision": TOKENIZER_REVISION},
        "n_conversations": len(data),
        "n_units": len(units),
        "n_qa": len(qa),
        "n_answerable": sum(r.answerable for r in qa),
        "n_unanswerable": len(unanswerable),
        "n_unanswerable_usable": len(usable_c),
        "excluded": {
            r.item_id: r.excluded for r in qa if r.excluded is not None
        },
        "sessions_per_conversation": {
            c["sample_id"]: len(
                [
                    k
                    for k in c["conversation"]
                    if k.startswith("session_") and not k.endswith("date_time")
                ]
            )
            for c in data
        },
        "category_counts": {
            str(cat): sum(r.category == cat for r in qa)
            for cat in sorted({r.category for r in qa})
        },
    }
    (args.out / "locomo_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
