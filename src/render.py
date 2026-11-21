"""Render retrieved memory units into the text a scorer sees (Section 5.3).

Units are always ordered chronologically, never by retrieval score: ordering by
score would confound every comparison (ground rule 8). Unit IDs stay visible
because the absence check cites them.
"""

from __future__ import annotations

from typing import Iterable

TURN_LINE = "[Session {session} | {date}] {unit_id} {speaker}: {text}"
FACT_LINE = "[Session {session} | {date}] {fact_id} (from {sources}): {text}"


def _chronological(units: list[dict]) -> list[dict]:
    return sorted(units, key=lambda u: (u["session"], u.get("turn", 0), u["unit_id"]))


def render_units(units: Iterable[dict]) -> str:
    """Substrate 1: verbatim turns."""
    return "\n".join(
        TURN_LINE.format(
            session=u["session"],
            date=u.get("date", ""),
            unit_id=u["unit_id"],
            speaker=u.get("speaker", ""),
            text=u["text"],
        )
        for u in _chronological(list(units))
    )


def render_facts(facts: Iterable[dict]) -> str:
    """Substrate 2: extracted facts, with provenance links visible."""
    rows = sorted(
        list(facts), key=lambda f: (f["session"], f["fact_id"])
    )
    return "\n".join(
        FACT_LINE.format(
            session=f["session"],
            date=f.get("date", ""),
            fact_id=f["fact_id"],
            sources=", ".join(f.get("source_unit_ids", [])),
            text=f["text"],
        )
        for f in rows
    )
