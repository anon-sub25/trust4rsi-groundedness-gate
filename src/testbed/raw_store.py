"""Substrate 1: raw-turn memory (Section 5.2).

The agent's history stored verbatim. Memory at age `t` is every turn from
sessions 1..t of that conversation; nothing is summarized or rewritten, so this
is the simple baseline against which Substrate 2 is the headline design.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Memory age is indexed by fraction, because conversations differ in length
# (LoCoMo ranges 19-32 sessions).
FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)


@dataclass
class RawTurnStore:
    conv_id: str
    units: list[dict]  # chronological

    @property
    def n_sessions(self) -> int:
        return max(u["session"] for u in self.units)

    def age_to_session(self, f: float) -> int:
        """t = max(1, ceil(f * S_c))."""
        return max(1, math.ceil(f * self.n_sessions))

    def at_age(self, t: int) -> list[dict]:
        """Every turn the agent has accumulated by the end of session t."""
        return [u for u in self.units if u["session"] <= t]


def load_stores(units_path: Path) -> dict[str, RawTurnStore]:
    by_conv: dict[str, list[dict]] = defaultdict(list)
    with units_path.open() as fh:
        for line in fh:
            if line.strip():
                u = json.loads(line)
                by_conv[u["conv_id"]].append(u)
    return {
        cid: RawTurnStore(
            cid, sorted(us, key=lambda u: (u["session"], u["turn"]))
        )
        for cid, us in by_conv.items()
    }
