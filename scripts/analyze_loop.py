#!/usr/bin/env python3
"""Closed-loop analysis: entry rate, latch persistence, retrieval capture.

Scores against docs/prereg-ratchet.md (predictions 1-4, retired) and
docs/prereg-ratchet-amendment.md (predictions 5-7, fixed before arms B/C/D ran).
Stdlib only -- safe to run while a model is resident.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARM_LABEL = {
    "A": "A  J2a, no provenance      (naive)",
    "B": "B  J2b, no provenance      (disposition: prompt)",
    "C": "C  J2a, marked inferences  (disposition: labels)",
    "D": "D  J2a, isolated evidence  (construction: obs/inf separation)",
}


def main(path: str) -> int:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    arms = sorted({r["arm"] for r in rows})
    rounds = sorted({r["round"] for r in rows})
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["arm"], r["round"], r["cell"])].append(r)

    # An arm is analysable only for the rounds that actually completed. A
    # partly-run round would otherwise read as a wave of rejections, because
    # its missing items look like absent accepts.
    full = max(len(v) for v in by.values())
    ok_rounds = {
        arm: [rd for rd in rounds
              if len(by[(arm, rd, "D_u")]) == full and len(by[(arm, rd, "C")]) == full]
        for arm in arms
    }
    for arm in arms:
        short = [rd for rd in rounds
                 if by[(arm, rd, "D_u")] and rd not in ok_rounds[arm]]
        if short:
            print(f"[partial] arm {arm} rounds {short} incomplete -- excluded\n")

    print("=== Per-round rates (claim acceptance is the false-accept rate; "
          "ground truth never moves)\n")
    for arm in arms:
        print(ARM_LABEL.get(arm, arm))
        print(f"  {'rnd':>3} {'claim-acc':>10} {'refusal-acc':>12} "
              f"{'mean n_inf_ret':>15} {'any inf':>9} {'mean writ':>10}")
        for rd in ok_rounds[arm]:
            du = by[(arm, rd, "D_u")]
            c = by[(arm, rd, "C")]
            ni = [r["n_inference_retrieved"] for r in du]
            print(f"  {rd:>3} {sum(r['accept'] for r in du)/len(du):10.3f} "
                  f"{sum(r['accept'] for r in c)/len(c):12.3f} "
                  f"{sum(ni)/len(ni):15.3f} {sum(1 for x in ni if x>0)/len(ni):9.3f} "
                  f"{sum(r['inference_units'] for r in du)//len(du):8d}")
        print()

    print("=== Latch persistence: of the claims accepted at round 0, "
          "how many are still accepted later?\n")
    persist = {}
    for arm in arms:
        rs = ok_rounds[arm]
        if 0 not in rs or len(rs) < 2:
            print(f"  {arm:>3}  needs a complete round 0 and at least one later "
                  f"round; has {rs} -- skipped")
            continue
        print(f"  {'arm':>3} {'n accepted r0':>14} "
              + " ".join(f"{'r'+str(rd):>7}" for rd in rs[1:]))
        first = {r["item_id"]: r["accept"] for r in by[(arm, 0, "D_u")]}
        acc0 = [i for i, a in first.items() if a]
        if not acc0:
            print(f"  {arm:>3} {0:>14}   (no claims admitted -- persistence undefined)")
            continue
        cells = []
        for rd in rs[1:]:
            m = {r["item_id"]: r["accept"] for r in by[(arm, rd, "D_u")]}
            cells.append(sum(m[i] for i in acc0 if i in m) / len(acc0))
        persist[arm] = dict(zip(rs[1:], cells))
        flag = "" if len(acc0) >= 20 else "  UNDERPOWERED (n<20)"
        print(f"  {arm:>3} {len(acc0):>14} " + " ".join(f"{v:7.3f}" for v in cells) + flag)

    print("\n=== Re-entry: of the claims rejected at round 0, how many are "
          "accepted later?\n")
    for arm in arms:
        rs = ok_rounds[arm]
        if 0 not in rs or len(rs) < 2:
            continue
        print(f"  {'arm':>3} {'n rejected r0':>14} "
              + " ".join(f"{'r'+str(rd):>7}" for rd in rs[1:]))
        first = {r["item_id"]: r["accept"] for r in by[(arm, 0, "D_u")]}
        rej0 = [i for i, a in first.items() if not a]
        if not rej0:
            continue
        cells = []
        for rd in rs[1:]:
            m = {r["item_id"]: r["accept"] for r in by[(arm, rd, "D_u")]}
            cells.append(sum(m[i] for i in rej0 if i in m) / len(rej0))
        print(f"  {arm:>3} {len(rej0):>14} " + " ".join(f"{v:7.3f}" for v in cells))

    if len(persist) > 1 and "A" in persist:
        print("\n=== Preregistered tests (amendment, predictions 5-7)\n")
        def at(arm):
            """Persistence for arm and A at the latest round complete in BOTH."""
            shared = sorted(set(persist[arm]) & set(persist["A"]))
            if not shared:
                return None, None, None
            rd = shared[-1]
            return persist[arm][rd], persist["A"][rd], rd
        print("  Each arm is compared with A at the latest round complete in both.")
        if "D" in persist:
            d, a, rd = at("D")
            ok = d is not None and d < 0.60
            print(f"  P5  arm D < 0.60 ....... D {d:.3f} at r{rd} (A {a:.3f})  "
                  f"{'HELD' if ok else 'FAILED'}")
            if not ok:
                print("      -> separation does not break the latch; the closed-loop")
                print("         framing is unnecessary and is reported as such.")
        if "C" in persist:
            c, a, rd = at("C")
            ok = c is not None and abs(c - a) <= 0.10
            print(f"  P6  arm C within 0.10 of A ... C {c:.3f} vs A {a:.3f} at r{rd}, "
                  f"|diff| {abs(c-a):.3f}  {'HELD' if ok else 'FAILED'}")
            if c is not None and c < a - 0.10:
                print("      -> marking works; 'construction beats disposition' WITHDRAWN.")
        if "B" in persist:
            b, a, rd = at("B")
            ok = b is not None and abs(b - a) <= 0.10
            print(f"  P7  arm B within 0.10 of A ... B {b:.3f} vs A {a:.3f} at r{rd}, "
                  f"|diff| {abs(b-a):.3f}  {'HELD' if ok else 'FAILED'}")
        print("\n  Entry rates at round 0 (P7's separate composition claim):")
        for arm in [x for x in arms if 0 in ok_rounds[x]]:
            du = by[(arm, 0, "D_u")]
            c = by[(arm, 0, "C")]
            print(f"    {arm}: claim {sum(r['accept'] for r in du)/len(du):.3f}  "
                  f"refusal {sum(r['accept'] for r in c)/len(c):.3f}")
    else:
        print("\n(Only arm A present -- the amendment's predictions 5-7 need B, C and D.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1
                          else str(REPO / "results" / "loop_qwen3b.jsonl")))
