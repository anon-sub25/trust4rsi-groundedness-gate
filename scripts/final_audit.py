#!/usr/bin/env python3
"""Final number audit: every number in the paper traced to a data file."""
from __future__ import annotations
import json, sys, collections
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def load(name):
    p = REPO / "results" / name
    return [json.loads(l) for l in p.open() if l.strip()] if p.exists() else []

def by(rows, *keys):
    d = collections.defaultdict(list)
    for r in rows:
        d[tuple(r[k] for k in keys)].append(r)
    return d

def rate(rows):
    return sum(int(r["accept"]) for r in rows) / len(rows) if rows else 0

def pers(rows, cell="D_u"):
    b = by(rows, "round", "cell")
    full = max((len(v) for v in b.values()), default=0)
    rnds = sorted(rd for (rd, cl) in b if cl == cell and len(b[(rd, cl)]) == full)
    if 0 not in rnds or len(rnds) < 2:
        return None
    first = {r["item_id"]: int(r["accept"]) for r in b[(0, cell)]}
    acc0 = [i for i, a in first.items() if a]
    if not acc0:
        return None
    last = rnds[-1]
    m = {r["item_id"]: int(r["accept"]) for r in b[(last, cell)]}
    return sum(m[i] for i in acc0 if i in m) / len(acc0), last, len(acc0)

print("=== FINAL AUDIT ===\n")

# Original four-arm loop
orig = load("loop_qwen3b.jsonl")
if orig:
    print(f"loop_qwen3b.jsonl: {len(orig)} rows")
    for arm in sorted({r["arm"] for r in orig}):
        sub = [r for r in orig if r["arm"] == arm]
        du = [r for r in sub if r["cell"] == "D_u"]
        c = [r for r in sub if r["cell"] == "C"]
        rds = by(du, "round")
        print(f"  arm {arm}: {len(sub)} rows, rounds {sorted(rds)}")
        for rd in sorted(rds):
            print(f"    r{rd}: claim-acc {rate(rds[rd]):.3f}  refusal-acc {rate(by(c, 'round')[rd]):.3f}")
        p = pers(sub)
        if p:
            print(f"    persistence at r{p[1]}: {p[0]:.3f} (n={p[2]})")

# B2
b2 = load("loop_qwen3b_b2.jsonl")
if b2:
    print(f"\nloop_qwen3b_b2.jsonl: {len(b2)} rows")
    du = [r for r in b2 if r["cell"] == "D_u"]
    c = [r for r in b2 if r["cell"] == "C"]
    rds_du = by(du, "round")
    rds_c = by(c, "round")
    for rd in sorted(rds_du):
        inr = sum(float(r["n_inference_retrieved"]) for r in rds_du[rd] + rds_c[rd]) / (len(rds_du[rd]) + len(rds_c[rd]))
        print(f"  r{rd}: claim-acc {rate(rds_du[rd]):.3f}  refusal-acc {rate(rds_c[rd]):.3f}  mean_inf_retr {inr:.2f}")
    p = pers(b2)
    if p:
        print(f"  persistence at r{p[1]}: {p[0]:.3f} (n={p[2]})")

# f-ramp
fr = load("loop_qwen3b_framp.jsonl")
if fr:
    print(f"\nloop_qwen3b_framp.jsonl: {len(fr)} rows")
    for arm in sorted({r["arm"] for r in fr}):
        sub = [r for r in fr if r["arm"] == arm]
        du = [r for r in sub if r["cell"] == "D_u"]
        rds = by(du, "round")
        print(f"  arm {arm}:")
        for rd in sorted(rds):
            f_val = rds[rd][0].get("f", "?")
            print(f"    r{rd} (f={f_val}): claim-acc {rate(rds[rd]):.3f}")
        p = pers(sub)
        if p:
            print(f"    persistence at r{p[1]}: {p[0]:.3f} (n={p[2]})")

print("\n=== AUDIT COMPLETE ===")
