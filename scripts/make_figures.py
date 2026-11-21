#!/usr/bin/env python3
"""Figures for the Trust4RSI-Agent submission.

    fig1.pdf  false acceptance of unsupported claims against memory level
    fig2.pdf  the placebo control: J2a / J2c / J2b honest-refusal acceptance
    fig3.pdf  the latch: per-item persistence and re-entry across loop rounds

Everything is read from results/, so a figure cannot silently disagree with
the prose. Usage:  ../.venv/bin/python3 scripts/make_figures.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# matplotlib stamps Creator/Producer -- and on some builds the OS account name
# as Author -- into every PDF. Reviewers' tooling indexes PDF metadata, so an
# un-stripped figure is a double-blind leak that survives every visual check.
BLIND = {"Creator": "", "Producer": "", "Author": "", "Title": "",
         "Subject": "", "Keywords": "", "CreationDate": None}

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]
MODELS = ["granite8b", "mistral7b", "qwen3b", "qwen7b", "phi4mini"]
NICE = {"granite8b": "Granite-8B", "mistral7b": "Mistral-7B", "qwen3b": "Qwen-3B",
        "qwen7b": "Qwen-7B", "phi4mini": "Phi-4-mini"}
COLOR = {"granite8b": "#1b6ca8", "mistral7b": "#c4461f", "qwen3b": "#2e7d32",
         "qwen7b": "#7b4fa3", "phi4mini": "#999999"}

plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150})


def grid(model: str, judge: str) -> dict[tuple[float, str], list[dict]]:
    out = collections.defaultdict(list)
    with (RES / f"grid_{model}_{judge}.jsonl").open() as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[(r["f"], r["cell"])].append(r)
    return out


def rate(rows: list[dict]) -> float:
    return sum(r["accept"] for r in rows) / len(rows)


def save(fig, name: str) -> None:
    path = FIG / name
    fig.savefig(path, bbox_inches="tight", metadata=BLIND)
    plt.close(fig)
    print(f"wrote {path}")


def fig1() -> None:
    """False acceptance rises with memory level under both gates."""
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3), sharey=True)
    for ax, judge in zip(axes, ["J2a", "J2b"]):
        for m in MODELS:
            g = grid(m, judge)
            ys = [rate(g[(f, "D_u")]) for f in FRACTIONS]
            # Phi-4-mini is degenerate: it accepts almost everything on both
            # cells, so neither can move. Excluded from every claim; drawn
            # faded so the figure shows what was excluded rather than hiding it.
            sat = m == "phi4mini"
            ax.plot(FRACTIONS, ys, marker="o", ms=3, lw=1.3, color=COLOR[m],
                    ls=":" if sat else "-", alpha=0.45 if sat else 1.0,
                    label=NICE[m] + (" (sat.)" if sat else ""))
        ax.set_title(f"{judge} gate", fontsize=8)
        ax.set_xlabel("memory level $f$")
        ax.set_xticks(FRACTIONS)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("unsupported claims accepted")
    axes[1].legend(frameon=False, fontsize=6.5, loc="upper left")
    save(fig, "fig1.pdf")


def fig2() -> None:
    """The placebo control, as acceptance of the honest refusal."""
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    order = [m for m in MODELS if m != "phi4mini"] + ["phi4mini"]
    width, judges = 0.26, ["J2a", "J2c", "J2b"]
    hatch = {"J2a": "", "J2c": "///", "J2b": ""}
    face = {"J2a": "#bdbdbd", "J2c": "#ffffff", "J2b": "#1b6ca8"}
    for j, judge in enumerate(judges):
        vals = [sum(rate(grid(m, judge)[(f, "C")]) for f in FRACTIONS) / len(FRACTIONS)
                for m in order]
        ax.bar([i + (j - 1) * width for i in range(len(order))], vals, width,
               label=judge, facecolor=face[judge], edgecolor="#333", lw=0.6,
               hatch=hatch[judge])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([NICE[m] for m in order], rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("honest refusals accepted")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
    ax.text(len(order) - 1, 1.02, "saturated", fontsize=6, ha="center", color="#777")
    save(fig, "fig2.pdf")


def fig3() -> None:
    """The latch: what gets in stays in; what is out stays out."""
    path = RES / "loop_qwen3b.jsonl"
    if not path.exists():
        print("no loop results -- skipping fig3", file=sys.stderr)
        return
    rows = [json.loads(l) for l in path.open() if l.strip()]
    by = collections.defaultdict(list)
    for r in rows:
        if r["arm"] == "A" and r["cell"] == "D_u":
            by[r["round"]].append(r)
    full = max(len(v) for v in by.values())
    rnds = sorted(rd for rd, v in by.items() if len(v) == full)
    first = {r["item_id"]: r["accept"] for r in by[0]}
    acc0 = [i for i, a in first.items() if a]
    rej0 = [i for i, a in first.items() if not a]
    pers, reen, agg = [], [], []
    for rd in rnds:
        m = {r["item_id"]: r["accept"] for r in by[rd]}
        pers.append(sum(m[i] for i in acc0 if i in m) / len(acc0))
        reen.append(sum(m[i] for i in rej0 if i in m) / len(rej0))
        agg.append(rate(by[rd]))
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    ax.plot(rnds, pers, marker="o", ms=4, lw=1.5, color="#c4461f",
            label=f"accepted at round 0 ($n$={len(acc0)})")
    ax.plot(rnds, reen, marker="s", ms=4, lw=1.5, color="#1b6ca8",
            label=f"rejected at round 0 ($n$={len(rej0)})")
    ax.plot(rnds, agg, marker="^", ms=4, lw=1.2, ls="--", color="#555",
            label="all items (aggregate)")
    ax.set_xlabel("loop round")
    ax.set_ylabel("unsupported claim accepted")
    ax.set_xticks(rnds)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=6.5, loc="center right")
    save(fig, "fig3.pdf")


ARM_LABEL = {
    "A": "A: none",
    "B2": "B2: J2b deconf.",
    "C": "C: labelled",
    "D": "D: separated",
    "R": "R: rollback",
}
ARM_COLOR = {"A": "#c4461f", "B2": "#7b3f9e", "C": "#d98a2b", "D": "#1b6ca8",
             "R": "#2e7d32"}


def fig4() -> None:
    """Latch persistence at fixed f: arms A, B2, C, R."""
    path = RES / "loop_qwen3b.jsonl"
    if not path.exists():
        print("no loop results -- skipping fig4", file=sys.stderr)
        return
    rows = [json.loads(l) for l in path.open() if l.strip()]
    b2_path = RES / "loop_qwen3b_b2.jsonl"
    if b2_path.exists():
        for line in b2_path.open():
            if line.strip():
                r = json.loads(line)
                r["arm"] = "B2"
                rows.append(r)
    r_path = RES / "loop_qwen3b_rollback.jsonl"
    if r_path.exists():
        for line in r_path.open():
            if line.strip():
                rows.append(json.loads(line))
    by = collections.defaultdict(list)
    for r in rows:
        if r["cell"] == "D_u":
            by[(r["arm"], r["round"])].append(r)
    if not by:
        return
    full = max(len(v) for v in by.values())
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    for arm in ("A", "B2", "C", "R"):
        rnds = sorted(rd for (a, rd), v in by.items()
                      if a == arm and len(v) == full)
        if 0 not in rnds:
            print(f"fig4: arm {arm} has no complete round 0 -- skipped",
                  file=sys.stderr)
            continue
        first = {r["item_id"]: r["accept"] for r in by[(arm, 0)]}
        acc0 = [i for i, a in first.items() if a]
        if not acc0:
            continue
        pers = []
        for rd in rnds:
            m = {r["item_id"]: r["accept"] for r in by[(arm, rd)]}
            pers.append(sum(m[i] for i in acc0 if i in m) / len(acc0))
        ax.plot(rnds, pers, marker="o", ms=4, lw=1.5, color=ARM_COLOR[arm],
                label=f"{ARM_LABEL[arm]} ($n$={len(acc0)})")
    ax.set_xlabel("loop round")
    ax.set_ylabel("round-0 accepts still accepted")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")
    save(fig, "fig4.pdf")


def fig5() -> None:
    """F-ramp claim-acceptance: A vs D under growing memory."""
    path = RES / "loop_qwen3b_framp.jsonl"
    if not path.exists():
        print("no f-ramp results -- skipping fig5", file=sys.stderr)
        return
    rows = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    by = collections.defaultdict(list)
    for r in rows:
        if r["cell"] == "D_u":
            by[(r["arm"], r["round"])].append(r)
    if not by:
        return
    fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    for arm in ("A", "D"):
        cas = []
        for rnd in range(5):
            claims = by[(arm, rnd)]
            if not claims:
                break
            cas.append(sum(r["accept"] for r in claims) / len(claims))
        ax.plot(fracs[:len(cas)], cas, marker="o", ms=4, lw=1.5,
                color=ARM_COLOR[arm],
                label=ARM_LABEL[arm])
    ax.set_xlabel("memory fraction $f$")
    ax.set_ylabel("claim acceptance rate")
    ax.set_ylim(-0.02, 0.45)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    save(fig, "fig5.pdf")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
