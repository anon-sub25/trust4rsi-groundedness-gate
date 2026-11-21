#!/usr/bin/env python3
"""Generate paper/appendix_tables.tex from results/. Stdlib only.

The appendix does not count against the page limit, so the full grid, the
saturation audit and the provenance of every run go here in full.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.sdt import criterion, d_prime, rate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MODELS = ["granite8b", "mistral7b", "phi4mini", "qwen3b", "qwen7b"]
NICE = {"granite8b": "Granite-3.3-8B", "mistral7b": "Mistral-7B",
        "phi4mini": "Phi-4-mini", "qwen3b": "Qwen2.5-3B", "qwen7b": "Qwen2.5-7B"}
JUDGES = ["J2a", "J2b", "J2c"]
FR = [0.1, 0.25, 0.5, 0.75, 1.0]


def grid(m, j):
    out = collections.defaultdict(list)
    for line in (RES / f"grid_{m}_{j}.jsonl").open():
        if line.strip():
            r = json.loads(line)
            out[(r["f"], r["cell"])].append(r)
    return out


def main() -> int:
    L = []
    A = L.append
    A(r"\section{Full grid}")
    A(r"\begin{table}[h]\centering\small")
    A(r"\caption{Accepts out of 40 per memory level $f$, bm25 $k{=}10$. "
      r"C is the honest refusal (correct verdict accept), $\mathrm{D_u}$ the "
      r"unsupported claim (correct verdict reject). $\dagger$ marks a cell "
      r"pinned at a floor or ceiling at one or both endpoints; $d'$ there "
      r"describes a boundary, not a discrimination.}")
    A(r"\begin{tabular}{llrrrrrrr}\toprule")
    A(r"model & gate & cell & " + " & ".join(f"${f}$" for f in FR) +
      r" & $\bar{d'}$ & $\bar{c}$ \\\midrule")
    for m in MODELS:
        for j in JUDGES:
            g = grid(m, j)
            dps, crs = [], []
            for f in FR:
                ch, cn = sum(r["accept"] for r in g[(f, "C")]), len(g[(f, "C")])
                dh, dn = sum(r["accept"] for r in g[(f, "D_u")]), len(g[(f, "D_u")])
                t, fa = rate(ch, cn), rate(dh, dn)
                dps.append(d_prime(t, fa))
                crs.append(criterion(t, fa))
            for cell in ("C", "D_u"):
                vals = [sum(r["accept"] for r in g[(f, cell)]) for f in FR]
                n = len(g[(FR[0], cell)])
                mark = r"$\dagger$" if (vals[0] in (0, n) or vals[-1] in (0, n)) else ""
                head = f"{NICE[m]} & {j} & " if cell == "C" else " & & "
                tail = (f" & ${sum(dps)/len(dps):+.2f}$ & ${sum(crs)/len(crs):+.2f}$"
                        if cell == "C" else " & & ")
                label = "C" if cell == "C" else r"$\mathrm{D_u}$"
                A(head + label + mark + " & " +
                  " & ".join(str(v) for v in vals) + tail + r" \\")
        A(r"\midrule")
    L[-1] = r"\bottomrule"
    A(r"\end{tabular}\end{table}")

    # Provenance.
    A(r"\section{Provenance}")
    seen = {}
    for m in MODELS:
        for j in JUDGES:
            for line in (RES / f"grid_{m}_{j}.jsonl").open():
                if line.strip():
                    r = json.loads(line)
                    seen[(r["model"], r["judge"])] = (r["revision"], r["prompt_sha256"],
                                                      r["dtype"], r["verdict_split"])
                    break
    A(r"\begin{table}[h]\centering\footnotesize")
    A(r"\caption{Pinned model revisions, prompt hashes and verdict splits. "
      r"Every run in this paper is reproducible from these.}")
    A(r"\begin{tabular}{llll}\toprule")
    A(r"model & gate & revision & prompt sha256 \\\midrule")
    for (model, judge), (rev, sha, _dt, _vs) in sorted(seen.items()):
        A(f"\\texttt{{{model.replace('_', '-')}}} & {judge} & "
          f"\\texttt{{{(rev or 'unpinned')[:12]}}} & \\texttt{{{sha[:16]}}} \\\\")
    A(r"\bottomrule\end{tabular}\end{table}")

    # Loop, whatever arms exist.
    loop = RES / "loop_qwen3b.jsonl"
    if loop.exists():
        rows = [json.loads(l) for l in loop.open() if l.strip()]
        by = collections.defaultdict(list)
        for r in rows:
            by[(r["arm"], r["round"], r["cell"])].append(r)
        full = max(len(v) for v in by.values())
        A(r"\section{Closed loop, per round}")
        A(r"\begin{table}[h]\centering\small")
        A(r"\caption{Per-round rates by arm. Only rounds with a complete row "
          r"count are shown. ``inf.\ retrieved'' is the mean number of "
          r"gate-written units among the ten retrieved.}")
        A(r"\begin{tabular}{llrrrr}\toprule")
        A(r"arm & round & claim acc. & refusal acc. & inf.\ retrieved & written \\\midrule")
        for arm in sorted({r["arm"] for r in rows}):
            for rd in sorted({r["round"] for r in rows}):
                du, c = by[(arm, rd, "D_u")], by[(arm, rd, "C")]
                if len(du) != full or len(c) != full:
                    continue
                A(f"{arm} & {rd} & {sum(r['accept'] for r in du)/len(du):.3f} & "
                  f"{sum(r['accept'] for r in c)/len(c):.3f} & "
                  f"{sum(r['n_inference_retrieved'] for r in du)/len(du):.3f} & "
                  f"{sum(r['inference_units'] for r in du)/len(du):.1f} \\\\")
            A(r"\midrule")
        L[-1] = r"\bottomrule"
        A(r"\end{tabular}\end{table}")

    out = ROOT / "paper" / "appendix_tables.tex"
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
