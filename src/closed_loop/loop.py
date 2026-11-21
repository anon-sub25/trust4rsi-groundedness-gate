"""Closed-loop memory evolution under a stochastic write gate.

An agent answers questions against its memory. A judge acts as the **write
policy**: an answer it accepts is committed to the store and becomes part of
what later questions retrieve. Topic (1) of the CFP names exactly this --
"provenance and write policies ... hallucination propagation".

The gate measured in the static grid is biased: under the default groundedness
prompt (J2a) it admits a confident unsupported claim more readily than an
honest refusal (365/1000 vs 209/1000 across five models). That asymmetry is a
selection pressure on the contents of memory. This module runs the selection.

Four arms, chosen to separate a fix *by disposition* from a fix *by
construction* (the CFP's own distinction):

  A  gate=J2a, provenance=none      naive deployment
  B  gate=J2b, provenance=none      fix the prompt (disposition)
  C  gate=J2a, provenance=marked    label inferences in the evidence (disposition)
  D  gate=J2a, provenance=isolated  observation-inference separation (construction)

Arm D keeps written inferences in the store and retrievable, but only
*observation* units are admissible as grounding evidence: an inference may not
ground another inference. That is the circularity the CFP's
"observation-inference separation" exists to break.

Ground truth never moves. A claim does not become true by being accepted, so
the correct verdict for a D_u stimulus is `reject` in every round.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEGACY = REPO.parent / "neuripsv2"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(LEGACY))

from src.judges import prompt_files  # noqa: E402
from src.judges.local_runner import (  # noqa: E402
    MODEL_REVISIONS,
    VERDICT_SPLITS,
    pick_device,
    score_batch,
    tokenizer_kwargs,
)
from src.memory_axis import BM25Retriever  # noqa: E402
from src.render import render_units  # noqa: E402
from src.testbed.raw_store import load_stores  # noqa: E402

STIMULI = REPO / "stimuli" / "locomo_cells.jsonl"
UNITS = REPO / "data" / "processed" / "locomo_units.jsonl"

ARMS = {
    "A": {"gate": "J2a", "provenance": "none"},
    "B": {"gate": "J2b", "provenance": "none"},
    "C": {"gate": "J2a", "provenance": "marked"},
    "D": {"gate": "J2a", "provenance": "isolated"},
    "R": {"gate": "J2a", "provenance": "none"},
}

MARK = "[unverified inference] "

#: How an accepted refusal is recorded. "echo" restates the question, which is
#: what arms A-D were run with; it makes the unit a near-copy of the BM25 query
#: that retrieves it, so it wins its own slot on string overlap rather than on
#: anything the gate did (docs/arm-b-confound.md). "record" states the same fact
#: without the question text, which removes that path.
ABSTENTION = {
    "echo": lambda q, rnd: f"No information is available about: {q}",
    "record": lambda q, rnd: (
        f"In round {rnd} the assistant was asked a question it could not answer "
        f"from memory, and said so."
    ),
}


def read_jsonl(path: Path):
    with path.open() as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_items(limit: int | None, seed: int):
    """One record per LoCoMo question carrying both candidate answers.

    C and D_u are built from the same question and see byte-identical memory;
    only the candidate answer differs. Keeping them on one record preserves
    that pairing through the loop.
    """
    by_item: dict[str, dict] = {}
    for stim in read_jsonl(STIMULI):
        if stim["cell"] not in ("C", "D_u"):
            continue
        rec = by_item.setdefault(
            stim["item_id"],
            {
                "item_id": stim["item_id"],
                "conv_id": stim["conv_id"],
                "question": stim["question"],
            },
        )
        rec["refusal" if stim["cell"] == "C" else "claim"] = stim["candidate_answer"]
    items = [r for r in by_item.values() if "refusal" in r and "claim" in r]
    items.sort(key=lambda r: r["item_id"])
    if limit:
        import random
        from collections import defaultdict

        by_conv = defaultdict(list)
        for r in items:
            by_conv[r["conv_id"]].append(r)
        for conv in by_conv:
            random.Random(f"{seed}:{conv}").shuffle(by_conv[conv])
        picked, idx = [], 0
        convs = sorted(by_conv)
        while len(picked) < limit:
            added = False
            for conv in convs:
                if idx < len(by_conv[conv]):
                    picked.append(by_conv[conv][idx])
                    added = True
                    if len(picked) == limit:
                        break
            if not added:
                break
            idx += 1
        items = sorted(picked, key=lambda r: r["item_id"])
    return items


def make_unit(arm: str, rnd: int, item: dict, kind: str, text: str, session: int) -> dict:
    """A unit written by the gate. `origin` is the provenance field."""
    body = MARK + text if ARMS[arm]["provenance"] == "marked" else text
    return {
        "conv_id": item["conv_id"],
        "unit_id": f"INF:{arm}:r{rnd}:{item['item_id']}:{kind}",
        "session": session,
        "turn": 10_000 + rnd,
        "speaker": "assistant",
        "date": f"written in round {rnd}",
        "text": body,
        "n_tokens": len(body.split()),
        "origin": "inference",
        "kind": kind,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--arms", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("-f", "--fraction", type=str, default="0.25",
                        help="single value or comma-separated per-round ramp")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument(
        "--abstention",
        choices=sorted(ABSTENTION),
        default="echo",
        help="How an accepted refusal is written. Default 'echo' reproduces the "
             "arms A-D runs. 'record' drops the question text, removing the "
             "retrieval artefact documented in docs/arm-b-confound.md.",
    )
    ap.add_argument(
        "--rollback",
        action="store_true",
        help="Re-evaluate all written claims each round and evict those the "
             "gate no longer accepts. Used by arm R.",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Keep arms already complete in --out and run only the rest. "
             "An arm is complete only if every round has the full row count; "
             "a partly-run arm is discarded and redone from round 0, because "
             "its store state cannot be reconstructed mid-arm.",
    )
    ap.add_argument("--out", default=str(REPO / "results" / "loop.jsonl"))
    args = ap.parse_args()

    fracs = [float(x) for x in args.fraction.split(",")]
    if len(fracs) == 1:
        fracs = fracs * args.rounds  # single value → constant
    elif len(fracs) != args.rounds:
        ap.error(f"--fraction has {len(fracs)} values but --rounds is {args.rounds}")

    items = load_items(args.limit, args.seed)
    stores = load_stores(UNITS)

    def build_base(f: float) -> tuple[dict[str, list[dict]], dict[str, int]]:
        b: dict[str, list[dict]] = {}
        se: dict[str, int] = {}
        for conv, store in stores.items():
            t = store.age_to_session(f)
            se[conv] = t
            b[conv] = [dict(u, origin="observation") for u in store.at_age(t)]
        return b, se

    base, sess = build_base(fracs[0])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done_arms: set[str] = set()
    kept: list[dict] = []
    if args.resume and out.exists():
        per: collections.Counter = collections.Counter()
        by_arm: dict[str, list[dict]] = collections.defaultdict(list)
        for r in read_jsonl(out):
            per[(r["arm"], r["round"])] += 1
            by_arm[r["arm"]].append(r)
        need = 2 * len(items)
        for arm in sorted(by_arm):
            if all(per[(arm, rd)] == need for rd in range(args.rounds)):
                done_arms.add(arm)
                kept.extend(by_arm[arm])
            else:
                have = {rd: per[(arm, rd)] for rd in range(args.rounds) if per[(arm, rd)]}
                print(f"[resume] arm {arm} partial {have} -- discarded, will redo", flush=True)
        print(f"[resume] complete arms {sorted(done_arms)} kept ({len(kept)} rows)", flush=True)

    prompts = {name: prompt_files.load(name) for name in {ARMS[a]["gate"] for a in args.arms}}
    print(f"abstention format: {args.abstention}", flush=True)
    print(
        f"model {args.model} | arms {args.arms} | {args.rounds} rounds | "
        f"{len(items)} items | f={','.join(str(x) for x in fracs)} k={args.k} | "
        f"{len(items) * 2 * args.rounds * len([a for a in args.arms if a not in done_arms])}"
        " scorings to run",
        flush=True,
    )
    for name, p in sorted(prompts.items()):
        print(f"  gate {name}: sha {p.sha256[:12]}", flush=True)

    if args.dry_run:
        conv = items[0]["conv_id"]
        retr = BM25Retriever(base[conv])
        got = retr.top_k(items[0]["question"], args.k)
        p = prompts[ARMS[args.arms[0]]["gate"]]
        print("-" * 70)
        print(p.render(question=items[0]["question"],
                       evidence=render_units(got),
                       answer=items[0]["claim"])[:1500])
        return 0

    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    from judgegeom.local_judge import verdict_from_logits  # noqa: PLC0415
    from src.judges import contrast  # noqa: PLC0415

    device = pick_device(args.device)
    revision = MODEL_REVISIONS.get(args.model)
    if revision is None:
        print(f"[revision] {args.model} unpinned -- probe only", flush=True)
    tok = AutoTokenizer.from_pretrained(
        args.model, revision=revision, **tokenizer_kwargs(args.model)
    )
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, dtype=dtype
    ).to(device).eval()
    split = VERDICT_SPLITS.get(args.model)
    print(f"device {device} | dtype {args.dtype} | verdict split {split}", flush=True)

    ids = {}
    for name, p in prompts.items():
        ids[name] = contrast.resolve_first_token_contrast(
            lambda s: tok.encode(s, add_special_tokens=False), p.positive, p.negative
        )

    n = 0
    with out.open("w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")
        n += len(kept)
        for arm in args.arms:
            if arm in done_arms:
                print(f"### ARM {arm} SKIPPED (complete in --out)", flush=True)
                continue
            gate = ARMS[arm]["gate"]
            prov = ARMS[arm]["provenance"]
            prompt = prompts[gate]
            id_pos, id_neg = ids[gate]
            base, sess = build_base(fracs[0])
            written: dict[str, list[dict]] = {c: [] for c in base}

            for rnd in range(args.rounds):
                # Update observation base if ramping f across rounds.
                if rnd > 0 and fracs[rnd] != fracs[rnd - 1]:
                    base, sess = build_base(fracs[rnd])
                # Store as it stands at the start of this round.
                store_now = {c: base[c] + written[c] for c in base}
                # Evidence path. Arm D admits only observations as grounding
                # evidence; the inferences stay in the store but may not ground
                # a later claim.
                ev_units = (
                    {c: [u for u in store_now[c] if u["origin"] == "observation"] for c in base}
                    if prov == "isolated"
                    else store_now
                )
                retr = {c: BM25Retriever(ev_units[c]) for c in ev_units if ev_units[c]}

                # Rollback: re-evaluate written claims and evict failures.
                evicted = 0
                if args.rollback and rnd > 0:
                    items_by_id = {it["item_id"]: it for it in items}
                    for conv in list(written):
                        surviving = []
                        claims_to_check = [u for u in written[conv] if u["kind"] == "claim"]
                        keep = [u for u in written[conv] if u["kind"] != "claim"]
                        for u in claims_to_check:
                            iid = u["unit_id"].split(":")[3]
                            it = items_by_id.get(iid)
                            if it is None:
                                surviving.append(u)
                                continue
                            got = retr[conv].top_k(it["question"], args.k) if conv in retr else []
                            ev = render_units(got)
                            user = prompt.render(
                                question=it["question"], evidence=ev, answer=it["claim"]
                            )
                            text = tok.apply_chat_template(
                                [{"role": "system", "content": prompt.system},
                                 {"role": "user", "content": user}],
                                tokenize=False, add_generation_prompt=True,
                            )
                            pairs = score_batch(model, tok, [text], id_pos, id_neg, device)
                            v = verdict_from_logits(pairs[0][0], pairs[0][1])
                            if v["says_correct"]:
                                surviving.append(u)
                            else:
                                evicted += 1
                        written[conv] = keep + surviving
                    if evicted:
                        store_now = {c: base[c] + written[c] for c in base}
                        ev_units = store_now
                        retr = {c: BM25Retriever(ev_units[c]) for c in ev_units if ev_units[c]}
                    print(f"  rollback r{rnd}: evicted {evicted} claims", flush=True)

                rendered = []
                for it in items:
                    got = retr[it["conv_id"]].top_k(it["question"], args.k)
                    ev = render_units(got)
                    n_inf = sum(1 for u in got if u["origin"] == "inference")
                    for cell, ans in (("C", it["refusal"]), ("D_u", it["claim"])):
                        user = prompt.render(
                            question=it["question"], evidence=ev, answer=ans
                        )
                        text = tok.apply_chat_template(
                            [{"role": "system", "content": prompt.system},
                             {"role": "user", "content": user}],
                            tokenize=False, add_generation_prompt=True,
                        )
                        rendered.append((it, cell, text, n_inf, len(got)))

                accepts: dict[tuple[str, str], int] = {}
                for i in range(0, len(rendered), args.batch_size):
                    chunk = rendered[i : i + args.batch_size]
                    pairs = score_batch(
                        model, tok, [c[2] for c in chunk], id_pos, id_neg, device
                    )
                    for (it, cell, _t, n_inf, n_got), (lp, ln) in zip(chunk, pairs):
                        v = verdict_from_logits(lp, ln)
                        accepts[(it["item_id"], cell)] = v["says_correct"]
                        fh.write(json.dumps({
                            "arm": arm, "gate": gate, "provenance": prov,
                            "round": rnd, "item_id": it["item_id"],
                            "conv_id": it["conv_id"], "cell": cell,
                            "correct_verdict": "accept" if cell == "C" else "reject",
                            "accept": v["says_correct"], "margin": v["margin"],
                            "n_inference_retrieved": n_inf, "n_retrieved": n_got,
                            "store_units": len(store_now[it["conv_id"]]),
                            "inference_units": len(written[it["conv_id"]]),
                            "model": args.model, "revision": revision,
                            "prompt_sha256": prompt.sha256,
                            "f": fracs[rnd], "k": args.k,
                            "evicted": evicted,
                        }) + "\n")
                        n += 1
                    if i % (args.batch_size * 100) == 0:
                        print(f"  arm {arm} r{rnd} {i}/{len(rendered)}", flush=True)

                # Apply the write policy: what the gate accepted enters memory.
                for it in items:
                    if accepts.get((it["item_id"], "D_u")):
                        written[it["conv_id"]].append(
                            make_unit(arm, rnd, it, "claim", it["claim"], sess[it["conv_id"]])
                        )
                    if accepts.get((it["item_id"], "C")):
                        written[it["conv_id"]].append(
                            make_unit(
                                arm, rnd, it, "abstention",
                                ABSTENTION[args.abstention](it["question"], rnd),
                                sess[it["conv_id"]],
                            )
                        )
                fa = sum(accepts[(i["item_id"], "D_u")] for i in items) / len(items)
                ra = sum(accepts[(i["item_id"], "C")] for i in items) / len(items)
                tot = sum(len(v) for v in written.values())
                clm = sum(1 for v in written.values() for u in v if u["kind"] == "claim")
                print(
                    f"### arm {arm} round {rnd}: claim-accept {fa:.3f}  refusal-accept {ra:.3f}"
                    f"  written {tot} ({clm} claims, {tot - clm} abstentions)",
                    flush=True,
                )
            print(f"### ARM {arm} COMPLETE", flush=True)
    print(f"wrote {n} rows -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
