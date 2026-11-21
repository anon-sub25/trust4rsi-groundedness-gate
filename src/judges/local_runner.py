"""Score stimuli with a local open-weight judge: verdict and logit margin.

One forward pass per stimulus, no generation. The quantity we actually want is
the margin `logit(positive) - logit(negative)` at the first response position,
because the verdict alone is a threshold on it and thresholds floor. The legacy
NeurIPS runs established that a cell can sit at 0/11 accepts while its margin
still moves tens of logits; with no paid API this year, the margin is the only
decision variable we can measure at all.

Deliberately reuses `neuripsv2/judgegeom`: `verdict_from_logits` for the
tie-breaking rule and `judgegeom.hardware` for the memory gate. There must not
be a second copy of either -- a divergence in the tie rule or a missing gate is
silent, and the gate is what keeps this machine out of swap.

Run from the repo root:

    PYTHONPATH=. ../.venv/bin/python3 -m src.judges.local_runner \
        --judge J2a --cells C D_u --fractions 0.1 1.0 --limit 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEGACY = REPO.parent / "neuripsv2"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(LEGACY))

from src.judges import contrast, prompt_files  # noqa: E402
from src.render import render_units  # noqa: E402
from src.testbed.raw_store import load_stores  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
#: Pinned commit for every model we report. An unpinned checkpoint can move
#: under us between runs and the margins would shift with nothing recording why.
MODEL_REVISIONS = {
    "Qwen/Qwen2.5-3B-Instruct": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "Qwen/Qwen2.5-14B-Instruct": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
    "Qwen/Qwen2.5-32B-Instruct": "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
    "mistralai/Mistral-7B-Instruct-v0.3": "c170c708c41dac9275d15a8fff4eca08d52bab71",
    "mistralai/Mistral-Nemo-Instruct-2407": "04d8a90549d23fc6bd7f642064003592df51e9b3",
    "mistralai/Mistral-Small-24B-Instruct-2501": "9527884be6e5616bdd54de542f9ae13384489724",
    "microsoft/phi-4": "2db69c1c3e91a05d2c64a3185acfbaf36f744e25",
    "microsoft/Phi-4-mini-instruct": "cfbefacb99257ffa30c83adab238a50856ac3083",
    "ibm-granite/granite-3.3-8b-instruct": "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b",
    "allenai/OLMo-2-1124-13B-Instruct": "3a5c85baefbb1896a54d56fe2e76c0395627ddf4",
}

#: How each family splits the two verdict words. Recorded because it decides
#: whether a margin is a clean log-odds between the permitted outputs or a
#: first-token proxy carrying prefix mass from unrelated continuations.
#:
#: Granite is the only one that makes CORRECT and WRONG whole tokens, so its
#: margin needs no proxy at all -- which makes it the reference model for any
#: claim stated in margins.
VERDICT_SPLITS = {
    "ibm-granite/granite-3.3-8b-instruct": ("CORRECT", "WRONG"),
    "Qwen/Qwen2.5-3B-Instruct": ("COR|RECT", "WR|ONG"),
    "Qwen/Qwen2.5-7B-Instruct": ("COR|RECT", "WR|ONG"),
    "Qwen/Qwen2.5-14B-Instruct": ("COR|RECT", "WR|ONG"),
    "Qwen/Qwen2.5-32B-Instruct": ("COR|RECT", "WR|ONG"),
    "microsoft/phi-4": ("COR|RECT", "WR|ONG"),
    "microsoft/Phi-4-mini-instruct": ("COR|RECT", "WR|ONG"),
    "allenai/OLMo-2-1124-13B-Instruct": ("COR|RECT", "WR|ONG"),
    "mistralai/Mistral-7B-Instruct-v0.3": ("C|OR|RECT", "WR|ONG"),
    "mistralai/Mistral-Nemo-Instruct-2407": ("C|OR|RECT", "WR|ONG"),
    "mistralai/Mistral-Small-24B-Instruct-2501": ("C|OR|RECT", "WR|ONG"),
}


def tokenizer_kwargs(model: str) -> dict:
    """Per-family tokenizer fixes.

    Mistral's HF tokenizers ship a regex that transformers reports as wrong and
    fixes only behind an opt-in flag. It does not move the verdict tokens
    (checked: CORRECT/WRONG tokenize identically either way), but it does change
    how the evidence block is split, and the evidence is most of the prompt.
    Taking the fix means the prompt the model sees matches the prompt we wrote.
    """
    if model.startswith("mistralai/"):
        return {"fix_mistral_regex": True}
    return {}

STIMULI = REPO / "stimuli" / "locomo_cells.jsonl"
RETRIEVAL = REPO / "memory" / "substrate1_retrieval.jsonl"
UNITS = REPO / "data" / "processed" / "locomo_units.jsonl"
RESULTS = REPO / "results"


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_jsonl(path: Path):
    with path.open() as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def build_work(cells, fractions, retriever, k, limit, seed):
    """Join stimuli to retrieval rows: one (stimulus, f) pair per scored item.

    Retrieval is keyed on the *item*, not the stimulus, because every cell built
    from one LoCoMo question sees exactly the same memory; only the candidate
    answer differs. That is the whole point of the cell design, so the join has
    to preserve it -- C and D_u for one item must get byte-identical evidence.
    """
    wanted = {float(f) for f in fractions}
    retrieval: dict[tuple[str, float], dict] = {}
    for row in read_jsonl(RETRIEVAL):
        if row["retriever"] != retriever or row["k"] != k:
            continue
        if float(row["f"]) not in wanted:
            continue
        retrieval[(row["item_id"], float(row["f"]))] = row

    by_cell: dict[str, list[dict]] = {c: [] for c in cells}
    for stim in read_jsonl(STIMULI):
        if stim["cell"] in by_cell:
            by_cell[stim["cell"]].append(stim)
    for cell in cells:
        by_cell[cell].sort(key=lambda s: s["stimulus_id"])
        if limit:
            by_cell[cell] = _stratified(by_cell[cell], limit, seed)

    work = []
    for cell in cells:
        for stim in by_cell[cell]:
            for f in sorted(wanted):
                row = retrieval.get((stim["item_id"], f))
                if row is None:
                    continue
                work.append((stim, f, row))
    return work


def _stratified(stimuli: list[dict], limit: int, seed: int) -> list[dict]:
    """Take `limit` stimuli spread across conversations, not the first `limit`.

    Taking the head of a sorted list looks like a sample and is not: stimulus_ids
    sort by conversation, so `--limit 40` drew all forty items from conv-26 and
    the first probe measured one conversation. Round-robin over conversations,
    each conversation's own items shuffled under a fixed seed, so a small n still
    spans the corpus and the same n is reproducible.

    The cells stay aligned item-for-item: every cell is built from the same
    LoCoMo questions in the same order, so the same seed picks the same items in
    C and in D_u, which is what lets the two be compared pairwise.
    """
    import random
    from collections import defaultdict

    by_conv: dict[str, list[dict]] = defaultdict(list)
    for s in stimuli:
        by_conv[s["conv_id"]].append(s)
    for conv in by_conv:
        random.Random(f"{seed}:{conv}").shuffle(by_conv[conv])

    picked: list[dict] = []
    convs = sorted(by_conv)
    round_index = 0
    while len(picked) < limit:
        added = False
        for conv in convs:
            if round_index < len(by_conv[conv]):
                picked.append(by_conv[conv][round_index])
                added = True
                if len(picked) == limit:
                    break
        if not added:
            break
        round_index += 1
    return sorted(picked, key=lambda s: s["stimulus_id"])



def pick_device(requested: str | None = None) -> str:
    """cuda if there is one, else Apple mps, else cpu.

    Written so the same runner serves a Colab GPU and this laptop. The judge
    must be the same code in both places: a second scoring path is how the two
    would silently drift apart, and a drifted margin is unfalsifiable.
    """
    import torch

    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def score_batch(model, tokenizer, texts, id_positive, id_negative, device):
    """Logits at each sequence's final real token, for a batch of prompts.

    Padding side is load-bearing. The verdict is read at the last position, so
    with right padding that position holds PAD and every number would be noise
    -- a failure that produces a complete, plausible, meaningless results file.
    Left padding puts the real final token last for every row in the batch.
    """
    import torch

    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        batch = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**batch).logits[:, -1, :].float().cpu()
    finally:
        tokenizer.padding_side = previous_side
    return [
        (float(row[id_positive]), float(row[id_negative])) for row in logits
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge", default="J2a", choices=sorted(prompt_files.CONVENTIONS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cells", nargs="+", default=["C", "D_u"])
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.1, 1.0])
    parser.add_argument("--retriever", default="bm25")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument(
        "--limit", type=int, default=0,
        help="items per cell, sampled round-robin across conversations (0 = all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="build and render every prompt, print the longest, load no model",
    )
    parser.add_argument(
        "--seed", type=int, default=20261107,
        help="sampling seed; fixed so a --limit run is reproducible",
    )
    parser.add_argument(
        "--device", default="auto",
        help="auto | cuda | mps | cpu",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="1 on mps (unified memory); raise on a CUDA GPU",
    )
    parser.add_argument(
        "--dtype", default="bf16", choices=["bf16", "fp16"],
        help="bf16 everywhere unless the GPU cannot do it (Turing, e.g. a Colab "
             "T4). fp16 margins are NOT comparable with bf16 ones.",
    )
    parser.add_argument(
        "--load-4bit", action="store_true",
        help="NF4 quantization (bitsandbytes, CUDA only). The ONLY way a 32B "
             "fits on Colab Pro -- and it perturbs the logits the margin is "
             "made of, so such rows self-label and must not be pooled with "
             "full-precision ones.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    prompt = prompt_files.load(args.judge)
    stores = load_stores(UNITS)
    work = build_work(
        args.cells, args.fractions, args.retriever, args.k, args.limit, args.seed
    )
    print(
        f"judge {prompt.name} ({prompt.sha256[:12]}) | {len(work)} scorings "
        f"| cells {args.cells} | f {args.fractions} | {args.retriever} k={args.k}",
        flush=True,
    )

    rendered = []
    for stim, f, row in work:
        store = stores[stim["conv_id"]]
        by_id = {u["unit_id"]: u for u in store.units}
        units = [by_id[uid] for uid in row["retrieved_units"] if uid in by_id]
        user = prompt.render(
            question=stim["question"],
            evidence=render_units(units),
            answer=stim["candidate_answer"],
        )
        rendered.append((stim, f, row, user))

    if args.dry_run:
        longest = max(rendered, key=lambda r: len(r[3]))
        print(f"longest prompt: {len(longest[3])} chars "
              f"({longest[0]['stimulus_id']}, f={longest[1]})")
        print("-" * 70)
        print(prompt.system)
        print("-" * 70)
        print(longest[3])
        return 0

    import torch  # noqa: PLC0415 -- never imported in a dry run
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    from judgegeom.local_judge import verdict_from_logits  # noqa: PLC0415

    revision = MODEL_REVISIONS.get(args.model)
    if revision is None:
        print(
            f"[revision] {args.model} is not in MODEL_REVISIONS -- running "
            "unpinned. Fine for a probe, not for a reported number.",
            flush=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=revision, **tokenizer_kwargs(args.model)
    )
    id_positive, id_negative = contrast.resolve_first_token_contrast(
        lambda s: tokenizer.encode(s, add_special_tokens=False),
        prompt.positive,
        prompt.negative,
    )
    split = VERDICT_SPLITS.get(args.model)
    if split:
        print(f"verdict split: {split[0]} / {split[1]}", flush=True)
    print(
        f"verdict contrast: {prompt.positive}->{id_positive} "
        f"({tokenizer.decode([id_positive])!r}) "
        f"{prompt.negative}->{id_negative} ({tokenizer.decode([id_negative])!r})",
        flush=True,
    )

    chats = []
    for stim, f, row, user in rendered:
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": prompt.system},
             {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        chats.append((stim, f, row, text, len(tokenizer.encode(text))))
    max_tokens = max(c[4] for c in chats)
    print(f"max prompt length: {max_tokens} tokens", flush=True)

    device = pick_device(args.device)

    # The hardware gate guards *this laptop*: unified memory means a 15 GB model
    # swapping to SSD, not a clean OOM. A hosted CUDA box has neither problem and
    # no vm_stat to sample, so the gate is skipped there rather than faked.
    hardware = None
    if device != "cuda":
        from judgegeom import hardware  # noqa: PLC0415
        snapshot = hardware.sample()
        print(
            f"[pre-load] RAM avail {snapshot.free_ram_gb:.1f} GB | swap free "
            f"{snapshot.swap_free_gb:.1f} GB | thermal_ok {snapshot.thermal_ok}",
            flush=True,
        )
        if not snapshot.thermal_ok:
            print("[pre-load] thermal pressure reported -- stopping to protect "
                  "the machine.")
            return 1

    # Precision is part of the measurement, not a performance knob. The margin
    # is a difference of two raw logits, so bf16 and fp16 runs of the same model
    # give slightly different margins -- fine within a table, wrong across one.
    # Refuse bf16 on hardware that cannot do it rather than let torch silently
    # emulate it, and say loudly when fp16 is in use.
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    precision = args.dtype
    if dtype is torch.bfloat16 and device == "cuda" and not torch.cuda.is_bf16_supported():
        print(
            "[dtype] this GPU has no bf16 (Turing-class, e.g. a Colab T4). Rerun "
            "with --dtype fp16, and keep those margins in their own table: they "
            "are not comparable with the bf16 runs. An L4 or A100 avoids this.",
            flush=True,
        )
        return 1
    if dtype is torch.float16:
        print("[dtype] fp16 -- margins NOT comparable with bf16 runs.", flush=True)

    quantization_config = None
    if args.load_4bit:
        if device != "cuda":
            print("[dtype] --load-4bit needs CUDA (bitsandbytes).", flush=True)
            return 1
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        precision = f"nf4/{args.dtype}"
        print(
            "[dtype] NF4 4-bit. The margin is a difference of two raw logits and "
            "quantization moves them, so these rows are tagged "
            f"dtype={precision!r} and belong in their own table -- never pooled "
            "with full-precision runs.",
            flush=True,
        )

    print(f"loading model ({precision}, {device})...", flush=True)
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=revision,
        dtype=dtype,
        low_cpu_mem_usage=True,
        # device_map lets accelerate place the shards; with 4-bit the weights
        # must be placed at load time, not moved afterwards.
        **({"quantization_config": quantization_config, "device_map": "auto"}
           if quantization_config else {}),
    )
    if quantization_config is None:
        model.to(device)
    model.eval()
    print(f"loaded in {(time.monotonic() - started) / 60:.1f} min", flush=True)

    out_path = Path(args.out) if args.out else (
        RESULTS / f"local_{prompt.name}_{args.model.split('/')[-1]}_{stamp()}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing {out_path}", flush=True)

    batches = [
        chats[i : i + args.batch_size] for i in range(0, len(chats), args.batch_size)
    ]
    done = 0
    with out_path.open("w") as fh:
        for batch in batches:
            scored = score_batch(
                model, tokenizer, [c[3] for c in batch],
                id_positive, id_negative, device,
            )
            for (stim, f, row, text, n_tokens), (lp, ln) in zip(batch, scored):
                done += 1
                verdict = verdict_from_logits(lp, ln)
                record = {
                    "stimulus_id": stim["stimulus_id"],
                    "item_id": stim["item_id"],
                    "cell": stim["cell"],
                    "correct_verdict": stim["correct_verdict"],
                    "judge": prompt.name,
                    "prompt_sha256": prompt.sha256,
                    "model": args.model,
                    "revision": revision,
                    "verdict_split": "/".join(split) if split else None,
                    "dtype": precision,
                    "device": device,
                    "substrate": 1,
                    "f": f,
                    "t": row["t"],
                    "retriever": args.retriever,
                    "k": args.k,
                    "support_retrieved": row["support_retrieved"],
                    "n_tokens_prompt": n_tokens,
                    # `says_correct` is the judge accepting the candidate answer.
                    # For cells C and D_u, accept is correct for C and wrong for D_u.
                    "accept": verdict["says_correct"],
                    "margin": verdict["margin"],
                    "logit_positive": verdict["logit_correct"],
                    "logit_negative": verdict["logit_incorrect"],
                }
                fh.write(json.dumps(record) + "\n")
            fh.flush()
            if device == "cuda":
                # No thermal/unified-memory worry on a hosted GPU; the gate below
                # is this laptop's protection and would only add noise there.
                print(f"  {done}/{len(chats)}", flush=True)
                continue
            snap = hardware.sample()
            print(
                f"  {done}/{len(chats)} | RAM {snap.free_ram_gb:.1f} GB | "
                f"thermal_ok {snap.thermal_ok}",
                flush=True,
            )
            if not snap.thermal_ok:
                print("  thermal pressure -- stopping early, partial file kept.")
                break
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
