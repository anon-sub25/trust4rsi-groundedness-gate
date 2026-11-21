"""Memory-age axis: session accumulation plus real retrieval (Sections 5.2, 5.4).

This replaces the legacy "how many distractors" ladder with "how much history
has the agent accumulated". The retrieval budget k is FIXED as memory grows, so
the scorer's input length stays roughly constant and any drift with memory age
comes from *what* is retrieved, not *how much*.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.testbed.raw_store import FRACTIONS, load_stores

REPO = Path(__file__).resolve().parents[1]
PROCESSED = REPO / "data" / "processed"
MEMORY = REPO / "memory"

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

K_PRIMARY = 10
K_SECONDARY = 30

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercased, whitespace-and-punctuation tokenized (Section 5.2)."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    name = "bm25"

    def __init__(self, units: list[dict]):
        from rank_bm25 import BM25Okapi

        self.units = units
        self.bm25 = BM25Okapi([tokenize(u["text"]) for u in units])

    def top_k(self, query: str, k: int) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:k]
        return [self.units[i] for i in order]


class DenseRetriever:
    name = "dense"

    def __init__(self, units: list[dict], embedder):
        self.units = units
        self.embedder = embedder
        self.matrix = embedder.encode([u["text"] for u in units])

    def top_k(self, query: str, k: int) -> list[dict]:
        q = self.embedder.encode([query])[0]
        sims = self.matrix @ q
        order = np.argsort(-sims)[:k]
        return [self.units[i] for i in order]


class Embedder:
    """Pinned sentence embedder, normalized so a dot product is cosine."""

    def __init__(self, model_id: str = DENSE_MODEL, revision: str = DENSE_REVISION):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.revision = revision
        self.model = SentenceTransformer(model_id, revision=revision)

        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        """Normalized embeddings, memoized by text.

        The grid revisits the same turns and questions at every memory age, so
        without the cache the same string is embedded dozens of times.
        """
        missing = [t for t in texts if t not in self._cache]
        if missing:
            vecs = self.model.encode(
                missing, normalize_embeddings=True, show_progress_bar=False
            )
            for t, v in zip(missing, vecs):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts]) if texts else np.zeros(
            (0, self.model.get_sentence_embedding_dimension())
        )

    def one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


@dataclass
class Diagnostics:
    """Section 5.4, logged per item x t x retriever x k."""

    support_retrieved: bool
    related_retrieved: float | None
    # max_sim_to_answer needs a candidate answer, which is per-cell rather than
    # per-item, so it is filled during scoring and left None in this pass.
    max_sim_to_answer: float | None
    mean_sim_to_question: float
    n_tokens_evidence: int
    n_retrieved: int


def diagnose(
    item: dict,
    retrieved: list[dict],
    embedder: Embedder,
    candidate_answer: str | None,
) -> Diagnostics:
    retrieved_ids = {u["unit_id"] for u in retrieved}
    evidence = set(item.get("evidence_units") or [])

    if item["answerable"]:
        support = bool(evidence & retrieved_ids)
        related = None
    else:
        # For adversarial items the evidence list is the set of highly topical
        # related turns, not support (Section 3.1).
        support = False
        related = (len(evidence & retrieved_ids) / len(evidence)) if evidence else None

    texts = [u["text"] for u in retrieved]
    vecs = embedder.encode(texts)
    q = embedder.one(item["question"])
    mean_sim_q = float(np.mean(vecs @ q)) if len(vecs) else 0.0

    max_sim_a = None
    if candidate_answer and len(vecs):
        a = embedder.one(candidate_answer)
        max_sim_a = float(np.max(vecs @ a))

    return Diagnostics(
        support_retrieved=support,
        related_retrieved=related,
        max_sim_to_answer=max_sim_a,
        mean_sim_to_question=mean_sim_q,
        n_tokens_evidence=sum(u["n_tokens"] for u in retrieved),
        n_retrieved=len(retrieved),
    )


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[K_PRIMARY, K_SECONDARY])
    ap.add_argument("--retrievers", nargs="+", default=["bm25", "dense"])
    ap.add_argument("--out", type=Path, default=MEMORY / "substrate1_retrieval.jsonl")
    args = ap.parse_args()

    stores = load_stores(PROCESSED / "locomo_units.jsonl")
    qa = read_jsonl(PROCESSED / "locomo_qa.jsonl")
    qa = [q for q in qa if q["excluded"] is None]

    embedder = Embedder()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with args.out.open("w") as fh:
        for conv_id, store in sorted(stores.items()):
            items = [q for q in qa if q["conv_id"] == conv_id]
            for f in FRACTIONS:
                t = store.age_to_session(f)
                units = store.at_age(t)
                built = {}
                for rname in args.retrievers:
                    built[rname] = (
                        BM25Retriever(units)
                        if rname == "bm25"
                        else DenseRetriever(units, embedder)
                    )
                for item in items:
                    for rname, retr in built.items():
                        ks = args.k if rname == "bm25" else [K_PRIMARY]
                        for k in ks:
                            got = retr.top_k(item["question"], k)
                            d = diagnose(item, got, embedder, None)
                            fh.write(
                                json.dumps(
                                    {
                                        "conv_id": conv_id,
                                        "item_id": item["item_id"],
                                        "answerable": item["answerable"],
                                        "category": item["category"],
                                        "f": f,
                                        "t": t,
                                        "n_sessions": store.n_sessions,
                                        "t_min": item["t_min"],
                                        "retriever": rname,
                                        "k": k,
                                        "retrieved_units": [
                                            u["unit_id"] for u in got
                                        ],
                                        **d.__dict__,
                                    }
                                )
                                + "\n"
                            )
                            n += 1
            print(f"  {conv_id}: done ({n} rows so far)", flush=True)

    print(f"wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
