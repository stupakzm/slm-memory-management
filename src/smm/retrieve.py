"""The phase 2 retrieval pipeline: hybrid candidates -> rerank -> gate.

    dense top-N  ┐
                 ├─ reciprocal rank fusion ─→ top-50 ─→ cross-encoder ─→ top-5 ─→ gate
    BM25  top-N  ┘

Fusion is by rank, not by score. Dense returns cosine similarity and BM25 returns
an Okapi score on an unbounded negative scale; there is no principled way to add
them, and every attempt to normalise one onto the other bakes in a corpus-specific
constant. RRF only needs the orderings, which is also why it survives the reranker
being the thing that actually decides.

The gate is deliberately the last step and deliberately dumb: one threshold on the
reranker's top-1 score. Finding 04's point is that the model must never be invoked
on thin evidence, and a mechanism the model participates in is not that.
"""

from __future__ import annotations

import sqlite3

from . import lexical, store

RRF_K = 60          # standard constant; damps the influence of any single list's top
CANDIDATES = 50     # what the reranker sees, per the briefing
KEEP = 5            # what the model sees


def rrf(lists: list[list[dict]], k: int = RRF_K, weights: list[float] | None = None) -> list[dict]:
    """Reciprocal rank fusion over ranked lists keyed by chunk_id."""
    weights = weights or [1.0] * len(lists)
    agg: dict[str, dict] = {}
    for lst, w in zip(lists, weights):
        for rank, c in enumerate(lst, 1):
            cur = agg.setdefault(c["chunk_id"], dict(c, rrf=0.0))
            cur["rrf"] += w / (k + rank)
            # keep whichever component scores we have, for diagnostics
            for f in ("score", "bm25"):
                if f in c and f not in cur:
                    cur[f] = c[f]
    out = list(agg.values())
    out.sort(key=lambda c: -c["rrf"])
    return out


class Retriever:
    """Composable so the eval can score each layer against the same questions.

    `mode` selects which candidate generator runs: dense only reproduces phase 1,
    bm25 only isolates the lexical contribution, hybrid fuses them. `rerank=False`
    measures fusion without the cross-encoder. Every phase 2 number comes from
    turning exactly one of these off.
    """

    def __init__(self, db: sqlite3.Connection, embedder=None, reranker=None,
                 mode: str = "hybrid", candidates: int = CANDIDATES,
                 dense_weight: float = 1.0, bm25_weight: float = 1.0):
        self.db = db
        self.embedder = embedder
        self.reranker = reranker
        self.mode = mode
        self.candidates = candidates
        self.weights = (dense_weight, bm25_weight)

    def candidates_for(self, question: str, n: int | None = None) -> list[dict]:
        n = n or self.candidates
        if self.mode == "dense":
            return store.search(self.db, self.embedder.embed_query(question), k=n)
        if self.mode == "bm25":
            return lexical.search(self.db, question, k=n)
        dense = store.search(self.db, self.embedder.embed_query(question), k=n)
        sparse = lexical.search(self.db, question, k=n)
        return rrf([dense, sparse], weights=list(self.weights))[:n]

    def retrieve(self, question: str, k: int = KEEP) -> list[dict]:
        cands = self.candidates_for(question)
        if self.reranker is None:
            return cands[:k]
        return self.reranker.rerank(question, cands, top_k=k)


def expand(db, hits: list[dict], span: int = 1, budget: int = 6000) -> list[dict]:
    """Widen each hit to its neighbours in the same section, preserving hit order.

    Ranking is untouched: hit i stays at position i and only its text grows. A chunk
    already absorbed into an earlier hit's expansion is not repeated, so the model
    never reads the same paragraph twice under two numbers.
    """
    if not hits or not hits[0].get("sec_id"):
        return hits                      # a flat index has no sections to expand into
    out, used, total = [], set(), 0
    for h in hits:
        block = store.neighbours(db, h["doc_id"], h["sec_id"], h["ord"], span)
        keep = [c for c in block if c["chunk_id"] not in used] or [h]
        text = "\n\n".join(c["text"] for c in keep)
        if total + len(text) > budget and out:
            text = h["text"]             # out of budget: fall back to the chunk itself
            keep = [h]
        used.update(c["chunk_id"] for c in keep)
        total += len(text)
        out.append(dict(h, text=text, expanded=len(keep)))
    return out


def gate_score(hits: list[dict]) -> float:
    """The number the gate thresholds: top-1 reranker score, or dense if unreranked."""
    if not hits:
        return float("-inf")
    return hits[0].get("rerank_score", hits[0].get("score", 0.0))
