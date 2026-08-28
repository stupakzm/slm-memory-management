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

# Two distinct per-variant knobs when fusing the question with rewrites - do not
# collapse them into one, and do not reuse the caller's output `k` for either.
# VARIANT_CANDIDATES is the dense/BM25 candidate pool handed to the reranker for
# each variant (input to the cross-encoder). VARIANT_K is how deep each variant's
# RERANKED list goes before it reaches rrf() (input to the fusion) - it is NOT
# the number of hits the caller ultimately wants (`k`/KEEP), which only truncates
# the fused output at the very end.
#
# Measured (tsk_20260828_a47f0496), query "how to list files via size", gold
# chunk ls.1:4:
#   per-variant reranked depth  5 (== k, the bug)  -> gold@4, two size.1 chunks
#                                                       still rank above it
#   per-variant reranked depth 20, 1 rewrite        -> gold@2
#   per-variant reranked depth 20, 3 rewrites       -> gold@1
# Fusing 5-deep lists starves rrf() of exactly the lower-ranked-but-right hits
# it exists to promote - by the time a list is truncated to 5, the variant that
# would have surfaced the gold chunk at rank 6-15 has already dropped it. 20 is
# the same measured depth as VARIANT_CANDIDATES's own justification (1 rewrite
# at 20 candidates/variant: 2.56s wall, vs 2.98s for today's single query at 50;
# 3 rewrites at pool 50 cost 11.6s for no further win) - a deeper reranked list
# needs a deeper candidate pool to draw from, which is why both default to 20.
VARIANT_CANDIDATES = 20    # candidate pool per variant, fed to the reranker
VARIANT_K = 20             # reranked-list depth per variant, fed to rrf()


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


def fuse_variants(retrieve_fn, variants: list[str], k: int = KEEP) -> tuple[list[dict], int]:
    """Fuse a question with its rewrites, each retrieved AND reranked separately.

    Measured and load-bearing (tsk_20260828_a47f0496): fusing the dense candidate
    lists first and reranking once barely moves gold (15 -> 13 on the failing
    query). The reranker has to run per variant, and it is the RERANKED lists that
    get fused here by rrf() - fusing anything earlier in the pipeline was tried and
    was not enough.

    `retrieve_fn` is any callable(query: str) -> list[dict], best-first, each dict
    carrying `chunk_id` - typically "retrieve this variant's candidates and rerank
    them", but the fusion itself neither knows nor cares how that list was built.
    This is what keeps it testable with no DB, no GPU, and no reranker: stub
    `retrieve_fn` with plain dicts and it runs under bare stdlib python3.

    `variants` is `[question, *rewrites]` by convention - index 0 is the original
    question - so the caller can name whichever one actually won.

    Returns `(fused[:k], winner)` where `winner` is the index into `variants` whose
    list ranked the top fused hit highest (ties favour the earlier variant, i.e.
    the original question over a rewrite).
    """
    lists = [retrieve_fn(v) for v in variants]
    fused = rrf(lists)
    if not fused:
        return [], 0
    top_id = fused[0]["chunk_id"]
    winner, best_rank = 0, None
    for i, lst in enumerate(lists):
        for rank, c in enumerate(lst, 1):
            if c["chunk_id"] == top_id:
                if best_rank is None or rank < best_rank:
                    winner, best_rank = i, rank
                break
    return fused[:k], winner


class Retriever:
    """Composable so the eval can score each layer against the same questions.

    `mode` selects which candidate generator runs: dense only reproduces phase 1,
    bm25 only isolates the lexical contribution, hybrid fuses them. `rerank=False`
    measures fusion without the cross-encoder. Every phase 2 number comes from
    turning exactly one of these off.
    """

    def __init__(self, db: sqlite3.Connection, embedder=None, reranker=None,
                 mode: str = "hybrid", candidates: int = CANDIDATES,
                 dense_weight: float = 1.0, bm25_weight: float = 1.0,
                 domain: str | None = None):
        self.db = db
        self.embedder = embedder
        self.reranker = reranker
        self.mode = mode
        self.candidates = candidates
        self.weights = (dense_weight, bm25_weight)
        # None means every domain competes, which is the thing phase 5 measures
        # rather than assumes.
        self.domain = domain

    def candidates_for(self, question: str, n: int | None = None) -> list[dict]:
        n = n or self.candidates
        if self.mode == "dense":
            return store.search(self.db, self.embedder.embed_query(question), k=n,
                                domain=self.domain)
        if self.mode == "bm25":
            return lexical.search(self.db, question, k=n)
        dense = store.search(self.db, self.embedder.embed_query(question), k=n,
                             domain=self.domain)
        sparse = lexical.search(self.db, question, k=n)
        return rrf([dense, sparse], weights=list(self.weights))[:n]

    def retrieve(self, question: str, k: int = KEEP) -> list[dict]:
        cands = self.candidates_for(question)
        if self.reranker is None:
            return cands[:k]
        return self.reranker.rerank(question, cands, top_k=k)

    def retrieve_fused(self, question: str, rewrites: list[str] | None = None,
                        k: int = KEEP, variant_candidates: int = VARIANT_CANDIDATES,
                        variant_k: int = VARIANT_K,
                        ) -> tuple[list[dict], int, list[str]]:
        """`retrieve()`, fused across the question and its rewrites.

        Each variant (index 0 is `question`, the rest are `rewrites`) is retrieved
        at `variant_candidates` and reranked down to `variant_k` on its own; the
        reranked lists are then combined with `fuse_variants` (see module docstring
        there for why per-variant reranking, not a single fused-then-reranked pool)
        and the fused result is truncated to `k`. `variant_k` is deliberately NOT
        `k`: rrf() needs the lower-ranked-but-right hits a shallow list would have
        already dropped - see the measured numbers on VARIANT_K above. Returns
        `(fused hits, winning variant index, variants)` so a caller can report which
        rewrite - or the original question - produced the top hit.
        """
        variants = [question] + list(rewrites or [])

        def rerank_variant(q: str) -> list[dict]:
            cands = self.candidates_for(q, n=variant_candidates)
            if self.reranker is None:
                return cands[:variant_k]
            return self.reranker.rerank(q, cands, top_k=variant_k)

        fused, winner = fuse_variants(rerank_variant, variants, k=k)
        return fused, winner, variants


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
