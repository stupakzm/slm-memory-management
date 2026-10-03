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

Phase 11 R8 (`question_vectors` > 0, docs/phase11-results.md "R8 pre-registration"):
the chunk pool is joined by the chunks whose generated-question vectors sit nearest
the query. That route only widens the pool - the union is still reranked against the
original question and the gate reads that top-1.
"""

from __future__ import annotations

import sqlite3
from collections import Counter

from . import lexical, store

RRF_K = 60          # standard constant; damps the influence of any single list's top
CANDIDATES = 50     # what the reranker sees, per the briefing
KEEP = 5            # what the model sees

# Phase 13 R9 (docs/phase13-results.md): the open search's first ROUTE_VOTE_K hits
# vote on the domain the real search then runs in. 5 is the reader's own k.
ROUTES = ("dense-vote", "quota", "floor")
ROUTE_VOTE_K = 5

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


def floor_pool(open_hits: list[dict], per_domain: dict[str, list[dict]], n: int,
               floor: int) -> list[dict]:
    """Phase 14 R11 floor quota (docs/phase14-results.md): every domain is guaranteed
    its own top `floor` chunks, and the rest of the n slots go to the best remaining
    chunks by dense distance, wherever they come from.

    `open_hits` is an open dense search, best first; `per_domain` maps each domain to
    its own filtered search, best first. The result is each domain's first `floor`
    chunks (domains in sorted name order), then the fill: the smallest `distance`
    (ties by chunk_id) across `open_hits` and every `per_domain` list, skipping any
    chunk already taken, until there are n or nothing is left. floor == 0 is the open
    top-n; with two domains and floor == n // 2 it is the R10 quota's chunk set.
    """
    if floor < 0:
        raise ValueError(f"floor must be >= 0, got {floor}")
    if len(per_domain) * floor > n:
        raise ValueError(f"{len(per_domain)} domains x floor {floor} exceeds {n} candidates")
    out: list[dict] = []
    seen: set[str] = set()
    for d in sorted(per_domain):
        for h in per_domain[d][:floor]:
            if h["chunk_id"] not in seen:
                seen.add(h["chunk_id"])
                out.append(h)
    rest: dict[str, dict] = {}
    for lst in [open_hits, *(per_domain[d] for d in sorted(per_domain))]:
        for h in lst:
            if h["chunk_id"] not in seen:
                rest.setdefault(h["chunk_id"], h)
    fill = sorted(rest.values(), key=lambda h: (h["distance"], h["chunk_id"]))
    out.extend(fill[: max(n - len(out), 0)])
    return out[:n]


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
                 domain: str | None = None, question_vectors: int = 0,
                 route: str | None = None, floor: int | None = None):
        if (route == "floor") != (floor is not None):
            raise ValueError("floor goes with route='floor', and route='floor' needs floor")
        if route is not None:
            # BM25 has no domain filter, so a hybrid route would be half-routed.
            if route not in ROUTES:
                raise ValueError(f"unknown route {route!r}; expected one of {ROUTES}")
            if domain is not None:
                raise ValueError("route and domain are exclusive: a route chooses the domain")
            if question_vectors > 0:
                raise ValueError("route does not compose with question_vectors")
            if mode != "dense":
                raise ValueError("route needs mode='dense' (BM25 has no domain filter)")
        self.db = db
        self.embedder = embedder
        self.reranker = reranker
        self.mode = mode
        self.candidates = candidates
        self.weights = (dense_weight, bm25_weight)
        # None means every domain competes, which is the thing phase 5 measures
        # rather than assumes.
        self.domain = domain
        # R8: how many nearest question vectors join the pool; 0 is today's path.
        self.question_vectors = question_vectors
        # R9: how candidates_for() picks its domain, and what it picked last.
        self.route = route
        self.last_route: str | None = None
        self._quota_domains: list[str] | None = None
        # R11: the guaranteed per-domain minimum, and the cached domain list.
        self.floor = floor
        self._floor_domains: list[str] | None = None

    def candidates_for(self, question: str, n: int | None = None) -> list[dict]:
        n = n or self.candidates
        if self.route == "quota":
            return self._quota(question, n)
        if self.route == "dense-vote":
            return self._routed(question, n)
        if self.route == "floor":
            return self._floor(question, n)
        if self.question_vectors > 0 and self.mode != "bm25":
            return self._with_question_route(question, n)
        if self.mode == "dense":
            return store.search(self.db, self.embedder.embed_query(question), k=n,
                                domain=self.domain)
        if self.mode == "bm25":
            return lexical.search(self.db, question, k=n)
        dense = store.search(self.db, self.embedder.embed_query(question), k=n,
                             domain=self.domain)
        sparse = lexical.search(self.db, question, k=n)
        return rrf([dense, sparse], weights=list(self.weights))[:n]

    def _quota(self, question: str, n: int) -> list[dict]:
        """R10 quota: no routing decision. One embedding; each domain (sorted by
        name) gets n // D candidates, the first n % D domains one more; the pools
        are concatenated in domain order for the reranker."""
        if self._quota_domains is None:
            self._quota_domains = sorted(store.domains(self.db))
        doms = self._quota_domains
        vec = self.embedder.embed_query(question)
        out: list[dict] = []
        for i, d in enumerate(doms):
            share = n // len(doms) + (1 if i < n % len(doms) else 0)
            if share > 0:
                out.extend(store.search(self.db, vec, k=share, domain=d))
        return out

    def _floor(self, question: str, n: int) -> list[dict]:
        """R11 floor quota: no routing decision. One embedding, the open search, and
        (when floor > 0) each domain's own top `floor`; floor_pool() assembles them."""
        vec = self.embedder.embed_query(question)
        open_hits = store.search(self.db, vec, k=n, domain=None)
        per_domain: dict[str, list[dict]] = {}
        if self.floor > 0:
            if self._floor_domains is None:
                self._floor_domains = sorted(store.domains(self.db))
            for d in self._floor_domains:
                per_domain[d] = store.search(self.db, vec, k=self.floor, domain=d)
        return floor_pool(open_hits, per_domain, n, self.floor)

    def _routed(self, question: str, n: int) -> list[dict]:
        """R9 dense-vote: one embedding, an open search, a majority vote over the
        domains of its top ROUTE_VOTE_K hits, then the real search inside the
        winner. A tie goes to the domain of the top-ranked hit."""
        vec = self.embedder.embed_query(question)
        open_hits = store.search(self.db, vec, k=n, domain=None)
        votes = Counter(h["domain"] for h in open_hits[:ROUTE_VOTE_K] if h.get("domain"))
        top = open_hits[0].get("domain") if open_hits else None
        ranked = votes.most_common(2)
        winner = (ranked[0][0] if ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1])
                  else top)
        self.last_route = winner
        if winner is None:
            return open_hits         # nothing to route on: the open pool stands
        return store.search(self.db, vec, k=n, domain=winner)

    def _with_question_route(self, question: str, n: int) -> list[dict]:
        """The chunk pool exactly as above, then the question route's new chunks,
        each dict tagged `route`. One query embedding serves both searches."""
        vec = self.embedder.embed_query(question)
        dense = store.search(self.db, vec, k=n, domain=self.domain)
        if self.mode == "dense":
            pool = dense
        else:
            sparse = lexical.search(self.db, question, k=n)
            pool = rrf([dense, sparse], weights=list(self.weights))[:n]
        out = [dict(c, route="chunk") for c in pool]
        seen = {c["chunk_id"] for c in out}
        for c in store.search_questions(self.db, vec, k=self.question_vectors,
                                        domain=self.domain):
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                out.append(dict(c, route="question"))
        return out

    def retrieve(self, question: str, k: int = KEEP) -> list[dict]:
        cands = self.candidates_for(question)
        if self.reranker is None:
            return cands[:k]
        return self.reranker.rerank(question, cands, top_k=k)

    def retrieve_fused(self, question: str, rewrites: list[str] | None = None,
                        k: int = KEEP, variant_candidates: int = VARIANT_CANDIDATES,
                        variant_k: int = VARIANT_K,
                        ) -> tuple[list[dict], int, list[str], list[dict]]:
        """`retrieve()`, fused across the question and its rewrites.

        Each variant (index 0 is `question`, the rest are `rewrites`) is retrieved
        at `variant_candidates` and reranked down to `variant_k` on its own; the
        reranked lists are then combined with `fuse_variants` (see module docstring
        there for why per-variant reranking, not a single fused-then-reranked pool)
        and the fused result is truncated to `k`. `variant_k` is deliberately NOT
        `k`: rrf() needs the lower-ranked-but-right hits a shallow list would have
        already dropped - see the measured numbers on VARIANT_K above.

        Also returns `gate_hits`: variant 0's (the original question's) own
        reranked list, unfused. After `rrf()` the fused list is ordered by RANK,
        not score, so `fused[0]["rerank_score"]` is whatever the winning variant
        happened to score - not comparable across queries, and not what
        `sweep_gate.py` swept `GATE`/`GATE_ACT` against (measured,
        tsk_20260828_a47f0496: gating on the fused top-1 dropped answerable p10
        from 0.80 to 0.30 and no threshold recovered the baseline trade). The
        gate must keep reading `gate_score(gate_hits)` - the original question's
        own top-1 - while the generator reads the fused `hits`. This is why
        `gate_score()` itself is untouched: it is still one definition, applied
        to a different, correctly-scoped list.

        Returns `(fused hits, winning variant index, variants, gate_hits)`.
        """
        variants = [question] + list(rewrites or [])
        lists_by_variant: list[list[dict]] = []

        def rerank_variant(q: str) -> list[dict]:
            cands = self.candidates_for(q, n=variant_candidates)
            result = (cands[:variant_k] if self.reranker is None
                      else self.reranker.rerank(q, cands, top_k=variant_k))
            lists_by_variant.append(result)
            return result

        fused, winner = fuse_variants(rerank_variant, variants, k=k)
        gate_hits = lists_by_variant[0] if lists_by_variant else []
        return fused, winner, variants, gate_hits

    def retrieve_widened(self, question: str, extra_queries: list[str],
                          k: int = KEEP) -> list[dict]:
        """Phase 11 R4d cascade (docs/phase11-results.md): the question's own
        dense candidates plus each of `extra_queries`'s (a rewrite, typically),
        deduplicated by `chunk_id`, reranked ONCE against the ORIGINAL
        `question` - never a rewrite, which never ranks anything here. This is
        deliberately unlike `retrieve_fused`/`fuse_variants`: there is no rank
        fusion in this path, just one candidate pool and one rerank call, so
        the resulting top-1 is a genuine cross-encoder score against the
        question - the same kind of number the gate was swept on
        (blk_fusion_gate_semantic_slip), safe for the gate to read directly
        with no separate gate_hits list needed.

        Each entry in `extra_queries` is retrieved at `self.candidates` - the
        SAME candidate count as the question's own pool, not
        `VARIANT_CANDIDATES` (that knob belongs to `retrieve_fused` alone).

        Order of the deduplicated pool: the question's own candidates first,
        then each extra query's new (not-yet-seen) candidates, in order - the
        rerank call re-sorts it regardless, so this only matters for the
        `reranker is None` fallback and for tests asserting exact pool
        composition.
        """
        pools = [self.candidates_for(question)] + [
            self.candidates_for(q) for q in extra_queries]
        seen: set[str] = set()
        merged: list[dict] = []
        for pool in pools:
            for c in pool:
                if c["chunk_id"] not in seen:
                    seen.add(c["chunk_id"])
                    merged.append(c)
        if self.reranker is None:
            return merged[:k]
        return self.reranker.rerank(question, merged, top_k=k)


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


def cap_per_doc(hits: list[dict], k: int, cap: int) -> list[dict]:
    """Cap how many hits from any one `doc_id` survive into the top-k.

    Exists for the phase 7 same-document crowding measurement: 41/166 top-5
    lists held 4+ chunks of a single document, squeezing out other documents
    whose evidence sat just below k. This is a query-side cut of an already
    ranked pool - it never reads gate_hits and the gate never reads its
    output (blk_fusion_gate_semantic_slip: the gate keeps reading
    gate_score(gate_hits), computed before this runs).

    Walks `hits` in order, keeping a hit while fewer than `cap` hits from its
    `doc_id` have been kept so far, until `k` are kept. If fewer than `k`
    survive that pass, tops up with the skipped hits, in their original pool
    order, so the result always has `min(k, len(hits))` items. `cap <= 0`
    means no cap: returns `hits[:k]`. Never mutates `hits` or its dicts.
    """
    if cap <= 0:
        return hits[:k]
    kept, skipped, per_doc = [], [], {}
    for h in hits:
        if len(kept) >= k:
            break
        doc_id = h.get("doc_id")
        if per_doc.get(doc_id, 0) < cap:
            kept.append(h)
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
        else:
            skipped.append(h)
    if len(kept) < k:
        kept.extend(skipped[: k - len(kept)])
    return kept
