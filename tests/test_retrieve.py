"""Tests for query-rewrite fusion (tsk_20260828_a47f0496): fuse the original
question with model-generated rewrites, each retrieved AND reranked separately,
combined by rrf() - and surface which variant's list won.

Hermetic by design (blk_test_env_constraints): this worktree has no .venv, no
data/, no models/, and `smm.retrieve` imports `smm.store`, which imports the
third-party `sqlite_vec` (absent here, and not to be added - it's a real
extension module, not something worth stubbing meaningfully for a unit test).
`sqlite_vec` is stubbed in `sys.modules` before the import below so `smm.store`
loads without it; nothing here ever opens a real sqlite3 connection or calls
into the stub, so its contents don't matter, only its presence.

`fuse_variants` is exercised directly with a plain `retrieve_fn` callable (no DB,
no GPU, no third-party imports) exactly as designed for testability. The winner
scenario in `test_fusion_promotes_low_ranked_original_hit` is modelled on the
measured numbers from tsk_20260828_a47f0496: the query "how to list files via
size" reranks `ls.1:4` (the correct chunk) behind four `size.1` /
`x86_64-linux-gnu-size.1` chunks (0.9967 / 0.9957 / 0.9892 / 0.9848 vs 0.9661) -
a rewrite that does not surface the `size` pages at all lets rank fusion recover
the gold chunk to the top.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

if "sqlite_vec" not in sys.modules:
    stub = types.ModuleType("sqlite_vec")
    stub.load = lambda *a, **kw: None  # never called: no real db is opened here
    sys.modules["sqlite_vec"] = stub

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smm.retrieve import KEEP, VARIANT_K, Retriever, fuse_variants, gate_score, rrf  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def c(chunk_id, **kw):
    return dict(chunk_id=chunk_id, **kw)


# --------------------------------------------------------------------------
# fuse_variants: pure, hermetic, exercised with a plain retrieve_fn callable.
# --------------------------------------------------------------------------


def test_fusion_promotes_low_ranked_original_hit():
    """Modelled on the measured evidence: the original question's reranked list
    buries the gold chunk (ls.1:4) behind four size.1 chunks; a rewrite that
    does not surface `size` pages ranks it first. Fusing the two RERANKED lists
    by rrf() must recover ls.1:4 to the top - and name the rewrite as the winner.
    """
    question = "how to list files via size"
    rewrite = "command to see the largest files in a directory"
    variants = [question, rewrite]

    original_reranked = [
        c("size.1:1", rerank_score=0.9967),
        c("size.1:2", rerank_score=0.9957),
        c("x86_64-linux-gnu-size.1:1", rerank_score=0.9892),
        c("x86_64-linux-gnu-size.1:2", rerank_score=0.9848),
        c("ls.1:4", rerank_score=0.9661),
    ]
    rewrite_reranked = [
        c("ls.1:4", rerank_score=0.99),
        c("du.1:2", rerank_score=0.80),
        c("sort.1:1", rerank_score=0.75),
    ]
    lists = {question: original_reranked, rewrite: rewrite_reranked}

    calls = []

    def retrieve_fn(q):
        calls.append(q)
        return lists[q]

    fused, winner = fuse_variants(retrieve_fn, variants, k=KEEP)

    check(calls == variants, f"each variant must be retrieved exactly once, in order: {calls}")
    check(fused, "fusion should not return an empty list")
    check(fused[0]["chunk_id"] == "ls.1:4",
          f"gold chunk should win the fused ranking, got {fused[0]['chunk_id']}")
    check(winner == 1, f"the rewrite (index 1) should be named as the winner, got {winner}")
    check(variants[winner] == rewrite, "winner index must map back to the rewrite text")

    # Sanity check against a plain, un-fused rrf() call over the same lists:
    # fusing must genuinely move the chunk, not just relabel the original top.
    plain = rrf([original_reranked, rewrite_reranked])
    check(plain[0]["chunk_id"] == "ls.1:4", "cross-check: rrf() over these two lists agrees")


def test_fusion_original_wins_when_it_is_actually_better():
    """The original question is `variants[0]`; when its list ranks the gold hit
    first (and the rewrite only surfaces it lower, behind an off-target hit of
    its own), the fused winner must be attributed to the original question
    (index 0), not the rewrite."""
    question = "how do I watch a log file as it grows"
    rewrite = "monitor a file for changes"
    variants = [question, rewrite]

    original_reranked = [c("tail.1:2", rerank_score=0.98), c("other.1:1", rerank_score=0.4)]
    rewrite_reranked = [c("watch.1:1", rerank_score=0.6), c("tail.1:2", rerank_score=0.5),
                        c("inotifywait.1:1", rerank_score=0.3)]
    lists = {question: original_reranked, rewrite: rewrite_reranked}

    fused, winner = fuse_variants(lambda q: lists[q], variants, k=KEEP)

    check(fused[0]["chunk_id"] == "tail.1:2", f"expected tail.1:2 on top, got {fused[0]}")
    check(winner == 0, f"the original question should be named winner, got {winner}")


def test_fusion_respects_k():
    """The fused list is truncated to k, same contract as Retriever.retrieve()."""
    variants = ["q1", "q2"]
    lists = {
        "q1": [c(f"a{i}", rerank_score=1.0 - i * 0.01) for i in range(5)],
        "q2": [c(f"b{i}", rerank_score=1.0 - i * 0.01) for i in range(5)],
    }
    fused, _winner = fuse_variants(lambda q: lists[q], variants, k=3)
    check(len(fused) == 3, f"fused list should be truncated to k=3, got {len(fused)}")


def test_fusion_empty_variant_list_is_a_noop():
    fused, winner = fuse_variants(lambda q: [], [], k=KEEP)
    check(fused == [], "no variants -> no fused hits")
    check(winner == 0, "winner defaults to 0 with nothing retrieved")


# --------------------------------------------------------------------------
# Retriever.retrieve_fused: call-count contract at the level ask.py uses.
# --------------------------------------------------------------------------


def test_retrieve_fused_with_no_rewrites_issues_exactly_one_retrieval_call():
    """rewrites=0 (`ask.py --rewrites 0`, or omitted rewrites here) must retrieve
    the question and nothing else - one call, not N+1."""
    r = Retriever(db=None, embedder=None, reranker=None, mode="dense")
    calls = []

    def fake_candidates_for(question, n=None):
        calls.append((question, n))
        return [c("only.1:1", score=0.9)]

    r.candidates_for = fake_candidates_for  # instance override; no real db touched

    fused, winner, variants, gate_hits = r.retrieve_fused("how do I list files", rewrites=None, k=KEEP)

    check(len(calls) == 1, f"expected exactly one retrieval call, got {len(calls)}: {calls}")
    check(variants == ["how do I list files"], f"variants should be just the question: {variants}")
    check(winner == 0, "the only variant is the question itself")
    check(fused and fused[0]["chunk_id"] == "only.1:1", f"unexpected fused result: {fused}")


def test_retrieve_fused_issues_one_call_per_variant_at_variant_candidates():
    """N rewrites -> N+1 retrieval calls, each at variant_candidates (default 20
    per the measured briefing), independent of the top-level `candidates` the
    Retriever was constructed with."""
    r = Retriever(db=None, embedder=None, reranker=None, mode="dense", candidates=50)
    calls = []

    def fake_candidates_for(question, n=None):
        calls.append((question, n))
        return [c(f"{question}-hit", score=0.9)]

    r.candidates_for = fake_candidates_for

    fused, winner, variants, gate_hits = r.retrieve_fused(
        "how do I list files", rewrites=["show files by size", "ls command sort"], k=KEEP,
    )

    check(len(calls) == 3, f"expected 3 retrieval calls (question + 2 rewrites), got {len(calls)}")
    check(all(n == 20 for _, n in calls), f"each variant call should use candidates=20: {calls}")
    check(variants == ["how do I list files", "show files by size", "ls command sort"], variants)
    check(fused, "fused result should not be empty")


# --------------------------------------------------------------------------
# Depth contract: variant_k (the per-variant RERANKED-list depth fed to rrf())
# must never be confused with k (the caller's final OUTPUT size). Attempt 1
# passed `k` straight through as each variant's `top_k`, which starved rrf()
# of exactly the lower-ranked-but-right hits it exists to promote - measured
# live: gold only reached @4 (still behind two wrong hits) at depth 5, but @2
# at the intended depth 20. None of the tests above catch this: they all use
# `reranker=None`, which never exercises `top_k` at all.
# --------------------------------------------------------------------------


class StubReranker:
    """Records the query and top_k each call was asked for; returns the
    candidates truncated to top_k, preserving order - just enough behaviour to
    pin the depth contract without a real cross-encoder."""

    def __init__(self):
        self.calls: list[tuple[str, int | None]] = []

    def rerank(self, query, chunks, top_k=None, batch=16):
        self.calls.append((query, top_k))
        return chunks[:top_k] if top_k else list(chunks)


def test_retrieve_fused_reranks_each_variant_to_variant_k_not_k():
    """The regression this attempt fixes: the caller's output size `k` must
    never leak into the per-variant reranker's `top_k`. Each variant is
    reranked to `variant_k` (VARIANT_K by default); only the FUSED result is
    truncated to `k`."""
    r = Retriever(db=None, embedder=None, reranker=StubReranker(), mode="dense")

    def fake_candidates_for(question, n=None):
        return [c(f"{question}-{i}", score=1.0 - i * 0.01) for i in range(25)]

    r.candidates_for = fake_candidates_for

    fused, winner, variants, gate_hits = r.retrieve_fused(
        "how to list files via size", rewrites=["largest files command"], k=5,
    )

    calls = r.reranker.calls
    check(len(calls) == 2, f"expected one rerank call per variant, got {len(calls)}: {calls}")
    check(all(top_k == VARIANT_K for _, top_k in calls),
          f"each variant's rerank call must ask for top_k=VARIANT_K (20), not k=5: {calls}")
    check(all(top_k != 5 for _, top_k in calls),
          f"k must not leak into the per-variant rerank call: {calls}")
    check(len(fused) == 5, f"fused OUTPUT must still be truncated to k=5, got {len(fused)}")


def test_retrieve_fused_variant_k_is_overridable():
    """An explicit `variant_k` overrides the VARIANT_K default, independent of
    `k` - the two knobs stay separately controllable."""
    r = Retriever(db=None, embedder=None, reranker=StubReranker(), mode="dense")

    def fake_candidates_for(question, n=None):
        return [c(f"{question}-{i}", score=1.0 - i * 0.01) for i in range(10)]

    r.candidates_for = fake_candidates_for

    fused, winner, variants, gate_hits = r.retrieve_fused(
        "how to list files via size", rewrites=["largest files command"], k=3, variant_k=7,
    )

    calls = r.reranker.calls
    check(all(top_k == 7 for _, top_k in calls),
          f"explicit variant_k=7 must override the VARIANT_K default: {calls}")
    check(len(fused) == 3, f"fused output should still respect k=3, got {len(fused)}")


# --------------------------------------------------------------------------
# Gate contract: gate_hits must be variant 0's (the original question's) own
# reranked list, never the fused one. After rrf() the fused list is ordered by
# RANK, so its top-1 rerank_score is whatever the winning variant happened to
# score - not comparable across queries, and not the distribution GATE/GATE_ACT
# were swept against. Measured, tsk_20260828_a47f0496: gating on the fused
# top-1 dropped answerable p10 from 0.80 to 0.30; no threshold recovered the
# baseline trade. Modelled below on the real numbers: a rewrite wins the fused
# ranking with a low score (~0.30) while the original question's own top-1
# carries a high one (~0.98).
# --------------------------------------------------------------------------


def test_gate_hits_is_the_original_questions_own_list_not_the_fused_winner():
    """Modelled on the coordinator's measured numbers: a fused winner scoring
    ~0.30 vs. the original question's own top-1 scoring ~0.98. `ls.1:4` is
    buried at rank 3 in the original's own list (score 0.30 there) but wins
    the FUSED ranking because the rewrite also ranks it first; rrf() seeds a
    merged chunk's score fields from whichever list it saw first (here, the
    original's own low score, 0.30 - not the rewrite's 0.99). The gate must
    read `size.1:1` at 0.98 (the original's own top-1), never `ls.1:4` at 0.30
    (the fused winner's own score)."""
    question = "how to list files via size"
    rewrite = "command to see the largest files in a directory"

    original_reranked = [
        c("size.1:1", rerank_score=0.98),
        c("size.1:2", rerank_score=0.90),
        c("ls.1:4", rerank_score=0.30),
    ]
    rewrite_reranked = [c("ls.1:4", rerank_score=0.99)]
    lists = {question: original_reranked, rewrite: rewrite_reranked}

    r = Retriever(db=None, embedder=None, reranker=StubReranker(), mode="dense")
    r.candidates_for = lambda q, n=None: lists[q]

    fused, winner, variants, gate_hits = r.retrieve_fused(question, rewrites=[rewrite], k=5)

    check(fused and fused[0]["chunk_id"] == "ls.1:4",
          f"expected ls.1:4 to win the fused ranking, got {fused}")
    check(fused[0]["rerank_score"] == 0.30,
          f"sanity: the fused winner's own carried score is the low one (0.30), got {fused[0]}")
    check(gate_hits == original_reranked,
          f"gate_hits must be variant 0's own reranked list, got {gate_hits}")
    check(gate_score(gate_hits) == 0.98,
          f"gate_score(gate_hits) must read the original's own top-1 (0.98), "
          f"got {gate_score(gate_hits)}")
    check(gate_score(gate_hits) != gate_score(fused),
          f"the gate must not read the fused winner's own score (0.30): "
          f"gate_score(gate_hits)={gate_score(gate_hits)} gate_score(fused)={gate_score(fused)}")


def test_gate_hits_equals_fused_hits_when_there_are_no_rewrites():
    """`rewrites=0`/`None`: with only one variant, rrf() over a single list
    cannot reorder it, so gate_hits (variant 0's own reranked list) must carry
    the same ranking, and the same gate score, as the fused list - the
    single-variant case ask.py's `--rewrites 0` path relies on (there, ask.py
    skips retrieve_fused entirely and sets `gate_hits = hits` directly; this
    pins that retrieve_fused itself agrees when it is given zero rewrites)."""
    r = Retriever(db=None, embedder=None, reranker=None, mode="dense")

    def fake_candidates_for(question, n=None):
        return [c("only.1:1", score=0.9), c("only.1:2", score=0.5)]

    r.candidates_for = fake_candidates_for

    fused, winner, variants, gate_hits = r.retrieve_fused("q", rewrites=None, k=5)

    fused_ids = [h["chunk_id"] for h in fused]
    gate_ids = [h["chunk_id"] for h in gate_hits[:len(fused)]]
    check(fused_ids == gate_ids,
          f"with no rewrites, gate_hits and fused hits must rank identically: "
          f"{gate_ids} != {fused_ids}")
    check(gate_score(gate_hits) == gate_score(fused),
          f"with no rewrites, gate_score must agree whether read from gate_hits "
          f"or fused: {gate_score(gate_hits)} != {gate_score(fused)}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  pass  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001 - a crashing test is still a failure to report
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
