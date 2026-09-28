"""Tests for src/smm/cascade.py and its two call sites (phase 11 R4d build,
tsk_20260928_cascade): the tier 0/1/2 cascade pre-registered in
docs/phase11-results.md, "R4d pre-registration - cascade - rewrite only when
the plain search refuses".

Hermetic by design (blk_test_env_constraints): no pytest, no real
llama-server, no data/ or .venv. `smm.cascade` imports `smm.retrieve`, which
imports `smm.store`, which imports the third-party `sqlite_vec` (absent
here); stubbed in sys.modules before import, exactly as tests/test_retrieve.py
and tests/test_eval_answers.py do. The embedder, reranker, generator and
retriever are all stubs - no server, no GPU, no index - except in
`test_widened_pool_deduped_and_reranked_against_original`, which exercises
the REAL `smm.retrieve.Retriever.retrieve_widened` (with a stub reranker and
a monkeypatched `candidates_for`, the same house idiom as
tests/test_retrieve.py's StubReranker) so the dedup/rerank contract itself
is pinned, not just the tier loop's calls into it.

`eval_answers`-facing tests load scripts/eval_answers.py via importlib (same
pattern as tests/test_docvocab.py and tests/test_eval_answers.py) and replace
its Embedder/Reranker/Retriever/Generator/store module globals with in-process
stubs, repointing ROOT at a tempfile.TemporaryDirectory() so the run never
touches this repo's own data/eval/results/.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

if "sqlite_vec" not in sys.modules:
    stub = types.ModuleType("sqlite_vec")
    stub.load = lambda *a, **kw: None  # never called: no real db is opened here
    sys.modules["sqlite_vec"] = stub

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import cascade, grammar  # noqa: E402
from smm.retrieve import Retriever  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "eval_answers", ROOT / "scripts" / "eval_answers.py")
eval_answers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_answers)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def hit(chunk_id: str, score: float) -> dict:
    return {"chunk_id": chunk_id, "doc_id": chunk_id, "text": chunk_id, "score": score}


# --------------------------------------------------------------------------
# smm.cascade.run_cascade: the tier 0/1/2 loop, exercised with plain stubs.
# --------------------------------------------------------------------------


class FakeRetriever:
    """`.retrieve()` always returns the fixed tier-0 hits; each call to
    `.retrieve_widened()` returns the next list from `widened_by_tier`, in
    order - one call per escalated tier, so the Nth widen call is tier N."""

    def __init__(self, tier0_hits: list, widened_by_tier: list):
        self.tier0_hits = tier0_hits
        self.widened_by_tier = widened_by_tier
        self.retrieve_calls: list = []
        self.widen_calls: list = []

    def retrieve(self, question: str, k: int = 5) -> list:
        self.retrieve_calls.append((question, k))
        return self.tier0_hits

    def retrieve_widened(self, question: str, extra_queries: list, k: int = 5) -> list:
        idx = len(self.widen_calls)
        self.widen_calls.append((question, list(extra_queries), k))
        return self.widened_by_tier[idx]


class FakeGenerator:
    """Each call to `.rewrites()` returns the next list from `rewrites_by_call`,
    in order (tier 1's call first, tier 2's second)."""

    def __init__(self, rewrites_by_call: list):
        self.rewrites_by_call = rewrites_by_call
        self.rewrite_calls: list = []

    def rewrites(self, question: str, n: int = 1, style: str = "man") -> list:
        idx = len(self.rewrite_calls)
        self.rewrite_calls.append((question, n, style))
        return self.rewrites_by_call[idx]


def test_tier0_answer_stops_without_rewrite():
    r = FakeRetriever([hit("a", 0.9)], [])
    g = FakeGenerator([])
    answer_calls = []

    def answer_fn(q, hits):
        answer_calls.append((q, hits))
        return "Use `-r` [1] to recurse."

    result = cascade.run_cascade(r, g, "how do I recurse", answer_fn, gate=0.65, k=5)

    check(result["tier"] == 0, f"a passing tier 0 must be final, got {result}")
    check(g.rewrite_calls == [], "generator.rewrites() must never be called")
    check(r.widen_calls == [], "retrieve_widened must never be called")
    check(len(answer_calls) == 1, f"the reader must be called exactly once, got {answer_calls}")
    check(result["answer"] == "Use `-r` [1] to recurse.", result)
    check(result["rewrites"] == [], result)
    check(result["seconds"] == 0.0, "tier 0 adds no wall time beyond itself")


def test_gated_goes_to_tier1():
    r = FakeRetriever([hit("a", 0.30)], [[hit("b", 0.90)]])
    g = FakeGenerator([["a docs-style rewrite"]])

    def answer_fn(q, hits):
        return "Use `-r` [1] to recurse."

    result = cascade.run_cascade(r, g, "Q", answer_fn, gate=0.65, k=5)

    check(result["tier"] == 1, f"a tier-0 gate below threshold must escalate, got {result}")
    check(g.rewrite_calls == [("Q", 1, "docs")],
          f"tier 1 must ask for exactly 1 docs-style rewrite, got {g.rewrite_calls}")
    check(r.widen_calls == [("Q", ["a docs-style rewrite"], 5)], r.widen_calls)
    check(result["answer"] == "Use `-r` [1] to recurse.", result)


def test_reader_refusal_goes_to_tier1():
    r = FakeRetriever([hit("a", 0.90)], [[hit("b", 0.90)]])
    g = FakeGenerator([["a docs-style rewrite"]])
    seen = []

    def answer_fn(q, hits):
        seen.append(q)
        return "I don't know." if len(seen) == 1 else "Use `-x` [1]."

    result = cascade.run_cascade(r, g, "Q", answer_fn, gate=0.65, k=5)

    check(result["tier"] == 1,
          f"a tier-0 pass-the-gate reader refusal must still escalate, got {result}")
    check(g.rewrite_calls == [("Q", 1, "docs")], g.rewrite_calls)
    check(result["answer"] == "Use `-x` [1].", result)


def test_tier2_is_final():
    r = FakeRetriever([hit("a", 0.30)], [[hit("b", 0.30)], [hit("c", 0.30)]])
    g = FakeGenerator([["r1"], ["r2a", "r2b"]])

    def answer_fn(q, hits):
        return "I don't know."

    result = cascade.run_cascade(r, g, "Q", answer_fn, gate=0.65, k=5)

    check(result["tier"] == 2, f"tier 2 must be final, got {result}")
    check(len(g.rewrite_calls) == 2, f"exactly 2 rewrite calls total, got {g.rewrite_calls}")
    check(g.rewrite_calls[1] == ("Q", 2, "docs"),
          f"tier 2 must ask for a FRESH n=2 call, got {g.rewrite_calls[1]}")
    check(len(r.widen_calls) == 2, f"exactly 2 widen calls total (no tier 3), got {r.widen_calls}")
    check(result["gated"] is True, "still gated at tier 2 - returned anyway, whatever it is")
    check(result["answer"] == grammar.REFUSAL, result)


def test_gate_reads_widened_top1():
    """The widened list's TOP-1 (0.40, still below the 0.65 gate) must decide
    tier 1's outcome, not a higher-scoring hit further down the SAME list
    (0.99): a gate that read anything else (e.g. the list's max) would wrongly
    pass tier 1 and the cascade would stop there instead of escalating to
    tier 2 - so tier 2 actually running, with tier 1's reader never called,
    is what pins the gate to hits[0]."""
    r = FakeRetriever(
        [hit("a", 0.30)],
        [[hit("low", 0.40), hit("high_not_first", 0.99)], [hit("c", 0.90)]],
    )
    g = FakeGenerator([["r1"], ["r2a", "r2b"]])
    answered_at = []

    def answer_fn(q, hits):
        answered_at.append(hits[0]["chunk_id"])
        return "Use `-y` [1]."

    result = cascade.run_cascade(r, g, "Q", answer_fn, gate=0.65, k=5)

    check(len(g.rewrite_calls) == 2,
          f"tier 1 must have been gated (top-1 0.40 < 0.65) and escalated to "
          f"tier 2 - a gate reading anything else (e.g. the list's max, 0.99) "
          f"would have stopped at tier 1: {g.rewrite_calls}")
    check(result["tier"] == 2, result)
    check(answered_at == ["c"], f"the reader must only be called at tier 2: {answered_at}")


def test_reader_sees_original_question():
    r = FakeRetriever([hit("a", 0.30)], [[hit("b", 0.90)]])
    g = FakeGenerator([["a totally different rewritten query"]])
    seen = []

    def answer_fn(q, hits):
        seen.append(q)
        return "Use `-x` [1]."

    cascade.run_cascade(r, g, "original question text", answer_fn, gate=0.65, k=5)

    check(seen == ["original question text"],
          f"the reader must always see the ORIGINAL question, never a rewrite: {seen}")


# --------------------------------------------------------------------------
# retrieve_widened, through the tier loop: dedup + a single rerank call
# against the ORIGINAL question, per tier - the real Retriever, a stub
# reranker, and a monkeypatched candidates_for (test_retrieve.py's idiom).
# --------------------------------------------------------------------------


class RecordingReranker:
    """Records (query, chunk_ids, top_k) for every call; returns the pool
    truncated to top_k, unreordered - just enough to drive gating without
    hiding what pool it was actually asked to rerank."""

    def __init__(self):
        self.calls: list = []

    def rerank(self, query: str, candidates: list, top_k=None) -> list:
        self.calls.append((query, [c["chunk_id"] for c in candidates], top_k))
        return candidates[:top_k]


def test_widened_pool_deduped_and_reranked_against_original():
    """The question's own dense candidates are the SAME `candidates_for("Q")`
    call at every tier (tier 0's plain retrieve AND every widened pool), so
    the fixture below is deliberately built so every tier's rerank stays
    gated (0.10 < 0.65) - this pins the pool COMPOSITION at each tier, not
    the pass/fail path (already covered above)."""
    rr = RecordingReranker()
    retriever = Retriever(db=None, embedder=None, reranker=rr, mode="dense", candidates=3)

    pools = {
        "Q": [hit("d0a", 0.10)],
        "R1": [hit("d0a", 0.10), hit("b1", 0.90)],   # d0a overlaps Q: must be deduped
        "R2a": [hit("d0a", 0.10), hit("c1", 0.95)],  # d0a overlaps Q: must be deduped
        "R2b": [hit("c1", 0.95), hit("d1", 0.99)],   # c1 overlaps R2a: must be deduped
    }

    def fake_candidates_for(q, n=None):
        return pools[q]

    retriever.candidates_for = fake_candidates_for

    g = FakeGenerator([["R1"], ["R2a", "R2b"]])

    def answer_fn(q, hits):
        raise AssertionError("every tier here is gated: the reader must never be called")

    result = cascade.run_cascade(retriever, g, "Q", answer_fn, gate=0.65, k=3)
    check(result["tier"] == 2, f"every tier must be gated here, got {result}")

    check(len(rr.calls) == 3, f"expected 3 rerank calls (tier 0, 1, 2), got {rr.calls}")

    tier0_query, tier0_ids, tier0_k = rr.calls[0]
    check(tier0_query == "Q" and tier0_ids == ["d0a"] and tier0_k == 3, rr.calls[0])

    tier1_query, tier1_ids, tier1_k = rr.calls[1]
    check(tier1_query == "Q", f"tier 1 must rerank against the ORIGINAL question: {rr.calls[1]}")
    check(tier1_ids == ["d0a", "b1"] and tier1_k == 3,
          f"tier 1's pool must be Q's candidates + R1's NEW ones, d0a deduped: {rr.calls[1]}")

    tier2_query, tier2_ids, tier2_k = rr.calls[2]
    check(tier2_query == "Q", f"tier 2 must rerank against the ORIGINAL question: {rr.calls[2]}")
    check(tier2_ids == ["d0a", "c1", "d1"] and tier2_k == 3,
          f"tier 2's pool must be Q's + both fresh rewrites' NEW candidates, "
          f"d0a and c1 deduped: {rr.calls[2]}")


# --------------------------------------------------------------------------
# scripts/eval_answers.py --cascade: the stage-both requirement, the exact
# per-row record shape, and no-cascade byte-identical output - loaded via
# importlib with its Embedder/Reranker/Retriever/Generator/store module
# globals stubbed (tests/test_docvocab.py's idiom).
# --------------------------------------------------------------------------


class _StubHealthy:
    """Stands in for Embedder()/Reranker(): no-arg constructor, always up."""

    def __init__(self, *a, **kw):
        pass

    def health(self) -> bool:
        return True


def _make_eval_stub_retriever(low_hit: dict, widened_hit: dict, calls: dict):
    class _StubRetriever:
        def __init__(self, db, embedder=None, reranker=None, mode=None,
                     candidates=None, domain=None):
            pass

        def retrieve(self, question, k=5):
            calls.setdefault("retrieve", []).append(question)
            return [dict(low_hit)]

        def retrieve_widened(self, question, extra_queries, k=5):
            calls.setdefault("widen", []).append((question, list(extra_queries)))
            return [dict(widened_hit)]

        def retrieve_fused(self, question, rewrites=None, k=5,
                            variant_candidates=None, variant_k=None):
            calls.setdefault("retrieve_fused", []).append(question)
            return [dict(low_hit)], 0, [], [dict(low_hit)]

    return _StubRetriever


def _make_eval_stub_generator(answer_text: str, rewrite_texts: list, calls: dict):
    class _StubGenerator:
        def __init__(self, url=None, timeout=300):
            pass

        def health(self) -> bool:
            return True

        def correct(self, question, max_tokens=120):
            return question

        def rewrites(self, question, n=1, max_tokens=120, style="man"):
            calls.setdefault("rewrites", []).append((question, n, style))
            return rewrite_texts

        def answer(self, question, chunks, cite_grammar=False, mode="cite", **kw):
            calls.setdefault("answer", []).append(question)
            return answer_text

    return _StubGenerator


_GOOD_TEXT = "apt-get install installs a package."


def _write_eval_row(tmp: Path) -> Path:
    eval_path = tmp / "eval.jsonl"
    eval_path.write_text(json.dumps({
        "qid": "q1", "kind": "answerable", "tags": [],
        "question": "how do I install a package",
        "answer_contains": ["apt-get install"],
    }) + "\n")
    return eval_path


def _patch_eval_answers(**attrs):
    """Snapshot then overwrite module globals on `eval_answers` (always
    including ROOT, even if not passed here - every caller sets it itself
    right after, to a fresh tempdir); returns the snapshot for restoration."""
    attrs.setdefault("ROOT", eval_answers.ROOT)
    old = {name: getattr(eval_answers, name) for name in attrs}
    for name, val in attrs.items():
        setattr(eval_answers, name, val)
    return old


def _unpatch_eval_answers(old: dict):
    for name, val in old.items():
        setattr(eval_answers, name, val)


def test_no_cascade_output_unchanged():
    """Without --cascade, an answerable row's record must carry exactly the
    pre-existing key set - no cascade_tier/cascade_seconds/cascade_rewrites,
    nothing else added or removed."""
    calls: dict = {}
    old_argv = sys.argv
    old = _patch_eval_answers(
        Embedder=_StubHealthy, Reranker=_StubHealthy,
        Retriever=_make_eval_stub_retriever(
            {"chunk_id": "c1", "doc_id": "d1", "text": _GOOD_TEXT, "score": 0.9,
             "rerank_score": 0.9},
            {"chunk_id": "c1", "doc_id": "d1", "text": _GOOD_TEXT, "score": 0.9},
            calls),
        Generator=_make_eval_stub_generator("apt-get install [1]", [], calls),
        store=types.SimpleNamespace(connect=lambda path: "STUB_DB"),
    )
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            eval_answers.ROOT = tmp
            eval_path = _write_eval_row(tmp)

            sys.argv = ["eval_answers.py", "--stage", "both", "--eval", str(eval_path),
                        "--name", "no-cascade-test", "--db", "stub.db", "--no-aliases"]
            rc = eval_answers.main()
            check(rc == 0, f"expected exit 0, got {rc}")

            out = json.loads((tmp / "data" / "eval" / "results" /
                               "no-cascade-test-answers.json").read_text())
            rec = out["results"][0]
            expected_keys = {
                "qid", "kind", "tags", "variant_kind", "question", "answer",
                "abstained", "gated", "retrieved_docs", "top_score",
                "correct", "correct_strict", "evidence_retrieved",
                "evidence_retrieved_strict", "citations", "n_citations",
                "uncited", "out_of_range", "cite_supported",
            }
            check(set(rec.keys()) == expected_keys,
                  f"no-cascade record must carry exactly the pre-existing keys, "
                  f"got {sorted(rec.keys())}")
            check("cascade" not in out["config"], "config must not gain a cascade key")
    finally:
        sys.argv = old_argv
        _unpatch_eval_answers(old)


def test_eval_records_cascade_tier_and_seconds():
    """A stubbed tier-1 rescue: tier 0's own hit is gated (0.10 < 0.65),
    the widened tier-1 hit passes (0.90) and answers - the record must carry
    cascade_tier=1, a non-negative cascade_seconds, and cascade_rewrites
    equal to the one rewrite tier 1 actually used."""
    calls: dict = {}
    low_hit = {"chunk_id": "low", "doc_id": "low-doc", "text": "irrelevant",
               "score": 0.10, "rerank_score": 0.10}
    good_hit = {"chunk_id": "good", "doc_id": "d1", "text": _GOOD_TEXT,
                "score": 0.90, "rerank_score": 0.90}
    old_argv = sys.argv
    old = _patch_eval_answers(
        Embedder=_StubHealthy, Reranker=_StubHealthy,
        Retriever=_make_eval_stub_retriever(low_hit, good_hit, calls),
        Generator=_make_eval_stub_generator(
            "apt-get install [1]", ["a docs-style rewrite"], calls),
        store=types.SimpleNamespace(connect=lambda path: "STUB_DB"),
    )
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            eval_answers.ROOT = tmp
            eval_path = _write_eval_row(tmp)

            sys.argv = ["eval_answers.py", "--cascade", "--stage", "both",
                        "--eval", str(eval_path), "--name", "cascade-test",
                        "--db", "stub.db", "--no-aliases", "--gate", "0.65"]
            rc = eval_answers.main()
            check(rc == 0, f"expected exit 0, got {rc}")

            check(calls.get("rewrites") == [("how do I install a package", 1, "docs")],
                  f"tier 1 must ask for exactly 1 docs-style rewrite: {calls.get('rewrites')}")
            check(calls.get("widen") == [("how do I install a package",
                                          ["a docs-style rewrite"])],
                  f"widened retrieval must be queried with the ORIGINAL question: "
                  f"{calls.get('widen')}")

            out = json.loads((tmp / "data" / "eval" / "results" /
                               "cascade-test-answers.json").read_text())
            rec = out["results"][0]
            check(rec["cascade_tier"] == 1, f"expected a tier-1 rescue, got {rec}")
            check(rec["cascade_rewrites"] == ["a docs-style rewrite"], rec)
            check(isinstance(rec["cascade_seconds"], float) and rec["cascade_seconds"] >= 0.0,
                  rec)
            check(rec["gated"] is False, rec)
            check(rec["answer"] == "apt-get install [1]", rec)
            check(out["config"].get("cascade") is True, out["config"])
    finally:
        sys.argv = old_argv
        _unpatch_eval_answers(old)


def test_cascade_requires_stage_both():
    """--cascade with --stage retrieve or --stage generate must exit 2
    before touching any file or server - the tier depends on the reader's
    own answer, so the two stages cannot be split."""
    old_argv = sys.argv
    try:
        for stage in ("retrieve", "generate"):
            sys.argv = ["eval_answers.py", "--cascade", "--stage", stage]
            rc = eval_answers.main()
            check(rc == 2, f"--cascade --stage {stage} must exit 2, got {rc}")
    finally:
        sys.argv = old_argv


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
