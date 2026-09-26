"""Tests for phase 7 evidence depth (tsk_20260926_0967344e): reading more
extracts (read-k) and capping chunks per document (cap-per-doc), so a
same-document-crowded top-k does not starve other documents' evidence just
below it. `cap_per_doc` (smm.retrieve) is the pure query-side cut; `select_hits`
(scripts/eval_answers.py) is the generate-stage helper that applies it to what
the MODEL reads, while the gate keeps reading gate_hits untouched
(blk_fusion_gate_semantic_slip).

Hermetic by design (blk_test_env_constraints): this worktree has no .venv, no
data/, no models/, and `scripts.eval_answers` imports `smm.store`, which
imports the third-party `sqlite_vec` (absent here). `sqlite_vec` is stubbed in
sys.modules before the import below, exactly as tests/test_eval_answers.py and
tests/test_retrieve.py do; nothing here ever opens a real sqlite3 connection or
calls into the stub, so its contents don't matter, only its presence. The
script is then loaded via importlib so this file works with no changes to
sys.path beyond the src/ insert below.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

if "sqlite_vec" not in sys.modules:
    stub = types.ModuleType("sqlite_vec")
    stub.load = lambda *a, **kw: None  # never called: no real db is opened here
    sys.modules["sqlite_vec"] = stub

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm.retrieve import cap_per_doc, gate_score  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "eval_answers", ROOT / "scripts" / "eval_answers.py")
eval_answers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_answers)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def c(chunk_id, doc_id, **kw):
    return dict(chunk_id=chunk_id, doc_id=doc_id, **kw)


def test_cap_respects_cap_and_order():
    """docs [A,A,A,B,A,C,B] with k=4, cap=2 gives [A,A,B,C] in pool order:
    the third A is skipped (cap reached), B and C fill the remaining slots
    in the order they appear, and the walk stops once 4 are kept."""
    hits = [
        c("a1", "A"), c("a2", "A"), c("a3", "A"), c("b1", "B"),
        c("a4", "A"), c("c1", "C"), c("b2", "B"),
    ]
    got = cap_per_doc(hits, k=4, cap=2)
    check([h["chunk_id"] for h in got] == ["a1", "a2", "b1", "c1"],
          f"expected [a1,a2,b1,c1] in pool order, got {[h['chunk_id'] for h in got]}")


def test_cap_tops_up_when_pool_is_one_doc():
    """All-A hits with k=5, cap=2: only 2 can be kept under the cap, so the
    other 3 (skipped for being over-cap) top up the result in their original
    order, giving all 5 hits back."""
    hits = [c(f"a{i}", "A") for i in range(1, 6)]
    got = cap_per_doc(hits, k=5, cap=2)
    check(len(got) == 5, f"expected 5 hits (topped up), got {len(got)}")
    check([h["chunk_id"] for h in got] == ["a1", "a2", "a3", "a4", "a5"],
          f"expected the 2 kept then 3 topped up, in original order, got "
          f"{[h['chunk_id'] for h in got]}")


def test_cap_zero_is_plain_slice():
    """cap=0 means no cap: hits[:k], and the input list/dicts are untouched."""
    hits = [c("a1", "A"), c("b1", "B"), c("c1", "C")]
    import copy
    before = copy.deepcopy(hits)

    got = cap_per_doc(hits, k=2, cap=0)
    check([h["chunk_id"] for h in got] == ["a1", "b1"],
          f"cap=0 should be a plain hits[:k], got {[h['chunk_id'] for h in got]}")
    check(hits == before, f"cap_per_doc must not mutate the input list, got {hits}")


def test_select_hits_defaults_are_identity():
    """Defaults (read_k=0, cap=0) reproduce today's behaviour exactly: the
    full cached list, unchanged. read_k alone is a plain slice."""
    hits = [c("a1", "A"), c("b1", "B"), c("c1", "C"), c("d1", "D"), c("e1", "E"),
            c("f1", "F")]

    got_default = eval_answers.select_hits(hits, 0, 0)
    check(got_default is hits, f"select_hits(h, 0, 0) should be h unchanged, got a copy")

    got_sliced = eval_answers.select_hits(hits, 5, 0)
    check(got_sliced == hits[:5],
          f"select_hits(h, 5, 0) should equal h[:5], got "
          f"{[h['chunk_id'] for h in got_sliced]}")

    # Attempt 2 fix: read_k<=0 means "every cached hit" even when a cap is
    # also requested - cap_per_doc must be told to cap over the whole list
    # (len(h)), never over a bare read_k=0, which would silently empty the
    # result the model reads.
    got_capped_all = eval_answers.select_hits(hits, 0, 2)
    check(len(got_capped_all) == len(hits),
          f"select_hits(h, 0, 2) should keep all {len(hits)} hits "
          f"(capped/topped-up over the whole list), got {len(got_capped_all)}")
    check(len(got_capped_all) > 0,
          "select_hits(h, 0, 2) must never silently return an empty list")


def test_gate_ignores_selection():
    """Capping can change which chunk lands second in what the model reads,
    but gate_score(gate_hits) must be computed before selection ever runs,
    and selecting must never touch gate_hits itself."""
    gate_hits = [c("x1", "X", rerank_score=0.42), c("x2", "X", rerank_score=0.10)]
    hits = [c("x1", "X", rerank_score=0.42), c("x2", "X", rerank_score=0.40),
            c("y1", "Y", rerank_score=0.30)]

    score_before = gate_score(gate_hits)
    selected = eval_answers.select_hits(hits, 3, 2)  # cap=2 keeps both X's as-is
    check([h["chunk_id"] for h in selected] == ["x1", "x2", "y1"],
          f"sanity: uncapped-effective selection unchanged, got "
          f"{[h['chunk_id'] for h in selected]}")

    capped = eval_answers.select_hits(hits, 2, 1)  # cap=1 changes who is second
    check([h["chunk_id"] for h in capped] == ["x1", "y1"],
          f"capping should change the second hit read by the model, got "
          f"{[h['chunk_id'] for h in capped]}")

    check(gate_score(gate_hits) == score_before,
          "gate_score(gate_hits) must be unaffected by any selection over hits")
    check(gate_hits == [c("x1", "X", rerank_score=0.42), c("x2", "X", rerank_score=0.10)],
          f"selecting must never touch gate_hits, got {gate_hits}")


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
