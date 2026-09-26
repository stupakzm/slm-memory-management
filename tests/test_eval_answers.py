"""Tests for scripts/eval_answers.py's --rewrites cache plumbing
(tsk_20260926_7d90f5d1): the retrieval cache has to carry both the fused hits
(what the model reads) and gate_hits (the original question's own reranked
top-1, per blk_fusion_gate_semantic_slip) so the gate never thresholds the
fused list's own top-1 score.

Hermetic by design (blk_test_env_constraints): this worktree has no .venv, no
data/, no models/, and `scripts.eval_answers` imports `smm.store`, which
imports the third-party `sqlite_vec` (absent here). `sqlite_vec` is stubbed in
sys.modules before the import below, exactly as tests/test_retrieve.py does;
nothing here ever opens a real sqlite3 connection or calls into the stub, so
its contents don't matter, only its presence. The module is then loaded via
importlib so this file works with no changes to sys.path beyond that stub.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

if "sqlite_vec" not in sys.modules:
    stub = types.ModuleType("sqlite_vec")
    stub.load = lambda *a, **kw: None  # never called: no real db is opened here
    sys.modules["sqlite_vec"] = stub

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "eval_answers", ROOT / "scripts" / "eval_answers.py")
eval_answers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_answers)

from smm.retrieve import gate_score  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_gate_reads_original_question_hits():
    """A fused top-1 with a high rerank_score (0.99, the winning rewrite's own
    score) must not be what the gate thresholds - gate_hits (the original
    question's own reranked list, top-1 0.30) is what decides."""
    fused_hits = [{"chunk_id": "ls.1:4", "rerank_score": 0.99}]
    gate_hits = [{"chunk_id": "size.1:1", "rerank_score": 0.30},
                 {"chunk_id": "size.1:2", "rerank_score": 0.20}]
    entry = eval_answers.cache_entry(fused_hits, gate_hits, rewrites=1)

    hits, unpacked_gate_hits = eval_answers.unpack_entry(entry)

    check(hits == fused_hits, f"unpack_entry should return the fused hits: {hits}")
    check(unpacked_gate_hits == gate_hits,
          f"unpack_entry should return gate_hits, got {unpacked_gate_hits}")
    check(gate_score(unpacked_gate_hits) < 0.65,
          f"gate_score(gate_hits) should gate (0.30 < 0.65), got "
          f"{gate_score(unpacked_gate_hits)}")
    check(gate_score(hits) >= 0.65,
          f"gate_score(fused hits) would NOT have gated (0.99 >= 0.65) - "
          f"this is the semantic slip the gate must avoid, got {gate_score(hits)}")


def test_legacy_cache_entry_unpacks():
    """A plain list (an existing *-retrieved.json cache, written before
    --rewrites existed) must unpack to hits is gate_hits."""
    hits = [{"chunk_id": "a", "rerank_score": 0.9}]
    unpacked_hits, unpacked_gate_hits = eval_answers.unpack_entry(hits)

    check(unpacked_hits is hits or unpacked_hits == hits,
          f"legacy entry's hits should be (or equal) the original list: {unpacked_hits}")
    check(unpacked_gate_hits is hits or unpacked_gate_hits == hits,
          f"legacy entry's gate_hits should be (or equal) the same list: {unpacked_gate_hits}")
    check(unpacked_hits is unpacked_gate_hits or unpacked_hits == unpacked_gate_hits,
          "legacy cache entry must unpack to hits is gate_hits (or equal)")


def test_rewrites_zero_keeps_legacy_format():
    """--rewrites 0 must produce exactly today's format: the plain hits list,
    unchanged, JSON-identical to it - so existing runs reproduce byte-for-byte."""
    hits = [{"chunk_id": "x", "rerank_score": 0.5, "doc_id": "ls.1", "text": "..."}]
    entry = eval_answers.cache_entry(hits, hits, rewrites=0)

    check(entry == hits, f"rewrites=0 must return hits unchanged, got {entry}")
    check(isinstance(entry, list), f"rewrites=0 must return a plain list, got {type(entry)}")
    check(json.dumps(entry) == json.dumps(hits),
          "rewrites=0 cache_entry must be JSON-identical to the plain hits list")


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
