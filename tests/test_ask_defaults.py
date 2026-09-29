"""Tests for scripts/ask.py's --rewrites default (phase 11 R4b: the default
single rewrite lost 47/600 answers vs plain retrieval, so the default is 0 in
every mode). Hermetic: no servers, no GPU, no index. sqlite_vec is stubbed as in
tests/test_cascade.py; Generator/Embedder/Reranker/Retriever/store are stubs
monkeypatched onto the imported scripts/ask.py module.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import types
from pathlib import Path

if "sqlite_vec" not in sys.modules:
    stub = types.ModuleType("sqlite_vec")
    stub.load = lambda *a, **kw: None
    sys.modules["sqlite_vec"] = stub

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location("ask", ROOT / "scripts" / "ask.py")
ask = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ask)

CALLS: dict = {}


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _hit() -> dict:
    return {"chunk_id": "c1", "doc_id": "d1", "text": "text", "score": 0.9,
            "rerank_score": 0.9}  # above GATE (0.65)


class StubGenerator:
    def __init__(self, *a, **kw):
        CALLS["generator_constructed"] = CALLS.get("generator_constructed", 0) + 1

    def health(self):
        return True

    def rewrites(self, question, n=1, style="man"):
        CALLS.setdefault("rewrites", []).append(n)
        return ["a rewrite"] * n

    def answer(self, question, hits, cite_grammar=True):
        return "stub answer [1]"


class StubEmbedder:
    def health(self):
        return True


class StubReranker:
    def health(self):
        return True


class StubRetriever:
    def __init__(self, *a, **kw):
        pass

    def retrieve(self, question, k=5):
        CALLS["retrieve"] = CALLS.get("retrieve", 0) + 1
        return [_hit()]

    def retrieve_fused(self, question, rewrites, k=5):
        CALLS["retrieve_fused"] = CALLS.get("retrieve_fused", 0) + 1
        return [_hit()], 0, [question] + list(rewrites), [_hit()]


class StubStore:
    @staticmethod
    def connect(path):
        return object()


def _run(argv: list) -> int:
    CALLS.clear()
    saved = (ask.Generator, ask.Embedder, ask.Reranker, ask.Retriever, ask.store,
             sys.argv)
    ask.Generator, ask.Embedder, ask.Reranker = StubGenerator, StubEmbedder, StubReranker
    ask.Retriever, ask.store = StubRetriever, StubStore
    sys.argv = ["ask.py"] + argv
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return ask.main()
    finally:
        (ask.Generator, ask.Embedder, ask.Reranker, ask.Retriever, ask.store,
         sys.argv) = saved


def test_default_makes_no_rewrite_call():
    rc = _run(["how", "do", "I", "list", "files"])
    check(rc == 0, f"rc {rc}")
    check("rewrites" not in CALLS, f"rewrites called: {CALLS}")
    check("retrieve_fused" not in CALLS, f"retrieve_fused used: {CALLS}")
    check(CALLS.get("retrieve") == 1, f"plain retrieve not used once: {CALLS}")


def test_explicit_rewrites_still_fuse():
    rc = _run(["--rewrites", "1", "how do I list files"])
    check(rc == 0, f"rc {rc}")
    check(CALLS.get("rewrites") == [1], f"rewrites calls: {CALLS.get('rewrites')}")
    check(CALLS.get("retrieve_fused") == 1, f"retrieve_fused not used: {CALLS}")
    check("retrieve" not in CALLS, f"plain retrieve used: {CALLS}")


def test_retrieve_only_default_generator_free():
    rc = _run(["--retrieve-only", "how do I list files"])
    check(rc == 0, f"rc {rc}")
    check("generator_constructed" not in CALLS, f"Generator constructed: {CALLS}")
    check("rewrites" not in CALLS, f"rewrites called: {CALLS}")


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
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
