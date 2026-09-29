"""Tests for phase 11 R8's question-vector route (tsk_20260929_qvec): the
Retriever's `question_vectors` pool widening, the --question-vectors flag on
scripts/ask.py and scripts/eval_answers.py, and scripts/build_qvec.py's
parsing and resume logic. See docs/phase11-results.md, "R8 pre-registration".

Hermetic (blk_test_env_constraints): no pytest, no server, no data/ or .venv.
`sqlite_vec` is stubbed in sys.modules before import, as tests/test_cascade.py
does; the store's search functions, embedder, reranker and generator are stubs.
The real vec0 tables are covered by tests/integration_qvec_store.py.
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

from smm import retrieve, store  # noqa: E402
from smm.retrieve import Retriever  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eval_answers = _load("eval_answers")
ask = _load("ask")
build_qvec = _load("build_qvec")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def hit(chunk_id: str, score: float = 0.5) -> dict:
    return {"chunk_id": chunk_id, "doc_id": chunk_id, "text": chunk_id, "score": score}


class StubEmbedder:
    def __init__(self):
        self.queries: list = []

    def embed_query(self, q):
        self.queries.append(q)
        return [1.0, 0.0]


class StubReranker:
    def __init__(self):
        self.calls: list = []

    def rerank(self, question, cands, top_k=5):
        self.calls.append((question, [c["chunk_id"] for c in cands]))
        return [dict(c, rerank_score=1.0 - i / 100) for i, c in enumerate(cands)][:top_k]


class PatchedStore:
    """Replace store.search / store.search_questions with recorders."""

    def __init__(self, chunk_ids, question_ids):
        self.chunk_ids, self.question_ids = chunk_ids, question_ids
        self.search_calls: list = []
        self.question_calls: list = []

    def __enter__(self):
        self.old = (store.search, store.search_questions)
        store.search = self._search
        store.search_questions = self._search_questions
        return self

    def __exit__(self, *a):
        store.search, store.search_questions = self.old

    def _search(self, db, vec, k=5, domain=None, **kw):
        self.search_calls.append((tuple(vec), k, domain))
        return [hit(c) for c in self.chunk_ids][:k]

    def _search_questions(self, db, vec, k=30, domain=None, **kw):
        self.question_calls.append((tuple(vec), k, domain))
        return [dict(hit(c), matched_question=f"q for {c}") for c in self.question_ids][:k]


# --------------------------------------------------------------------------
# smm.retrieve.Retriever
# --------------------------------------------------------------------------


def test_zero_question_vectors_is_todays_path():
    emb, rr = StubEmbedder(), StubReranker()
    with PatchedStore(["a", "b"], ["z"]) as ps:
        r = Retriever(None, embedder=emb, reranker=rr, mode="dense",
                      candidates=7, domain="emacs")
        cands = r.candidates_for("Q")
        hits = r.retrieve("Q", k=2)
    check(ps.search_calls == [((1.0, 0.0), 7, "emacs")] * 2, ps.search_calls)
    check(ps.question_calls == [], "M=0 must never call search_questions")
    check(all("route" not in c for c in cands), cands)
    check(all("route" not in h for h in hits), hits)
    check([c["chunk_id"] for c in cands] == ["a", "b"], cands)


def test_question_hits_added_after_chunk_candidates_deduped():
    emb = StubEmbedder()
    with PatchedStore(["a", "b"], ["z", "b", "y"]) as ps:
        r = Retriever(None, embedder=emb, mode="dense", candidates=50,
                      domain="emacs", question_vectors=30)
        cands = r.candidates_for("Q")
    check([c["chunk_id"] for c in cands] == ["a", "b", "z", "y"], cands)
    check(emb.queries == ["Q"], f"the query must be embedded once: {emb.queries}")
    check(ps.search_calls == [((1.0, 0.0), 50, "emacs")], ps.search_calls)
    check(ps.question_calls == [((1.0, 0.0), 30, "emacs")], ps.question_calls)


def test_rerank_against_original_question_over_union():
    rr = StubReranker()
    with PatchedStore(["a", "b"], ["z", "b"]):
        r = Retriever(None, embedder=StubEmbedder(), reranker=rr, mode="dense",
                      question_vectors=30)
        out = r.retrieve("the original question", k=3)
        widened = r.retrieve_widened("the original question", ["a rewrite"], k=3)
    check(rr.calls[0] == ("the original question", ["a", "b", "z"]), rr.calls[0])
    check(len(out) == 3, out)
    check(rr.calls[1][0] == "the original question" and rr.calls[1][1] == ["a", "b", "z"],
          f"widened path reranks the union, deduped, against the original: {rr.calls[1]}")
    check(len(widened) == 3, widened)


def test_route_tag_marks_question_only_chunks():
    with PatchedStore(["a", "b"], ["b", "z"]):
        for mode in ("dense", "hybrid"):
            r = Retriever(None, embedder=StubEmbedder(), mode=mode, question_vectors=5)
            if mode == "hybrid":
                old, retrieve.lexical.search = retrieve.lexical.search, (
                    lambda db, q, k=5, **kw: [hit("b"), hit("a")])
            try:
                routes = {c["chunk_id"]: c["route"] for c in r.candidates_for("Q")}
            finally:
                if mode == "hybrid":
                    retrieve.lexical.search = old
            check(routes == {"a": "chunk", "b": "chunk", "z": "question"}, (mode, routes))


# --------------------------------------------------------------------------
# scripts/eval_answers.py and scripts/ask.py
# --------------------------------------------------------------------------


class _Healthy:
    def __init__(self, *a, **kw):
        pass

    def health(self):
        return True


def _eval_config(extra_argv: list) -> tuple[dict, dict]:
    """Run eval_answers.main() on one row with stubs; return (config, Retriever kwargs)."""
    seen: dict = {}

    class _Retriever:
        def __init__(self, db, embedder=None, reranker=None, mode=None,
                     candidates=None, domain=None, **kw):
            seen.update(kw)

        def retrieve(self, question, k=5):
            return [dict(hit("c1", 0.9), text="apt-get install installs it",
                         rerank_score=0.9)]

    class _Generator(_Healthy):
        def answer(self, question, chunks, cite_grammar=False, mode="cite", **kw):
            return "apt-get install [1]"

    names = ("ROOT", "Embedder", "Reranker", "Retriever", "Generator", "store")
    old = {n: getattr(eval_answers, n) for n in names}
    old_argv = sys.argv
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            eval_answers.Embedder = eval_answers.Reranker = _Healthy
            eval_answers.Retriever, eval_answers.Generator = _Retriever, _Generator
            eval_answers.store = types.SimpleNamespace(connect=lambda path: "DB")
            eval_answers.ROOT = tmp
            (tmp / "eval.jsonl").write_text(json.dumps({
                "qid": "q1", "kind": "answerable", "tags": [], "question": "how to install",
                "answer_contains": ["apt-get install"]}) + "\n")
            sys.argv = ["eval_answers.py", "--stage", "both", "--eval", str(tmp / "eval.jsonl"),
                        "--name", "qv", "--db", "stub.db", "--no-aliases"] + extra_argv
            check(eval_answers.main() == 0, "eval_answers.main() must exit 0")
            out = json.loads((tmp / "data" / "eval" / "results" / "qv-answers.json").read_text())
    finally:
        sys.argv = old_argv
        for n, v in old.items():
            setattr(eval_answers, n, v)
    return out["config"], seen


def test_eval_config_key_only_when_nonzero():
    cfg0, kw0 = _eval_config([])
    check("question_vectors" not in cfg0, f"M=0 config must not gain the key: {cfg0}")
    check(kw0 == {}, f"M=0 must construct Retriever as before: {kw0}")
    cfg30, kw30 = _eval_config(["--question-vectors", "30"])
    check(cfg30.get("question_vectors") == 30, cfg30)
    check(kw30 == {"question_vectors": 30}, kw30)


def test_ask_flag_passes_through():
    seen: dict = {}

    class _Retriever:
        def __init__(self, db, **kw):
            seen.update(kw)

        def retrieve(self, question, k=5):
            return [dict(hit("c1", 0.9), rerank_score=0.9)]

    names = ("Embedder", "Reranker", "Retriever", "store")
    old = {n: getattr(ask, n) for n in names}
    old_argv = sys.argv
    try:
        ask.Embedder = ask.Reranker = _Healthy
        ask.Retriever = _Retriever
        ask.store = types.SimpleNamespace(connect=lambda path: "DB", has_qvec=lambda db: True)
        sys.argv = ["ask.py", "--retrieve-only", "--rewrites", "0",
                    "--question-vectors", "30", "how to install"]
        check(ask.main() == 0, "ask.main() must exit 0")
        check(seen.get("question_vectors") == 30, seen)
        ask.store = types.SimpleNamespace(connect=lambda path: "DB", has_qvec=lambda db: False)
        check(ask.main() == 2, "no qvec table must exit 2")
    finally:
        sys.argv = old_argv
        for n, v in old.items():
            setattr(ask, n, v)


# --------------------------------------------------------------------------
# scripts/build_qvec.py
# --------------------------------------------------------------------------


def test_build_parse_three_questions():
    text = "1. How do I undo?\n- How to go back a change\n\n  * Can I revert my edit\n4) extra one\n"
    qs = build_qvec.parse_questions(text, 3)
    check(qs == ["How do I undo?", "How to go back a change", "Can I revert my edit"], qs)
    check(build_qvec.parse_questions("", 3) == [], "empty text gives no questions")
    check(len(build_qvec.parse_questions("a\nb", 3)) == 2, "fewer lines than n is fine")
    check("three short questions" in build_qvec.system_prompt(3),
          "the default prompt keeps the pre-registered wording")


def test_build_resume_skips_done_chunks():
    asked: list = []

    class _Gen:
        def chat(self, messages, max_tokens=400, temperature=0.0, grammar=None):
            asked.append(messages[1]["content"])
            check(max_tokens == 120 and temperature == 0.0, (max_tokens, temperature))
            return "one?\ntwo?\nthree?\nfour?"

    chunks = [("c1", "text one"), ("c2", "text two"), ("c3", "text three")]
    cache = {"c1": ["already"]}
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "cache.json"
        n = build_qvec.generate(chunks, cache, path, _Gen(), per_chunk=3, parallel=2)
        check(n == 2, n)
        check(sorted(asked) == ["text three", "text two"], asked)
        check(cache["c1"] == ["already"] and cache["c2"] == ["one?", "two?", "three?"], cache)
        check(json.loads(path.read_text()) == cache, "cache must be written to disk")
        check(build_qvec.generate(chunks, cache, path, _Gen(), per_chunk=3) == 0
              and len(asked) == 2, "a second run asks nothing")


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
