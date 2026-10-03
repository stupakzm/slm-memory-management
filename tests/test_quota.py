"""Tests for phase 13 R10's per-domain candidate quotas (tsk_20261003_quota): the
Retriever's `route="quota"`, and the --route quota flag on scripts/eval_answers.py
and scripts/ask.py. See docs/phase13-results.md.

Hermetic (blk_test_env_constraints): no pytest, no server, no data/ or .venv.
`sqlite_vec` is stubbed in sys.modules before import, as tests/test_qvec.py does;
the store's search, embedder, reranker and generator are stubs.
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

from smm import store  # noqa: E402
from smm.retrieve import Retriever  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eval_answers = _load("eval_answers")
ask = _load("ask")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def hit(chunk_id: str, domain: str = "linux", score: float = 0.5) -> dict:
    return {"chunk_id": chunk_id, "doc_id": chunk_id, "text": chunk_id, "score": score,
            "domain": domain}


class StubEmbedder:
    def __init__(self):
        self.queries: list = []

    def embed_query(self, q):
        self.queries.append(q)
        return [1.0, 0.0]


class PatchedSearch:
    """Replace store.search and store.domains: a search returns k hits of the
    requested domain (every domain when None); `domains` is {name: count}."""

    def __init__(self, domains):
        self.domains = domains
        self.calls: list = []
        self.domain_reads = 0

    def __enter__(self):
        self.old = (store.search, store.domains)
        store.search, store.domains = self._search, self._domains
        return self

    def __exit__(self, *a):
        store.search, store.domains = self.old

    def _domains(self, db):
        self.domain_reads += 1
        return dict(self.domains)

    def _search(self, db, vec, k=5, domain=None, **kw):
        self.calls.append((tuple(vec), k, domain))
        return [hit(f"{domain}{i}", domain) for i in range(k)]


# --------------------------------------------------------------------------
# smm.retrieve.Retriever
# --------------------------------------------------------------------------


def test_quota_splits_candidates_evenly():
    emb = StubEmbedder()
    with PatchedSearch({"man": 10, "emacs": 5}) as ps:
        r = Retriever(None, embedder=emb, mode="dense", candidates=50, route="quota")
        cands = r.candidates_for("Q")
    check(ps.calls == [((1.0, 0.0), 25, "emacs"), ((1.0, 0.0), 25, "man")], ps.calls)
    ids = [c["chunk_id"] for c in cands]
    check(len(ids) == 50, len(ids))
    check(ids == [f"emacs{i}" for i in range(25)] + [f"man{i}" for i in range(25)], ids)
    check(emb.queries == ["Q"], "one embedding per query")


def test_quota_remainder_goes_to_first_domains():
    with PatchedSearch({"man": 1, "emacs": 1, "linux": 1}) as ps:
        r = Retriever(None, embedder=StubEmbedder(), mode="dense", candidates=50, route="quota")
        cands = r.candidates_for("Q")
    check([(c[2], c[1]) for c in ps.calls] == [("emacs", 17), ("linux", 17), ("man", 16)],
          ps.calls)
    check(len(cands) == 50, len(cands))


def test_quota_one_search_per_domain_no_vote():
    with PatchedSearch({"man": 1, "emacs": 1}) as ps:
        r = Retriever(None, embedder=StubEmbedder(), mode="dense", candidates=10, route="quota")
        r.candidates_for("Q")
        r.candidates_for("Q2")
        hits = r.retrieve("Q3", k=2)
    check(all(c[2] is not None for c in ps.calls), f"no open search: {ps.calls}")
    check(len(ps.calls) >= 6 and len({c[2] for c in ps.calls}) == 2, ps.calls)
    check(r.last_route is None, r.last_route)
    check(ps.domain_reads == 1, f"domain list is cached: {ps.domain_reads}")
    check(len(hits) == 2, hits)


def test_quota_rejects_incompatible_settings():
    def rejects(**kw):
        try:
            Retriever(None, embedder=StubEmbedder(), **kw)
        except ValueError:
            return True
        return False

    check(rejects(mode="dense", route="quota", domain="emacs"), "quota with domain")
    check(rejects(mode="dense", route="quota", question_vectors=30), "quota with question_vectors")
    check(rejects(mode="hybrid", route="quota"), "quota with hybrid")
    check(rejects(mode="bm25", route="quota"), "quota with bm25")
    check(not rejects(mode="dense", route="quota"), "the plain combination is valid")


# --------------------------------------------------------------------------
# scripts/eval_answers.py and scripts/ask.py
# --------------------------------------------------------------------------


class _Healthy:
    def __init__(self, *a, **kw):
        pass

    def health(self):
        return True


def _eval_run(extra_argv: list, rows: list, stage: str = "both"):
    """Run eval_answers.main() with stubs over `rows`; return (exit code, answers json
    or None, results dir contents by name, Retriever kwargs, domains seen at retrieve)."""
    seen: dict = {}
    domains: list = []

    class _Retriever:
        def __init__(self, db, embedder=None, reranker=None, mode=None,
                     candidates=None, domain=None, **kw):
            seen.update(kw)
            self.domain = domain
            self.last_route = None

        def retrieve(self, question, k=5):
            domains.append(self.domain)
            if seen.get("route"):  # stand-in router: "emacs" in the question
                self.last_route = "emacs" if "emacs" in question else "linux"
            return [dict(hit("c1"), text="apt-get install installs it", rerank_score=0.9)]

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
            (tmp / "eval.jsonl").write_text("".join(json.dumps(dict({
                "kind": "answerable", "tags": [], "answer_contains": ["apt-get install"]},
                **row)) + "\n" for row in rows))
            sys.argv = ["eval_answers.py", "--stage", stage, "--eval", str(tmp / "eval.jsonl"),
                        "--name", "rt", "--db", "stub.db", "--no-aliases"] + extra_argv
            code = eval_answers.main()
            resdir = tmp / "data" / "eval" / "results"
            answers = resdir / "rt-answers.json"
            out = json.loads(answers.read_text()) if answers.exists() else None
            files = ({p.name: json.loads(p.read_text()) for p in resdir.glob("*.json")}
                     if resdir.exists() else {})
    finally:
        sys.argv = old_argv
        for n, v in old.items():
            setattr(eval_answers, n, v)
    return code, out, files, seen, domains


ROWS = [{"qid": "q1", "question": "how to install", "domain": "emacs"},
        {"qid": "q2", "question": "emacs buffers", "domain": None},
        {"qid": "q3", "question": "emacs windows", "domain": "emacs"}]


def test_eval_quota_route_passes_through():
    code, out, files, seen, _ = _eval_run(["--route", "quota"], ROWS, stage="both")
    check(code == 0, f"exit {code}")
    check(seen == {"route": "quota"}, f"quota passes route to Retriever: {seen}")
    check(out["config"].get("route") == "quota", out["config"])
    check("rt-routes.json" not in files, f"no routes sidecar for quota: {sorted(files)}")
    code, out, *_ = _eval_run(["--route", "quota", "--domain", "emacs"], ROWS)
    check(code == 2 and out is None, f"--route quota with --domain must exit 2: {code}")


def test_ask_quota_flag_passes_through():
    seen: list = []

    class _Retriever:
        def __init__(self, db, **kw):
            seen.append(kw)

        def retrieve(self, question, k=5):
            return [dict(hit("c1"), rerank_score=0.9)]

    names = ("Embedder", "Reranker", "Retriever", "store")
    old = {n: getattr(ask, n) for n in names}
    old_argv = sys.argv
    try:
        ask.Embedder = ask.Reranker = _Healthy
        ask.Retriever = _Retriever
        ask.store = types.SimpleNamespace(connect=lambda path: "DB", has_qvec=lambda db: True)
        sys.argv = ["ask.py", "--retrieve-only", "--rewrites", "0", "--route", "quota",
                    "how to install"]
        check(ask.main() == 0, "ask.main() must exit 0")
        check(seen[-1].get("route") == "quota" and seen[-1].get("domain") is None, seen)
        sys.argv = ["ask.py", "--retrieve-only", "--route", "quota", "--domain", "emacs",
                    "how to install"]
        n = len(seen)
        check(ask.main() == 2 and len(seen) == n, "--route with --domain must exit 2")
    finally:
        sys.argv = old_argv
        for n, v in old.items():
            setattr(ask, n, v)


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
