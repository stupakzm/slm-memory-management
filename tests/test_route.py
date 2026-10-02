"""Tests for phase 13 R9's domain routing (tsk_20261002_route): the Retriever's
`route="dense-vote"`, the --route flag on scripts/eval_answers.py (dense-vote and
oracle) and scripts/ask.py. See docs/phase13-results.md.

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
from smm.retrieve import ROUTE_VOTE_K, Retriever  # noqa: E402


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
    """Replace store.search: an open search returns `open_domains` (one hit per
    entry, best first); a domain search returns hits of that domain."""

    def __init__(self, open_domains):
        self.open_domains = open_domains
        self.calls: list = []

    def __enter__(self):
        self.old = store.search
        store.search = self._search
        return self

    def __exit__(self, *a):
        store.search = self.old

    def _search(self, db, vec, k=5, domain=None, **kw):
        self.calls.append((tuple(vec), k, domain))
        if domain is None:
            return [hit(f"open{i}", d) for i, d in enumerate(self.open_domains)][:k]
        return [hit(f"{domain}{i}", domain) for i in range(3)][:k]


# --------------------------------------------------------------------------
# smm.retrieve.Retriever
# --------------------------------------------------------------------------


def test_dense_vote_routes_to_majority_domain():
    emb = StubEmbedder()
    # top-1 is linux but emacs holds 3 of the first 5 votes; the 6th hit is outside the vote
    open_domains = ["linux", "emacs", "emacs", "linux", "emacs", "linux", "linux"]
    with PatchedSearch(open_domains) as ps:
        r = Retriever(None, embedder=emb, mode="dense", candidates=9, route="dense-vote")
        check(r.last_route is None, "nothing routed yet")
        cands = r.candidates_for("Q")
        hits = r.retrieve("Q", k=2)
    check(ROUTE_VOTE_K == 5, ROUTE_VOTE_K)
    check(ps.calls[:2] == [((1.0, 0.0), 9, None), ((1.0, 0.0), 9, "emacs")], ps.calls)
    check([c["chunk_id"] for c in cands] == ["emacs0", "emacs1", "emacs2"], cands)
    check(r.last_route == "emacs", r.last_route)
    check(emb.queries == ["Q", "Q"] and len(ps.calls) == 4, "one embedding per candidates_for")
    check([h["chunk_id"] for h in hits] == ["emacs0", "emacs1"], hits)
    # the votes past the 5th hit do not count: linux would win 4-3 over all 7
    with PatchedSearch(["emacs"] * 5 + ["linux"] * 5):
        r2 = Retriever(None, embedder=StubEmbedder(), mode="dense", route="dense-vote")
        r2.candidates_for("Q")
    check(r2.last_route == "emacs", r2.last_route)


def test_dense_vote_tie_goes_to_top_ranked():
    # 3 domains over 5 votes: emacs 2, linux 2, other 1; the top-ranked hit is "other"
    with PatchedSearch(["other", "emacs", "linux", "emacs", "linux"]) as ps:
        r = Retriever(None, embedder=StubEmbedder(), mode="dense", route="dense-vote")
        r.candidates_for("Q")
    check(r.last_route == "other", r.last_route)
    check(ps.calls[1][2] == "other", ps.calls)
    # fewer than 5 hits can tie 1-1 as well
    with PatchedSearch(["emacs", "linux"]):
        r = Retriever(None, embedder=StubEmbedder(), mode="dense", route="dense-vote")
        r.candidates_for("Q")
    check(r.last_route == "emacs", r.last_route)
    with PatchedSearch(["linux", "emacs"]):
        r = Retriever(None, embedder=StubEmbedder(), mode="dense", route="dense-vote")
        r.candidates_for("Q")
    check(r.last_route == "linux", r.last_route)


def test_no_route_is_todays_path():
    emb = StubEmbedder()
    with PatchedSearch(["linux", "emacs"]) as ps:
        r = Retriever(None, embedder=emb, mode="dense", candidates=7, domain="emacs")
        cands = r.candidates_for("Q")
    check(ps.calls == [((1.0, 0.0), 7, "emacs")], ps.calls)
    check(emb.queries == ["Q"], emb.queries)
    check([c["chunk_id"] for c in cands] == ["emacs0", "emacs1", "emacs2"], cands)
    check(r.route is None and r.last_route is None, (r.route, r.last_route))


def test_route_rejects_incompatible_settings():
    def rejects(**kw):
        try:
            Retriever(None, embedder=StubEmbedder(), **kw)
        except ValueError:
            return True
        return False

    check(rejects(mode="dense", route="nonsense"), "unknown route")
    check(rejects(mode="dense", route="dense-vote", domain="emacs"), "route with domain")
    check(rejects(mode="dense", route="dense-vote", question_vectors=30),
          "route with question_vectors")
    check(rejects(mode="hybrid", route="dense-vote"), "route with hybrid")
    check(rejects(mode="bm25", route="dense-vote"), "route with bm25")
    check(not rejects(mode="dense", route="dense-vote"), "the plain combination is valid")
    check(not rejects(mode="hybrid", domain="emacs"), "no route: nothing is rejected")


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


def test_eval_oracle_uses_row_domain():
    code, out, files, seen, domains = _eval_run(["--route", "oracle"], ROWS)
    check(code == 0, f"exit {code}")
    check(domains == ["emacs", "linux", "emacs"],
          f"each row retrieves in its own domain, null meaning linux: {domains}")
    check(eval_answers.ORACLE_NULL_DOMAIN == "linux", eval_answers.ORACLE_NULL_DOMAIN)
    check(seen == {}, f"oracle constructs the Retriever as today: {seen}")
    check(files["rt-routes.json"] == {"q1": "emacs", "q2": "linux", "q3": "emacs"}, files)
    check(out["config"]["route"] == "oracle", out["config"])


def test_eval_route_config_key_only_when_set():
    code, out, files, seen, _ = _eval_run([], ROWS)
    check(code == 0 and "route" not in out["config"], f"unset config must not gain the key: {out}")
    check(seen == {} and "rt-routes.json" not in files, (seen, sorted(files)))
    code, out, files, seen, _ = _eval_run(["--route", "dense-vote"], ROWS)
    check(code == 0 and out["config"].get("route") == "dense-vote", out)
    check(seen == {"route": "dense-vote"}, f"dense-vote passes route to Retriever: {seen}")
    for bad in (["--domain", "emacs"], ["--rewrites", "1"], ["--cascade"],
                ["--question-vectors", "5"], ["--mode", "hybrid"], ["--mode", "bm25"]):
        code, out, *_ = _eval_run(["--route", "oracle"] + bad, ROWS)
        check(code == 2 and out is None, f"--route with {bad} must exit 2: {code}")


def test_eval_writes_routes_sidecar():
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code, out, files, seen, _ = _eval_run(["--route", "dense-vote"], ROWS, stage="retrieve")
    check(code == 0, f"exit {code}")
    # the stub routes by "emacs" in the question: q1 -> linux (wrong), q2/q3 -> emacs
    check(files["rt-routes.json"] == {"q1": "linux", "q2": "emacs", "q3": "emacs"}, files)
    text = buf.getvalue()
    check("routed correctly: 1 of 3" in text, text)  # q3 only: q1 true emacs, q2 true linux
    check("emacs ->" in text and "linux ->" in text, text)


def test_ask_route_flag_passes_through():
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
        sys.argv = ["ask.py", "--retrieve-only", "--rewrites", "0", "--route", "dense-vote",
                    "how to install"]
        check(ask.main() == 0, "ask.main() must exit 0")
        check(seen[-1].get("route") == "dense-vote" and seen[-1].get("domain") is None, seen)
        sys.argv = ["ask.py", "--retrieve-only", "--route", "dense-vote", "--domain", "emacs",
                    "how to install"]
        n = len(seen)
        check(ask.main() == 2 and len(seen) == n, "--route with --domain must exit 2")
        sys.argv = ["ask.py", "--retrieve-only", "--rewrites", "0", "how to install"]
        check(ask.main() == 0, "ask.main() must exit 0")
        check("route" not in seen[-1], f"no --route: Retriever is built as before: {seen[-1]}")
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
