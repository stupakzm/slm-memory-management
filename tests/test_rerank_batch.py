"""Tests for the reranker batch-independence check (tsk_20261003_rbatch): the client
`batch` of smm.rerank.Reranker, the --rerank-batch flag on scripts/eval_answers.py,
scripts/ask.py and scripts/quota_sweep.py, and scripts/rerank_check.py. See
docs/phase14-results.md.

Hermetic (blk_test_env_constraints): no pytest, no server, no data/ or .venv.
`sqlite_vec` is stubbed in sys.modules before import; HTTP, the store, the embedder
and every reranker are stubs.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
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

from smm.rerank import Reranker  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "scripts"))  # quota_sweep imports eval_answers by name
    spec.loader.exec_module(mod)
    return mod


eval_answers = _load("eval_answers")
ask = _load("ask")
quota_sweep = _load("quota_sweep")
rerank_check = _load("rerank_check")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def hit(chunk_id: str, domain: str = "man", distance: float = 0.5) -> dict:
    return {"chunk_id": chunk_id, "doc_id": chunk_id, "text": chunk_id, "score": 1 - distance,
            "distance": distance, "domain": domain}


# --------------------------------------------------------------------------
# smm.rerank.Reranker
# --------------------------------------------------------------------------


class _HttpReranker(Reranker):
    """The real Reranker with HTTP stubbed: records the document count of each request."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.request_sizes: list = []

    def _post(self, path, payload):
        self.request_sizes.append(len(payload["documents"]))
        return {"results": [{"index": i, "relevance_score": float(len(d))}
                            for i, d in enumerate(payload["documents"])]}


def _chunks(n: int) -> list:
    return [dict(hit(f"c{i}"), text="x" * (i + 1)) for i in range(n)]


def test_reranker_default_batch_unchanged():
    rr = _HttpReranker()
    check(rr.batch == 16, f"default client batch is 16: {rr.batch}")
    out = rr.rerank("q", _chunks(40))
    check(rr.request_sizes == [16, 16, 8], f"16 docs per request by default: {rr.request_sizes}")
    check([c["chunk_id"] for c in out][:2] == ["c39", "c38"], "best-first order unchanged")
    check(all(c["rerank_score"] == len(c["text"]) for c in out), "scores map to their chunks")


def test_reranker_batch_one_doc_per_request():
    rr = _HttpReranker(batch=1)
    out = rr.rerank("q", _chunks(5))
    check(rr.request_sizes == [1] * 5, f"one document per request: {rr.request_sizes}")
    check(len(out) == 5 and out[0]["chunk_id"] == "c4", "same result shape")
    # the per-call argument overrides the instance's; None means "use the instance's"
    rr = _HttpReranker()
    rr.rerank("q", _chunks(5), batch=1)
    check(rr.request_sizes == [1] * 5, rr.request_sizes)
    rr = _HttpReranker(batch=2)
    rr.rerank("q", _chunks(5), batch=None)
    check(rr.request_sizes == [2, 2, 1], rr.request_sizes)


# --------------------------------------------------------------------------
# --rerank-batch on eval_answers, ask, quota_sweep
# --------------------------------------------------------------------------


class _Healthy:
    """Embedder/Reranker stand-in that records the kwargs it was built with."""

    built: list = []

    def __init__(self, *a, **kw):
        _Healthy.built.append(kw)

    def health(self):
        return True

    def embed_query(self, q):
        return [1.0, 0.0]

    def rerank(self, q, chunks, **kw):
        return [dict(c, rerank_score=0.5) for c in chunks]


def _eval_run(extra_argv: list):
    _Healthy.built = []

    class _Retriever:
        def __init__(self, db, **kw):
            self.last_route = None

        def retrieve(self, question, k=5):
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
            (tmp / "eval.jsonl").write_text(json.dumps({
                "qid": "q1", "question": "how", "kind": "answerable", "tags": [],
                "answer_contains": ["apt-get install"]}) + "\n")
            sys.argv = ["eval_answers.py", "--eval", str(tmp / "eval.jsonl"), "--name", "rt",
                        "--db", "stub.db", "--no-aliases", "--rerank"] + extra_argv
            with contextlib.redirect_stdout(io.StringIO()):
                code = eval_answers.main()
            answers = tmp / "data" / "eval" / "results" / "rt-answers.json"
            out = json.loads(answers.read_text()) if answers.exists() else None
    finally:
        sys.argv = old_argv
        for n, v in old.items():
            setattr(eval_answers, n, v)
    return code, out, list(_Healthy.built)


def test_eval_rerank_batch_flag_passes_through():
    code, out, built = _eval_run(["--rerank-batch", "1"])
    check(code == 0, f"exit {code}")
    check(out["config"].get("rerank_batch") == 1, out["config"])
    check({"batch": 1} in built, f"Reranker built with batch=1: {built}")
    code, out, built = _eval_run([])
    check(code == 0 and "rerank_batch" not in out["config"], out["config"])
    check(all("batch" not in kw for kw in built), f"default passes no batch: {built}")


def test_ask_rerank_batch_flag_passes_through():
    built: list = []

    class _RR(_Healthy):
        def __init__(self, *a, **kw):
            built.append(kw)

    class _Retriever:
        def __init__(self, db, **kw):
            pass

        def retrieve(self, question, k=5):
            return [dict(hit("c1"), rerank_score=0.9)]

    names = ("Embedder", "Reranker", "Retriever", "store")
    old = {n: getattr(ask, n) for n in names}
    old_argv = sys.argv
    try:
        ask.Embedder, ask.Reranker, ask.Retriever = _Healthy, _RR, _Retriever
        ask.store = types.SimpleNamespace(connect=lambda path: "DB", has_qvec=lambda db: True)
        base = ["ask.py", "--retrieve-only", "--rewrites", "0"]
        sys.argv = base + ["--rerank-batch", "1", "how to install"]
        with contextlib.redirect_stdout(io.StringIO()):
            check(ask.main() == 0, "ask.main() must exit 0")
        check(built == [{"batch": 1}], f"Reranker built with batch=1: {built}")
        built.clear()
        sys.argv = base + ["how to install"]
        with contextlib.redirect_stdout(io.StringIO()):
            check(ask.main() == 0, "ask.main() must exit 0")
        check(built == [{}], f"default passes no batch: {built}")
    finally:
        sys.argv = old_argv
        for n, v in old.items():
            setattr(ask, n, v)


class _Store:
    """store.search / store.domains over two small domains."""

    @staticmethod
    def domains(db):
        return {"emacs": 10, "man": 10}

    @staticmethod
    def search(db, vec, k=5, domain=None, **kw):
        pool = {"man": [hit(f"m{i}", "man", 0.1 + 0.01 * i) for i in range(10)],
                "emacs": [hit(f"e{i}", "emacs", 0.5 + 0.01 * i) for i in range(10)]}
        if domain is not None:
            return pool[domain][:k]
        return sorted(pool["man"] + pool["emacs"], key=lambda h: h["distance"])[:k]


def test_sweep_rerank_batch_flag_passes_through():
    def run(extra):
        _Healthy.built = []
        names = ("ROOT", "Embedder", "Reranker", "store")
        old = {n: getattr(quota_sweep, n) for n in names}
        old_argv = sys.argv
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                quota_sweep.ROOT = tmp
                quota_sweep.Embedder = quota_sweep.Reranker = _Healthy
                quota_sweep.store = types.SimpleNamespace(
                    connect=lambda path: "DB", domains=_Store.domains, search=_Store.search)
                (tmp / "eval.jsonl").write_text(json.dumps({"qid": "q1", "question": "a"}) + "\n")
                sys.argv = ["quota_sweep.py", "--eval", str(tmp / "eval.jsonl"), "--db", "x.db",
                            "--name", "rt", "--candidates", "6", "--floors", "0,2"] + extra
                with contextlib.redirect_stdout(io.StringIO()):
                    code = quota_sweep.main()
        finally:
            sys.argv = old_argv
            for n, v in old.items():
                setattr(quota_sweep, n, v)
        return code, list(_Healthy.built)

    code, built = run(["--rerank-batch", "1"])
    check(code == 0 and {"batch": 1} in built, (code, built))
    code, built = run([])
    check(code == 0 and all("batch" not in kw for kw in built), (code, built))


# --------------------------------------------------------------------------
# scripts/rerank_check.py
# --------------------------------------------------------------------------


class _StubReranker(Reranker):
    """Real batching (Reranker.rerank) over a stubbed `scores`. `coupled` makes a score
    depend on its batch-mates; otherwise it depends on the document alone."""

    def __init__(self, coupled: bool, **kw):
        super().__init__(**kw)
        self.coupled = coupled
        self.calls: list = []  # (query, [documents]) per rerank() call

    def health(self):
        return True

    def rerank(self, query, chunks, top_k=None, batch=None):
        self.calls.append((query, [c["text"] for c in chunks]))
        return super().rerank(query, chunks, top_k, batch)

    def scores(self, query, documents):
        mates = sum(len(d) for d in documents) if self.coupled else 0
        return [len(d) * 0.1 + mates * 0.001 for d in documents]


def _check(coupled: bool, batch: int = 4, context_seed: int = 0):
    rr = _StubReranker(coupled, batch=batch)
    old = rerank_check.store
    rerank_check.store = types.SimpleNamespace(search=_Store.search, domains=_Store.domains)
    try:
        # texts of different lengths, so a document's own score is not its batch-mates'
        res = rerank_check.check_question("how", _Healthy(), rr, "DB", ["emacs", "man"],
                                          6, 4, context_seed)
    finally:
        rerank_check.store = old
    return res, rr


def test_rerank_check_detects_batch_dependence():
    res, _ = _check(coupled=True)
    check(res["drift"] > 0, f"a batch-mate-dependent reranker must show drift: {res}")
    res, _ = _check(coupled=False)
    check(res["drift"] == 0.0, f"an independent reranker has exactly 0 drift: {res}")
    # end to end: the printed line and the --out file
    for coupled in (False, True):
        old = {n: getattr(rerank_check, n) for n in ("ROOT", "Embedder", "Reranker", "store")}
        old_argv = sys.argv
        out = io.StringIO()
        try:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td)
                rerank_check.ROOT = tmp
                rerank_check.Embedder = _Healthy
                rerank_check.Reranker = lambda batch=16: _StubReranker(coupled, batch=batch)
                rerank_check.store = types.SimpleNamespace(
                    connect=lambda path: "DB", domains=_Store.domains, search=_Store.search)
                (tmp / "eval.jsonl").write_text("".join(
                    json.dumps({"qid": f"b{i:02d}", "question": f"q{i}"}) + "\n"
                    for i in range(8)))
                sys.argv = ["rerank_check.py", "--eval", "eval.jsonl", "--db", "x.db",
                            "--sample", "3", "--batch", "4", "--candidates", "6",
                            "--domain-k", "4", "--out", str(tmp / "o" / "r.json")]
                with contextlib.redirect_stdout(out):
                    code = rerank_check.main()
                detail = json.loads((tmp / "o" / "r.json").read_text())
        finally:
            sys.argv = old_argv
            for n, v in old.items():
                setattr(rerank_check, n, v)
        check(code == 0, code)
        line = out.getvalue().strip()
        check(line.startswith("batch 4: max_drift ") and " mean_rerank_s " in line
              and line.endswith(" n 3"), line)
        check(len(detail["questions"]) == 3 and detail["batch"] == 4, detail["batch"])
        check((float(line.split()[3]) > 0) == coupled, f"coupled={coupled}: {line}")


def test_rerank_check_contexts_share_pairs():
    res, rr = _check(coupled=False, context_seed=3)
    check(len(rr.calls) == 2, f"two rerank calls, one per context: {len(rr.calls)}")
    (q1, docs1), (q2, docs2) = rr.calls
    check(q1 == q2 == "how", "both contexts score the same question")
    # open top 6 are man; the union adds emacs e0-e3 and drops no one
    check(set(docs1) <= set(docs2), "every context-(i) chunk is also scored in context (ii)")
    check(len(docs1) == 6 and len(docs2) == 10 and len(set(docs2)) == 10, (docs1, docs2))
    check(res["n_open"] == 6 and res["n_union"] == 10 and res["n_shared"] == 6, res)
    check(docs2 != sorted(docs2), "context (ii) is shuffled")
    # the shuffle is seeded: same seed, same order; another seed, another order
    _, rr_same = _check(coupled=False, context_seed=3)
    check(rr_same.calls[1][1] == docs2, "same context seed -> same order")
    orders = {tuple(_check(coupled=False, context_seed=s)[1].calls[1][1]) for s in range(5)}
    check(len(orders) > 1, "different context seeds give different orders")


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
