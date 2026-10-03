"""Tests for phase 14 R11's floor quota (tsk_20261003_floor): `floor_pool`, the
Retriever's `route="floor"`, scripts/quota_sweep.py (the one-pass offline sweep and its
base-level dev/test split), and the --route floor / --floor flags on
scripts/eval_answers.py and scripts/ask.py. See docs/phase14-results.md.

Hermetic (blk_test_env_constraints): no pytest, no server, no data/ or .venv.
`sqlite_vec` is stubbed in sys.modules before import, as tests/test_qvec.py does;
the store's search and domains, the embedder, the reranker and the generator are stubs.
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

from smm import store  # noqa: E402
from smm.retrieve import Retriever, floor_pool  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "scripts"))  # quota_sweep imports eval_answers by name
    spec.loader.exec_module(mod)
    return mod


eval_answers = _load("eval_answers")
ask = _load("ask")
quota_sweep = _load("quota_sweep")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def hit(chunk_id: str, domain: str = "man", distance: float = 0.5) -> dict:
    return {"chunk_id": chunk_id, "doc_id": chunk_id, "text": chunk_id, "score": 1 - distance,
            "distance": distance, "domain": domain}


def ids(hits: list) -> list:
    return [h["chunk_id"] for h in hits]


# A two-domain corpus: every `man` chunk is nearer the query than any `emacs` chunk, so
# the open top-n is all man and only a floor brings emacs in.
def corpus() -> dict:
    return {"emacs": [hit(f"e{i}", "emacs", 0.50 + 0.01 * i) for i in range(100)],
            "man": [hit(f"m{i}", "man", 0.10 + 0.01 * i) for i in range(100)]}


class StubEmbedder:
    def __init__(self, *a, **kw):
        self.queries: list = []

    def embed_query(self, q):
        self.queries.append(q)
        return [1.0, 0.0]

    def health(self):
        return True


class PatchedSearch:
    """Replace store.search and store.domains over `corpus()`: a search returns the k
    nearest chunks of the requested domain (every domain when None)."""

    def __init__(self):
        self.data = corpus()
        self.calls: list = []
        self.domain_reads = 0

    def __enter__(self):
        self.old = (store.search, store.domains)
        store.search, store.domains = self.search, self.domains
        return self

    def __exit__(self, *a):
        store.search, store.domains = self.old

    def domains(self, db):
        self.domain_reads += 1
        return {d: len(v) for d, v in self.data.items()}

    def search(self, db, vec, k=5, domain=None, **kw):
        self.calls.append((tuple(vec), k, domain))
        pool = (self.data[domain] if domain is not None
                else sorted((h for v in self.data.values() for h in v),
                            key=lambda h: h["distance"]))
        return list(pool[:k])


# --------------------------------------------------------------------------
# smm.retrieve.floor_pool
# --------------------------------------------------------------------------


def test_floor_zero_is_open_top_n():
    open_hits = [hit(f"o{i}", "man", 0.1 + 0.01 * i) for i in range(10)]
    per_domain = {"emacs": [], "man": []}
    check(ids(floor_pool(open_hits, per_domain, 6, 0)) == ids(open_hits[:6]),
          "floor 0 is the open top-n")
    # per_domain lists that hold nothing nearer than the open top-n change nothing
    per_domain = {"emacs": [hit(f"e{i}", "emacs", 0.6 + 0.01 * i) for i in range(5)],
                  "man": open_hits[:5]}
    check(ids(floor_pool(open_hits, per_domain, 6, 0)) == ids(open_hits[:6]),
          "floor 0 with domain lists still reproduces the open top-n")
    check(floor_pool(open_hits, {}, 20, 0) == open_hits, "fewer than n chunks: all of them")


def test_floor_equal_split_matches_quota():
    with PatchedSearch():
        quota = Retriever(None, embedder=StubEmbedder(), mode="dense", candidates=50,
                          route="quota").candidates_for("Q")
        floor = Retriever(None, embedder=StubEmbedder(), mode="dense", candidates=50,
                          route="floor", floor=25).candidates_for("Q")
    check(len(floor) == 50 and len(set(ids(floor))) == 50, ids(floor))
    check(set(ids(floor)) == set(ids(quota)),
          f"floor 25 of 50 over two domains is the quota's chunk set: "
          f"{sorted(set(ids(floor)) ^ set(ids(quota)))}")
    check(set(ids(floor)) == {f"e{i}" for i in range(25)} | {f"m{i}" for i in range(25)},
          ids(floor))


def test_floor_fill_by_distance_dedupes():
    open_hits = [hit("a", "man", 0.10), hit("b", "emacs", 0.20), hit("c", "man", 0.30),
                 hit("d", "man", 0.40), hit("e", "emacs", 0.50)]
    per_domain = {"man": [hit("a", "man", 0.10), hit("c", "man", 0.30), hit("d", "man", 0.40)],
                  "emacs": [hit("b", "emacs", 0.20), hit("e", "emacs", 0.50),
                            hit("f", "emacs", 0.60), hit("g", "emacs", 0.70)]}
    # floor 2: emacs {b, e}, man {a, c}; the fill takes the nearest of the rest, d (0.40)
    pool = floor_pool(open_hits, per_domain, 5, 2)
    check(ids(pool) == ["b", "e", "a", "c", "d"], ids(pool))
    check(len(set(ids(pool))) == len(pool), "no duplicate chunk_id")
    # a fill candidate already taken as a floor chunk is not taken twice, and its slot
    # goes to the next one: floor 1 takes a and b, so c, d, e fill the other three
    check(ids(floor_pool(open_hits, per_domain, 5, 1)) == ["b", "a", "c", "d", "e"],
          ids(floor_pool(open_hits, per_domain, 5, 1)))
    # the fill reaches into a per-domain list past the open top-n
    check(ids(floor_pool(open_hits[:2], per_domain, 5, 1))[-3:] == ["c", "d", "e"],
          ids(floor_pool(open_hits[:2], per_domain, 5, 1)))
    # equal distances: the smaller chunk_id first
    tied = [hit("z", "man", 0.3), hit("y", "man", 0.3), hit("x", "man", 0.3)]
    check(ids(floor_pool(tied, {"man": []}, 2, 0)) == ["x", "y"], "ties break by chunk_id")
    # never more than n
    check(len(floor_pool(open_hits, per_domain, 3, 1)) == 3, "at most n chunks")


def test_floor_rejects_too_large():
    per_domain = {"emacs": [hit("e0", "emacs")], "man": [hit("m0")]}
    for n, floor in ((50, 26), (5, 3)):
        try:
            floor_pool([], per_domain, n, floor)
        except ValueError:
            continue
        raise AssertionError(f"2 domains x floor {floor} > {n} must raise ValueError")
    try:
        floor_pool([], per_domain, 50, -1)
    except ValueError:
        pass
    else:
        raise AssertionError("a negative floor must raise ValueError")
    check(len(floor_pool([], per_domain, 50, 25)) == 2, "2 x 25 == 50 is allowed")


def test_live_route_uses_floor_pool():
    emb = StubEmbedder()
    with PatchedSearch() as ps:
        r = Retriever(None, embedder=emb, mode="dense", candidates=20, route="floor", floor=5)
        cands = r.candidates_for("Q")
        r.candidates_for("Q2")
        n_domain_reads = ps.domain_reads
        calls = list(ps.calls[:3])
    check(calls == [((1.0, 0.0), 20, None), ((1.0, 0.0), 5, "emacs"),
                    ((1.0, 0.0), 5, "man")],
          f"open search first, then each domain in name order: {calls}")
    check(emb.queries == ["Q", "Q2"], "one embedding per query")
    check(n_domain_reads == 1, f"domain list is cached: {n_domain_reads}")
    check(set(ids(cands)) == {f"e{i}" for i in range(5)} | {f"m{i}" for i in range(15)},
          ids(cands))

    # floor 0 is today's path: exactly one open search, no domain lookup
    with PatchedSearch() as ps:
        r = Retriever(None, embedder=StubEmbedder(), mode="dense", candidates=20,
                      route="floor", floor=0)
        cands = r.candidates_for("Q")
        check(ps.calls == [((1.0, 0.0), 20, None)], ps.calls)
        check(ps.domain_reads == 0, "floor 0 never reads the domain list")
        check(ids(cands) == [f"m{i}" for i in range(20)], ids(cands))
        plain = Retriever(None, embedder=StubEmbedder(), mode="dense", candidates=20)
        check(ids(plain.candidates_for("Q")) == ids(cands), "floor 0 == the route=None path")

    # D * F > n is rejected when it is first used
    with PatchedSearch():
        r = Retriever(None, embedder=StubEmbedder(), mode="dense", candidates=20,
                      route="floor", floor=11)
        try:
            r.candidates_for("Q")
        except ValueError:
            pass
        else:
            raise AssertionError("2 domains x floor 11 > 20 must raise ValueError")

    def rejects(**kw):
        try:
            Retriever(None, embedder=StubEmbedder(), **kw)
        except ValueError:
            return True
        return False

    check(rejects(mode="dense", route="floor"), "route floor needs floor")
    check(rejects(mode="dense", floor=5), "floor needs route floor")
    check(rejects(mode="dense", route="quota", floor=5), "floor with another route")
    check(rejects(mode="dense", route="floor", floor=5, domain="emacs"), "floor with domain")
    check(rejects(mode="dense", route="floor", floor=5, question_vectors=30),
          "floor with question_vectors")
    check(rejects(mode="hybrid", route="floor", floor=5), "floor with hybrid")
    check(rejects(mode="bm25", route="floor", floor=5), "floor with bm25")
    check(not rejects(mode="dense", route="floor", floor=5), "the plain combination is valid")
    check(not rejects(mode="dense"), "no route and no floor is today's default")


# --------------------------------------------------------------------------
# scripts/quota_sweep.py
# --------------------------------------------------------------------------


class StubReranker:
    """Scores e0 highest of all, then every man chunk above every other emacs chunk."""

    calls: list = []

    def __init__(self, *a, **kw):
        pass

    def health(self):
        return True

    @staticmethod
    def score(chunk_id: str) -> float:
        i = int(chunk_id[1:])
        return 9.0 if chunk_id == "e0" else (5.0 - 0.1 * i if chunk_id[0] == "m" else 1.0 - 0.01 * i)

    def rerank(self, query, chunks, top_k=None, batch=16):
        StubReranker.calls.append((query, ids(chunks), top_k))
        out = [dict(c, rerank_score=self.score(c["chunk_id"])) for c in chunks]
        out.sort(key=lambda c: -c["rerank_score"])
        return out[:top_k] if top_k else out


def _sweep_run(extra_argv: list, rows: list, files: dict | None = None):
    """Run quota_sweep.main() over `rows` with every server and the store stubbed.
    Returns (exit code, stdout, results dir contents by name, domain/search calls)."""
    StubReranker.calls = []
    names = ("ROOT", "Embedder", "Reranker", "store")
    old = {n: getattr(quota_sweep, n) for n in names}
    old_argv = sys.argv
    out = io.StringIO()
    try:
        with tempfile.TemporaryDirectory() as td, PatchedSearch() as ps:
            tmp = Path(td)
            quota_sweep.ROOT = tmp
            quota_sweep.Embedder, quota_sweep.Reranker = StubEmbedder, StubReranker
            quota_sweep.store = types.SimpleNamespace(
                connect=lambda path: "DB", domains=store.domains, search=store.search)
            for name, body in (files or {"eval.jsonl": rows}).items():
                (tmp / name).write_text("".join(json.dumps(r) + "\n" for r in body))
            sys.argv = ["quota_sweep.py", "--eval", str(tmp / "eval.jsonl"),
                        "--db", "stub.db", "--name", "rt"] + extra_argv
            with contextlib.redirect_stdout(out):
                code = quota_sweep.main()
            resdir = tmp / "data" / "eval" / "results"
            written = ({p.name: json.loads(p.read_text()) for p in resdir.glob("*.json")}
                       if resdir.exists() else {})
    finally:
        sys.argv = old_argv
        for n, v in old.items():
            setattr(quota_sweep, n, v)
    return code, out.getvalue(), written, ps.calls


SWEEP_ROWS = [{"qid": "q1", "question": "first"}, {"qid": "q2", "question": "second"}]
SWEEP_ARGS = ["--candidates", "10", "--floors", "0,2,4", "-k", "3"]


def test_sweep_reranks_union_once():
    code, _, written, search_calls = _sweep_run(SWEEP_ARGS, SWEEP_ROWS)
    check(code == 0, f"exit {code}")
    check(len(StubReranker.calls) == 2, f"one rerank call per question: {StubReranker.calls}")
    for (query, chunk_ids, top_k), row in zip(StubReranker.calls, SWEEP_ROWS):
        check(query == row["question"], query)
        check(top_k is None, f"no top_k: the whole union is scored: {top_k}")
        check(len(chunk_ids) == len(set(chunk_ids)), "the union is deduped")
        # open top 10 (m0..m9) + emacs top 4 (e0..e3); man's own top 4 is inside the open
        check(set(chunk_ids) == {f"m{i}" for i in range(10)} | {f"e{i}" for i in range(4)},
              chunk_ids)
    check(sorted(written) == ["rt-f0-retrieved.json", "rt-f2-retrieved.json",
                              "rt-f4-retrieved.json"], sorted(written))
    # per question: the open search, then each domain's own at the largest floor
    check(search_calls[:3] == [((1.0, 0.0), 10, None), ((1.0, 0.0), 4, "emacs"),
                               ((1.0, 0.0), 4, "man")], search_calls[:3])
    check(len(search_calls) == 6, f"no further searches per floor: {len(search_calls)}")


def test_sweep_topk_from_cached_scores():
    _, _, written, _ = _sweep_run(SWEEP_ARGS, SWEEP_ROWS)
    # F=0: the open top-10 is m0..m9 -> the three best by score are m0 m1 m2
    check(ids(written["rt-f0-retrieved.json"]["q1"]) == ["m0", "m1", "m2"],
          ids(written["rt-f0-retrieved.json"]["q1"]))
    # F=2: e0, e1 join (floor), e0 scores 9 -> first; m0 m1 follow
    check(ids(written["rt-f2-retrieved.json"]["q1"]) == ["e0", "m0", "m1"],
          ids(written["rt-f2-retrieved.json"]["q1"]))
    check(ids(written["rt-f4-retrieved.json"]["q2"]) == ["e0", "m0", "m1"],
          ids(written["rt-f4-retrieved.json"]["q2"]))
    for f in (0, 2, 4):
        for hits in written[f"rt-f{f}-retrieved.json"].values():
            scores = [h["rerank_score"] for h in hits]
            check(scores == sorted(scores, reverse=True) and len(hits) == 3, scores)
            check(all(h["rerank_score"] == StubReranker.score(h["chunk_id"]) for h in hits),
                  "scores are the cached rerank scores")
    # a pool smaller than the union: F=0 never sees e0, whatever it scored
    check("e0" not in ids(written["rt-f0-retrieved.json"]["q1"]), "F=0 is the open pool")


def test_sweep_cache_format_matches_eval():
    code, _, written, _ = _sweep_run(SWEEP_ARGS, SWEEP_ROWS)
    check(code == 0, code)
    for name, cache in written.items():
        for qid, entry in cache.items():
            hits, gate_hits = eval_answers.unpack_entry(entry)
            check(entry == eval_answers.cache_entry(hits, hits, 0),
                  f"{name}/{qid}: the entry is cache_entry(hits, hits, 0)")
            check(isinstance(entry, list) and gate_hits == hits, f"{name}/{qid}: plain list")
            check(eval_answers.correct_text(entry) is None, "no corrected question")
            check(all({"chunk_id", "text", "rerank_score"} <= set(h) for h in hits),
                  f"{name}/{qid}: hits carry what the generate stage reads")
    # --split writes the qids sidecar eval_answers --qids reads
    rows = [{"qid": f"b{i:02d}", "question": f"q{i}"} for i in range(6)]
    code, _, written, _ = _sweep_run(SWEEP_ARGS + ["--split", "test"], rows)
    check(code == 0, code)
    qids = written["rt-qids.json"]
    check(len(qids) == 3 and set(qids) == set(written["rt-f0-retrieved.json"]), qids)


def test_split_is_base_level_and_seeded():
    rows = []
    for i in range(20):
        base = f"b{i:02d}"
        rows.append({"qid": base, "question": base})
        rows.append({"qid": f"{base}v1", "question": base, "variant_of": base})
        rows.append({"qid": f"{base}v2", "question": base, "variant_of": f"{base}v1"})
        rows.append({"qid": f"{base}p", "question": base, "paraphrase_of": base})
    # variants of a base that is not in the eval set group under that missing qid
    rows += [{"qid": "x1", "question": "x", "variant_of": "gone"},
             {"qid": "x2", "question": "x", "variant_of": "gone"}]
    by_qid = {r["qid"]: r for r in rows}
    check(quota_sweep.base_of("b03v2", by_qid) == "b03", "variant of a variant -> root")
    check(quota_sweep.base_of("b03p", by_qid) == "b03", "paraphrase -> root")
    check(quota_sweep.base_of("x1", by_qid) == quota_sweep.base_of("x2", by_qid) == "gone",
          "missing base")

    dev, test = quota_sweep.split_rows(rows, 14)
    check(len(dev) + len(test) == len(rows) and not {r["qid"] for r in dev} & {r["qid"] for r in test},
          "a partition")
    dev_bases = {quota_sweep.base_of(r["qid"], by_qid) for r in dev}
    test_bases = {quota_sweep.base_of(r["qid"], by_qid) for r in test}
    check(not dev_bases & test_bases, "no base on both sides")
    check(len(dev_bases) == 10 and len(test_bases) == 11, (len(dev_bases), len(test_bases)))
    for base in dev_bases | test_bases:
        group = {r["qid"] for r in rows if quota_sweep.base_of(r["qid"], by_qid) == base}
        side = {r["qid"] for r in (dev if base in dev_bases else test)}
        check(group <= side, f"every variant follows its base: {base}")

    check(quota_sweep.split_rows(rows, 14) == (dev, test), "the same seed gives the same split")
    others = [quota_sweep.split_rows(rows, s)[0] for s in range(15, 25)]
    check(any(o != dev for o in others), "a different seed changes the split")

    # --print-split: exactly two lines, no servers (none are stubbed healthy here)
    code, out, written, _ = _sweep_run(["--print-split"], rows)
    check(code == 0 and written == {}, (code, written))
    check(out == f"dev: bases 10 rows {len(dev)}\ntest: bases 11 rows {len(test)}\n", out)
    # a qid in two --eval files is an error
    dup = [{"qid": "d1", "question": "a"}]
    code, _, written, _ = _sweep_run(["--eval", "eval2.jsonl"], dup,
                                     files={"eval.jsonl": dup, "eval2.jsonl": dup})
    check(code == 2 and written == {}, f"a duplicate qid must exit 2: {code}")
    code, _, written, _ = _sweep_run(["--eval", "eval2.jsonl"], dup,
                                     files={"eval.jsonl": dup, "eval2.jsonl": [
                                         {"qid": "d2", "question": "b"}]})
    check(code == 0 and set(written["rt-f0-retrieved.json"]) == {"d1", "d2"},
          "rows from several --eval files are merged")


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
    or None, results dir contents by name, Retriever kwargs)."""
    seen: dict = {}

    class _Retriever:
        def __init__(self, db, embedder=None, reranker=None, mode=None,
                     candidates=None, domain=None, **kw):
            seen.update(kw)
            self.domain = domain
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
    return code, out, files, seen


ROWS = [{"qid": "q1", "question": "how to install", "domain": "emacs"},
        {"qid": "q2", "question": "emacs buffers", "domain": None}]


def test_eval_and_ask_floor_flags_pass_through():
    code, out, files, seen = _eval_run(["--route", "floor", "--floor", "10"], ROWS)
    check(code == 0, f"exit {code}")
    check(seen == {"route": "floor", "floor": 10}, f"route and floor reach Retriever: {seen}")
    check(out["config"].get("route") == "floor" and out["config"].get("floor") == 10,
          out["config"])
    check("rt-routes.json" not in files, f"no routes sidecar for floor: {sorted(files)}")
    code, out, files, seen = _eval_run(["--route", "floor", "--floor", "0"], ROWS)
    check(code == 0 and seen == {"route": "floor", "floor": 0}, (code, seen))
    code, out, files, seen = _eval_run([], ROWS)
    check(code == 0 and seen == {} and "floor" not in out["config"], (seen, out["config"]))
    code, out, *_ = _eval_run(["--route", "floor"], ROWS)
    check(code == 2 and out is None, f"--route floor without --floor must exit 2: {code}")
    code, out, *_ = _eval_run(["--floor", "10"], ROWS)
    check(code == 2 and out is None, f"--floor without --route floor must exit 2: {code}")
    code, out, *_ = _eval_run(["--route", "quota", "--floor", "10"], ROWS)
    check(code == 2 and out is None, f"--floor with another route must exit 2: {code}")
    code, out, *_ = _eval_run(["--route", "floor", "--floor", "10", "--domain", "emacs"], ROWS)
    check(code == 2 and out is None, f"--route floor with --domain must exit 2: {code}")

    seen_ask: list = []

    class _Retriever:
        def __init__(self, db, **kw):
            seen_ask.append(kw)

        def retrieve(self, question, k=5):
            return [dict(hit("c1"), rerank_score=0.9)]

    names = ("Embedder", "Reranker", "Retriever", "store")
    old = {n: getattr(ask, n) for n in names}
    old_argv = sys.argv
    try:
        ask.Embedder = ask.Reranker = _Healthy
        ask.Retriever = _Retriever
        ask.store = types.SimpleNamespace(connect=lambda path: "DB", has_qvec=lambda db: True)
        sys.argv = ["ask.py", "--retrieve-only", "--rewrites", "0", "--route", "floor",
                    "--floor", "15", "how to install"]
        check(ask.main() == 0, "ask.main() must exit 0")
        check(seen_ask[-1].get("route") == "floor" and seen_ask[-1].get("floor") == 15
              and seen_ask[-1].get("domain") is None, seen_ask)
        sys.argv = ["ask.py", "--retrieve-only", "--rewrites", "0", "how to install"]
        check(ask.main() == 0 and "floor" not in seen_ask[-1], f"default passes no floor: {seen_ask[-1]}")
        n = len(seen_ask)
        for argv in (["--route", "floor"], ["--floor", "5"], ["--route", "quota", "--floor", "5"],
                     ["--route", "floor", "--floor", "5", "--domain", "emacs"]):
            sys.argv = ["ask.py", "--retrieve-only"] + argv + ["how to install"]
            check(ask.main() == 2 and len(seen_ask) == n, f"{argv} must exit 2")
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
