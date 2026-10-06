"""Tests for scripts/eval_answers.py --widen-spell (tsk_20261006_widenspell).

A question the vocabulary corrector changes gets its candidate pool widened
with the corrected question's candidates and is reranked ONCE against the
question as typed (Retriever.retrieve_widened); an unchanged question retrieves
exactly as before; the reader and the gate keep the typed question.

Hermetic (blk_test_env_constraints): no pytest, no server, no data/ or .venv.
`sqlite_vec` is stubbed before import, as tests/test_qvec.py does; Embedder,
Reranker, Retriever, Generator and the vocabulary build are stubs, and the
module's ROOT is pointed at a temp directory.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import types
from contextlib import redirect_stderr, redirect_stdout
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

from smm import normalize as real_qnorm  # noqa: E402

VOCAB = {"install": 50, "package": 40, "how": 100, "to": 200, "a": 100}
TYPED = "how to instal a pakage"
CORRECTED = "how to install a package"
CLEAN = "how to install a package"


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _hit(chunk_id: str, score: float) -> dict:
    return {"chunk_id": chunk_id, "doc_id": chunk_id, "text": "apt-get install installs it",
            "prefix": "", "rerank_score": score}


class StubRetriever:
    """Records every retrieval call; `widened_score` is the top-1 rerank score
    retrieve_widened returns (what the gate reads), `plain_score` retrieve's."""

    def __init__(self, widened_score=0.9, plain_score=0.9):
        self.widened_score, self.plain_score = widened_score, plain_score
        self.plain_calls: list = []
        self.widened_calls: list = []

    def retrieve(self, question, k=5):
        self.plain_calls.append((question, k))
        return [_hit("plain", self.plain_score)]

    def retrieve_widened(self, question, extra_queries, k=5):
        self.widened_calls.append((question, list(extra_queries), k))
        return [_hit("widened", self.widened_score)]


class _Healthy:
    def __init__(self, *a, **kw):
        pass

    def health(self):
        return True


class _Vocab:
    """Stand-in for eval_answers.qnorm: the real normalize(), a counted build_vocab."""

    def __init__(self, forbid_build=False):
        self.builds: list = []
        self.forbid_build = forbid_build
        self.normalize_calls: list = []

    def build_vocab(self, db_path, cache_path=None):
        if self.forbid_build:
            raise AssertionError("the vocabulary must not be built without --widen-spell")
        self.builds.append(db_path)
        return VOCAB

    def normalize(self, question, vocab):
        self.normalize_calls.append(question)
        return real_qnorm.normalize(question, vocab)

    def __getattr__(self, name):
        return getattr(real_qnorm, name)


def _run_main(questions: list, argv: list, retriever: StubRetriever, qnorm=None,
              stage: str = "both") -> dict:
    """Run eval_answers.main() on `questions` with stubs. Returns the answers
    file (stage both), the files written under results/, the exit code and
    what the reader was shown."""
    seen_questions: list = []

    class _Generator(_Healthy):
        def answer(self, question, chunks, cite_grammar=False, mode="cite", **kw):
            seen_questions.append(question)
            return "apt-get install [1]"

    names = ("ROOT", "Embedder", "Reranker", "Retriever", "Generator", "store", "qnorm")
    old = {n: getattr(eval_answers, n) for n in names}
    old_argv = sys.argv
    out: dict = {"seen": seen_questions}
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            eval_answers.Embedder = eval_answers.Reranker = _Healthy
            eval_answers.Retriever = lambda *a, **kw: retriever
            eval_answers.Generator = _Generator
            eval_answers.store = types.SimpleNamespace(connect=lambda path: "DB")
            eval_answers.qnorm = qnorm or _Vocab()
            eval_answers.ROOT = tmp
            (tmp / "eval.jsonl").write_text("".join(json.dumps({
                "qid": f"q{i}", "kind": "answerable", "tags": [], "question": q,
                "answer_contains": ["apt-get install"]}) + "\n"
                for i, q in enumerate(questions, 1)))
            sys.argv = ["eval_answers.py", "--stage", stage, "--eval", str(tmp / "eval.jsonl"),
                        "--name", "ws", "--db", "stub.db", "--no-aliases"] + argv
            with redirect_stdout(io.StringIO()):
                out["rc"] = eval_answers.main()
            results = tmp / "data" / "eval" / "results"
            out["files"] = {p.name: json.loads(p.read_text()) for p in results.glob("*.json")}
    finally:
        sys.argv = old_argv
        for n, v in old.items():
            setattr(eval_answers, n, v)
    return out


def test_changed_question_uses_retrieve_widened_with_corrected_text():
    r = StubRetriever()
    hits, info = eval_answers.retrieve_one(r, TYPED, 5, VOCAB)
    check(r.widened_calls == [(TYPED, [CORRECTED], 5)],
          f"widened pool must be the typed question plus the corrected one: {r.widened_calls}")
    check(r.plain_calls == [], f"a changed question must not take the plain path: {r.plain_calls}")
    check(hits[0]["chunk_id"] == "widened", hits)
    check(info["typed"] == TYPED and info["corrected"] == CORRECTED and info["edits"],
          f"audit record: {info}")


def test_unchanged_question_uses_plain_retrieve():
    r = StubRetriever()
    hits, info = eval_answers.retrieve_one(r, CLEAN, 5, VOCAB)
    check(r.plain_calls == [(CLEAN, 5)], r.plain_calls)
    check(r.widened_calls == [], f"an unchanged question must not widen: {r.widened_calls}")
    check(info is None and hits[0]["chunk_id"] == "plain", (hits, info))


def test_reader_and_gate_see_the_typed_question():
    out = _run_main([TYPED], ["--gate", "0.5", "--widen-spell"], StubRetriever(widened_score=0.9))
    check(out["rc"] == 0, out["rc"])
    check(out["seen"] == [TYPED], f"the reader must be shown the typed question: {out['seen']}")
    row = out["files"]["ws-answers.json"]["results"][0]
    check(row["question"] == TYPED and not row["gated"], row)
    # the gate reads the widened (typed-question) rerank score, not a separate list
    low = _run_main([TYPED], ["--gate", "0.5", "--widen-spell"], StubRetriever(widened_score=0.2))
    row = low["files"]["ws-answers.json"]["results"][0]
    check(row["gated"] and low["seen"] == [], f"a low widened top-1 must gate: {row}")
    check(row["top_score"] == 0.2, row)


def test_flag_off_calls_never_touch_vocabulary():
    r = StubRetriever()
    qn = _Vocab(forbid_build=True)
    out = _run_main([TYPED, CLEAN], [], r, qnorm=qn)
    check(out["rc"] == 0, out["rc"])
    check(qn.normalize_calls == [], f"normalize must not run with the flag off: {qn.normalize_calls}")
    check(r.widened_calls == [], r.widened_calls)
    check([c[0] for c in r.plain_calls] == [TYPED, CLEAN], r.plain_calls)
    check("ws-widened.json" not in out["files"], sorted(out["files"]))
    # helper level: no vocab means plain retrieve and no corrector call
    r2 = StubRetriever()
    check(eval_answers.retrieve_one(r2, TYPED, 5, None)[1] is None and r2.widened_calls == [],
          "vocab=None must be the plain path")


def test_config_key_only_when_on():
    off = _run_main([TYPED], [], StubRetriever())
    on = _run_main([TYPED], ["--widen-spell"], StubRetriever())
    check("widen_spell" not in off["files"]["ws-answers.json"]["config"],
          f"flag off must not add the key: {off['files']['ws-answers.json']['config']}")
    check(on["files"]["ws-answers.json"]["config"].get("widen_spell") is True,
          on["files"]["ws-answers.json"]["config"])


def test_rejects_incompatible_flags():
    for flags, named in ((["--normalize", "spell"], "--normalize"),
                         (["--normalize", "local"], "--normalize"),
                         (["--llm-correct"], "--llm-correct"),
                         (["--rewrites", "2"], "--rewrites"),
                         (["--route", "dense-vote"], "--route"),
                         (["--cascade"], "--cascade")):
        err = io.StringIO()
        with redirect_stderr(err):
            out = _run_main([TYPED], ["--widen-spell"] + flags, StubRetriever())
        check(out["rc"] == 2, f"{flags}: expected return 2, got {out['rc']}")
        msg = err.getvalue()
        check("--widen-spell" in msg and named in msg, f"{flags}: message must name both: {msg!r}")


def test_corrected_text_is_cached_for_audit():
    r = StubRetriever()
    out = _run_main([TYPED, CLEAN], ["--widen-spell"], r, stage="retrieve")
    check(out["rc"] == 0, out["rc"])
    audit = out["files"].get("ws-widened.json")
    check(audit is not None, f"no ws-widened.json written: {sorted(out['files'])}")
    check(list(audit) == ["q1"], f"only the widened row is recorded: {audit}")
    check(audit["q1"]["typed"] == TYPED and audit["q1"]["corrected"] == CORRECTED
          and audit["q1"]["edits"], audit)
    cached = out["files"]["ws-retrieved.json"]
    check(isinstance(cached["q1"], list) and cached["q1"][0]["chunk_id"] == "widened"
          and cached["q2"][0]["chunk_id"] == "plain",
          f"cache keeps the plain list shape: {cached}")


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
