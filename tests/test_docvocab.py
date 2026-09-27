"""Tests for phase 11 R4b (documentation-vocabulary rewrites) and R4a' (model
spelling correction) - tsk_20260927_docvocab.

Hermetic by design (blk_test_env_constraints): no pytest, no real
llama-server, no data/ or .venv. `scripts.eval_answers` imports `smm.store`,
which imports the third-party `sqlite_vec` (absent here); stubbed in
sys.modules before import, exactly as tests/test_eval_answers.py does.
Generator.correct's actual HTTP call is exercised against a loopback stub
HTTP server (stdlib http.server), the same house idiom as
tests/test_cli.py's stub llama-server. The eval-path test replaces
scripts.eval_answers's Embedder/Reranker/Retriever/store/Generator module
globals with in-process stubs and repoints its ROOT constant at a
tempfile.TemporaryDirectory(), so the run never touches this repo's own
data/eval/results/ (blk_env_masking_failure: no writes outside scope, even
temporarily).
"""

from __future__ import annotations

import http.server
import importlib.util
import json
import socket
import sys
import tempfile
import threading
import types
from pathlib import Path

if "sqlite_vec" not in sys.modules:
    stub = types.ModuleType("sqlite_vec")
    stub.load = lambda *a, **kw: None  # never called: no real db is opened here
    sys.modules["sqlite_vec"] = stub

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import generate  # noqa: E402
from smm.generate import Generator, build_correct_prompt, build_rewrite_prompt  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "eval_answers", ROOT / "scripts" / "eval_answers.py")
eval_answers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_answers)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# --------------------------------------------------------------------------
# 1-3: prompts, verbatim

def test_man_style_byte_identical_to_original_prompt():
    """build_rewrite_prompt(q, n, style='man') - and the no-style-arg call,
    since 'man' is the default - must be byte-identical to the prompt that
    existed before this task: the exact system string the acceptance check
    pins by sha256."""
    question = "how do I close a window in emacs"
    n = 3
    expected_system = (
        f"Rewrite the user's request as {n} alternative search queries for a "
        "Linux manual-page search engine. Each on its own line, no numbering, "
        "no explanation. Keep them short, name the likely command if you can, "
        "and vary the wording."
    )
    expected = [
        {"role": "system", "content": expected_system},
        {"role": "user", "content": question},
    ]
    check(build_rewrite_prompt(question, n, style="man") == expected,
          f"style='man' must equal today's prompt byte-for-byte, "
          f"got {build_rewrite_prompt(question, n, style='man')!r}")
    check(build_rewrite_prompt(question, n) == expected,
          "default style (no style arg) must also equal today's prompt")


def test_docs_style_verbatim_system_text():
    """style='docs' must carry the R4b spec's system text verbatim, with {n}
    filled in, and must keep the user message as the raw question (rewriter
    input is unchanged by style)."""
    question = "install a package"
    n = 4
    expected_system = (
        f"Rewrite the user's request as {n} alternative search queries for a "
        "search engine over software documentation (Linux manual pages and the "
        "GNU Emacs manuals). Use the terminology that documentation itself would "
        "use for the commands, options and concepts, which is often different "
        "from everyday words. Keep any program, package or command name the user "
        "wrote exactly as written. Each on its own line, no numbering, no "
        "explanation. Keep them short."
    )
    got = build_rewrite_prompt(question, n, style="docs")
    check(got[0]["content"] == expected_system,
          f"docs-style system text mismatch:\n  got:      {got[0]['content']!r}\n"
          f"  expected: {expected_system!r}")
    check(got[1] == {"role": "user", "content": question},
          f"docs-style user message must be the raw question unchanged, got {got[1]!r}")
    check(got[0]["content"] != build_rewrite_prompt(question, n, style="man")[0]["content"],
          "docs and man styles must not be the same prompt")


def test_correct_prompt_verbatim():
    """build_correct_prompt must carry the R4a' spec's system text verbatim,
    and the user message must be the raw question, unmodified."""
    question = "instal a packge with apt-get"
    expected_system = (
        "Fix spelling mistakes in the user's question. Change nothing else: keep "
        "the wording, and keep every program, package, command, option and file "
        "name exactly as written, even unfamiliar ones. Output only the corrected "
        "question on one line."
    )
    got = build_correct_prompt(question)
    check(got == [
        {"role": "system", "content": expected_system},
        {"role": "user", "content": question},
    ], f"build_correct_prompt mismatch, got {got!r}")


# --------------------------------------------------------------------------
# 4: Generator.correct's guard, against a loopback stub llama-server

class _ChatStub(http.server.BaseHTTPRequestHandler):
    """Stands in for llama-server's /v1/chat/completions: always 200s with a
    canned `content`, read from the class-level `next_content` (set by the
    test before each call). Never inspects the request body - Generator.chat
    already covers request shape elsewhere; this only has to drive the
    response side of Generator.correct's guard."""
    next_content = ""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps({
            "choices": [{"message": {"content": type(self).next_content}}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_generator_correct_guard():
    """Generator.correct: empty model output -> input unchanged; an output
    more than 2x the input's length plus 20 chars -> input unchanged (the
    model answered instead of correcting, blk_r4a_spell_normalisation_fails_rule's
    failure mode, generalised); a normal-length, different output -> that
    corrected text."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    httpd = http.server.HTTPServer(("127.0.0.1", port), _ChatStub)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        gen = Generator(url=f"http://127.0.0.1:{port}")
        question = "instal a packge"  # 15 chars

        _ChatStub.next_content = ""
        check(gen.correct(question) == question,
              "empty model output must fall back to the input unchanged")

        _ChatStub.next_content = "x" * (2 * len(question) + 21)
        check(gen.correct(question) == question,
              "over-long model output (> 2x input + 20 chars) must fall back "
              "to the input unchanged")

        _ChatStub.next_content = "install a package"
        check(gen.correct(question) == "install a package",
              "a normal-length, different correction must be returned as-is")
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


# --------------------------------------------------------------------------
# 5: eval_answers.py --llm-correct - same corrected text feeds retrieval AND
#    generation, and the generate stage reuses the retrieve stage's cached
#    text rather than asking the model again.

class _StubEmbedder:
    def health(self):
        return True


class _StubReranker:
    def health(self):
        return True


_HIT = {"chunk_id": "c1", "doc_id": "d1", "text": "apt-get install installs a package.",
        "score": 0.9, "rerank_score": 0.9}


def _make_stub_retriever(retrieve_calls: list):
    class _StubRetriever:
        def __init__(self, db, embedder=None, reranker=None, mode=None,
                     candidates=None, domain=None):
            pass

        def retrieve(self, question, k=5):
            retrieve_calls.append(question)
            return [dict(_HIT)]

        def retrieve_fused(self, question, rewrites=None, k=5,
                            variant_candidates=None, variant_k=None):
            retrieve_calls.append(question)
            return [dict(_HIT)], 0, [], [dict(_HIT)]

    return _StubRetriever


def _make_stub_generator(correct_calls: list, answer_calls: list):
    class _StubGenerator:
        def __init__(self, url=None, timeout=300):
            pass

        def health(self):
            return True

        def correct(self, question, max_tokens=120):
            correct_calls.append(question)
            return "install a package"  # the "corrected" text, always different

        def rewrites(self, question, n=1, max_tokens=120, style="man"):
            return []

        def answer(self, question, chunks, cite_grammar=False, mode="cite", **kw):
            answer_calls.append(question)
            return "apt-get install [1]"

    return _StubGenerator


def test_eval_llm_correct_reused_across_retrieve_and_generate_stages():
    retrieve_calls: list = []
    correct_calls: list = []
    answer_calls: list = []

    old = {
        name: getattr(eval_answers, name)
        for name in ("ROOT", "Embedder", "Reranker", "Retriever", "Generator", "store")
    }
    old_argv = sys.argv
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            eval_path = tmp / "eval.jsonl"
            eval_path.write_text(json.dumps({
                "qid": "q1", "kind": "answerable", "tags": [],
                "question": "instal a packge", "answer_contains": ["apt-get install"],
            }) + "\n")

            eval_answers.ROOT = tmp
            eval_answers.Embedder = _StubEmbedder
            eval_answers.Reranker = _StubReranker
            eval_answers.Retriever = _make_stub_retriever(retrieve_calls)
            eval_answers.Generator = _make_stub_generator(correct_calls, answer_calls)
            eval_answers.store = types.SimpleNamespace(connect=lambda path: "STUB_DB")

            common = ["--eval", str(eval_path), "--name", "docvocab-test",
                      "--db", "stub.db", "--no-aliases", "--llm-correct"]

            sys.argv = ["eval_answers.py", "--stage", "retrieve"] + common
            rc = eval_answers.main()
            check(rc == 0, f"retrieve stage must exit 0, got {rc}")
            check(correct_calls == ["instal a packge"],
                  f"correct() must be called once at retrieve stage with the raw "
                  f"question, got {correct_calls}")
            check(retrieve_calls == ["install a package"],
                  f"retrieval must be queried with the CORRECTED text, got {retrieve_calls}")

            sys.argv = ["eval_answers.py", "--stage", "generate"] + common
            rc = eval_answers.main()
            check(rc == 0, f"generate stage must exit 0, got {rc}")
            check(correct_calls == ["instal a packge"],
                  f"generate stage must NOT call correct() again - it must reuse "
                  f"the cached text - got {correct_calls}")
            check(answer_calls == ["install a package"],
                  f"generation prompt must use the SAME corrected text retrieval "
                  f"used, got {answer_calls}")

            out = json.loads((tmp / "data" / "eval" / "results" /
                               "docvocab-test-answers.json").read_text())
            check(out["config"].get("llm_correct") is True,
                  f"config must record llm_correct: true, got {out['config']}")
            check(out["results"][0]["question_corrected"] == "install a package",
                  f"result must record question_corrected, got {out['results'][0]}")
    finally:
        sys.argv = old_argv
        for name, val in old.items():
            setattr(eval_answers, name, val)


# --------------------------------------------------------------------------
# 6: defaults off leave the config and results keys exactly as before

def test_eval_defaults_off_unchanged_config_and_keys():
    """With neither --llm-correct nor --rewrite-style passed (both default
    off), the run's config dict and each result's keys must carry none of
    the new fields - the exact set that existed before this task."""
    retrieve_calls: list = []
    correct_calls: list = []
    answer_calls: list = []

    old = {
        name: getattr(eval_answers, name)
        for name in ("ROOT", "Embedder", "Reranker", "Retriever", "Generator", "store")
    }
    old_argv = sys.argv
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            eval_path = tmp / "eval.jsonl"
            eval_path.write_text(json.dumps({
                "qid": "q1", "kind": "answerable", "tags": [],
                "question": "how do I install a package", "answer_contains": ["apt-get install"],
            }) + "\n")

            eval_answers.ROOT = tmp
            eval_answers.Embedder = _StubEmbedder
            eval_answers.Reranker = _StubReranker
            eval_answers.Retriever = _make_stub_retriever(retrieve_calls)
            eval_answers.Generator = _make_stub_generator(correct_calls, answer_calls)
            eval_answers.store = types.SimpleNamespace(connect=lambda path: "STUB_DB")

            sys.argv = ["eval_answers.py", "--stage", "both", "--eval", str(eval_path),
                        "--name", "docvocab-default-test", "--db", "stub.db",
                        "--no-aliases"]
            rc = eval_answers.main()
            check(rc == 0, f"expected exit 0, got {rc}")
            check(correct_calls == [], "--llm-correct off must never call correct()")

            out = json.loads((tmp / "data" / "eval" / "results" /
                               "docvocab-default-test-answers.json").read_text())
            expected_config_keys = {
                "mode", "rerank", "gate", "grammar", "candidates", "expand", "db",
                "domain", "rewrites", "aliases", "cache", "read_k", "cap_per_doc",
                "answer_mode", "qids", "abstain_rule", "evidence_rule",
            }
            check(set(out["config"].keys()) == expected_config_keys,
                  f"defaults-off config must carry exactly the pre-existing keys, "
                  f"got {sorted(out['config'].keys())}")
            check("llm_correct" not in out["config"], "llm_correct must be absent by default")
            check("rewrite_style" not in out["config"], "rewrite_style must be absent by default")
            check("question_corrected" not in out["results"][0],
                  f"question_corrected must be absent from results by default, "
                  f"got {sorted(out['results'][0].keys())}")
    finally:
        sys.argv = old_argv
        for name, val in old.items():
            setattr(eval_answers, name, val)


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
