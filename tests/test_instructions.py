"""Tests for phase 15 R14's three instruction switches (tsk_20261004_instruct): the
embedder's query instruction (--embed-task), the reader's system prompt
(--reader-prompt), and scripts/rerank_instruct.py's template swap. Defaults must stay
byte-identical to before; the non-default texts are pre-registered, so they are
pinned here verbatim.

Hermetic (blk_test_env_constraints): no pytest, no server, no data/, models/ or gguf.
`sqlite_vec` is stubbed in sys.modules before import, as tests/test_quota.py does;
the embedder, reranker, retriever and generator are stubs.
"""

from __future__ import annotations

import contextlib
import hashlib
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

from smm import embed, generate  # noqa: E402


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eval_answers = _load("eval_answers")
ask = _load("ask")
rerank_instruct = _load("rerank_instruct")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


LINUX = "Given a question about using a Linux system, retrieve the manual page passage that answers it"
NEUTRAL = ("Given a question about using the software on this computer, Linux commands and "
           "configuration or the GNU Emacs editor, retrieve the documentation passage that "
           "answers it")
V1 = """You answer questions about a Linux system using only the manual page extracts provided. Follow these rules exactly:

- Use only the extracts. Never use knowledge from outside them.
- If the extracts do not contain the answer, reply exactly: I don't know.
- Prefer naming the exact flag, option or setting, spelled as the manual spells it.
- Cite the extract number you used, like [2].
- Be brief. No preamble."""
V2 = """You answer questions about the software on this computer (Linux commands and configuration, and the GNU Emacs editor) using only the documentation extracts provided. The question may be misspelt, terse, or use everyday words instead of the documentation's own terms: answer what the user most plausibly means, if the extracts say it. Follow these rules exactly:

- Use only the extracts. Never use knowledge from outside them.
- If the extracts do not contain the answer, reply exactly: I don't know.
- Prefer naming the exact command, flag, option or setting, spelled as the documentation spells it.
- Cite the extract number you used, like [2].
- Be brief. No preamble."""
CHUNKS = [{"doc_id": "d.1", "prefix": "p ", "text": "t"}]


# --------------------------------------------------------------------------
# the embedder's query instruction
# --------------------------------------------------------------------------


def test_linux_task_unchanged():
    check(embed.TASK == LINUX, f"TASK text changed: {embed.TASK!r}")
    check(sha(embed.TASK) == "47f6cca90cf0", sha(embed.TASK))
    check(embed.TASKS["linux"] == embed.TASK,
          "TASKS['linux'] must be today's TASK")
    check(embed.query_text("q") == f"Instruct: {LINUX}\nQuery: q", embed.query_text("q"))
    check(set(embed.TASKS) == {"linux", "neutral"}, sorted(embed.TASKS))


def test_neutral_task_text():
    check(embed.TASKS["neutral"] == NEUTRAL, f"neutral text: {embed.TASKS['neutral']!r}")
    check(embed.query_text("q", embed.TASKS["neutral"]) == f"Instruct: {NEUTRAL}\nQuery: q",
          "query_text wraps the neutral task the same way")


def test_embedder_task_default_and_override():
    sent: list = []

    class Capture(embed.Embedder):
        def embed(self, texts):
            sent.extend(texts)
            return [[1.0]]

    Capture().embed_query("q")
    Capture(task=embed.TASKS["neutral"]).embed_query("q")
    Capture(task=embed.TASKS["neutral"]).embed_query("q", task="explicit")
    Capture().embed_query("q", task="explicit")
    check(sent == [f"Instruct: {LINUX}\nQuery: q", f"Instruct: {NEUTRAL}\nQuery: q",
                   "Instruct: explicit\nQuery: q", "Instruct: explicit\nQuery: q"], sent)
    check(embed.Embedder().task == embed.TASK, "the default task is today's TASK")


# --------------------------------------------------------------------------
# the reader's system prompt
# --------------------------------------------------------------------------


class _Chat(generate.Generator):
    """A Generator whose chat() records (messages, grammar) instead of calling a server."""

    def __init__(self):
        super().__init__()
        self.calls: list = []

    def chat(self, messages, max_tokens=400, temperature=0.0, grammar=None):
        self.calls.append((messages, grammar))
        return "stub"


def test_reader_v1_prompt_unchanged():
    check(generate.SYSTEM == V1, f"SYSTEM text changed: {generate.SYSTEM!r}")
    check(sha(generate.SYSTEM) == "0e6631f5facc", sha(generate.SYSTEM))
    m = generate.build_prompt("q?", CHUNKS)
    check(sha(repr(m)) == "c6e3f6c11e49", sha(repr(m)))
    check(m[1]["content"].startswith("Manual page extracts:\n\n"), m[1]["content"])
    g = _Chat()
    g.answer("q?", CHUNKS)
    g.answer("q?", CHUNKS, reader_prompt="v1")
    check(g.calls[0][0] == m and g.calls[1][0] == m, "v1 sends build_prompt's messages")
    g.answer("q?", CHUNKS, mode="quote")
    check(g.calls[2][0][0]["content"] == generate.QUOTE_SYSTEM, "quote mode is unchanged")


def test_reader_v2_prompt_text():
    check(generate.SYSTEM_V2 == V2, f"SYSTEM_V2 text: {generate.SYSTEM_V2!r}")
    g = _Chat()
    g.answer("q?", CHUNKS, cite_grammar=True, reader_prompt="v2")
    msgs, grammar = g.calls[0]
    check(msgs[0] == {"role": "system", "content": V2}, msgs[0])
    check(msgs[1]["content"] == "Documentation extracts:\n\n[1] d.1\np t\n\nQuestion: q?",
          msgs[1]["content"])
    check(grammar is not None, "cite grammar is still applied under v2")
    for bad in (dict(reader_prompt="v2", mode="quote"), dict(reader_prompt="v4")):
        try:
            g.answer("q?", CHUNKS, **bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} must raise ValueError")


# --------------------------------------------------------------------------
# scripts/eval_answers.py and scripts/ask.py
# --------------------------------------------------------------------------


class _Healthy:
    def __init__(self, *a, **kw):
        pass

    def health(self):
        return True


def _eval_run(extra_argv: list, stage: str = "both"):
    """Run eval_answers.main() over one row with stubs; return (exit code, answers json
    or None, Embedder kwargs, Generator.answer kwargs)."""
    emb_kw: list = []
    ans_kw: list = []

    class _Embedder(_Healthy):
        def __init__(self, *a, **kw):
            emb_kw.append(kw)

    class _Retriever:
        def __init__(self, db, embedder=None, reranker=None, mode=None,
                     candidates=None, domain=None):
            pass

        def retrieve(self, question, k=5):
            return [{"chunk_id": "c1", "doc_id": "c1", "text": "apt-get install installs it",
                     "score": 0.5, "rerank_score": 0.9}]

    class _Generator(_Healthy):
        def answer(self, question, chunks, cite_grammar=False, mode="cite", **kw):
            ans_kw.append(kw)
            return "apt-get install [1]"

    names = ("ROOT", "Embedder", "Reranker", "Retriever", "Generator", "store")
    old = {n: getattr(eval_answers, n) for n in names}
    old_argv = sys.argv
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            eval_answers.Embedder, eval_answers.Reranker = _Embedder, _Healthy
            eval_answers.Retriever, eval_answers.Generator = _Retriever, _Generator
            eval_answers.store = types.SimpleNamespace(connect=lambda path: "DB")
            eval_answers.ROOT = tmp
            (tmp / "eval.jsonl").write_text(json.dumps({
                "qid": "q1", "question": "how to install", "kind": "answerable", "tags": [],
                "answer_contains": ["apt-get install"]}) + "\n")
            sys.argv = ["eval_answers.py", "--stage", stage, "--eval", str(tmp / "eval.jsonl"),
                        "--name", "rt", "--db", "stub.db", "--no-aliases"] + extra_argv
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = eval_answers.main()
            answers = tmp / "data" / "eval" / "results" / "rt-answers.json"
            out = json.loads(answers.read_text()) if answers.exists() else None
    finally:
        sys.argv = old_argv
        for n, v in old.items():
            setattr(eval_answers, n, v)
    return code, out, emb_kw, ans_kw


def test_eval_answers_flags_default_off():
    code, out, emb_kw, ans_kw = _eval_run([])
    check(code == 0, f"exit {code}")
    check(emb_kw == [{}], f"default Embedder() takes no kwargs: {emb_kw}")
    check(ans_kw == [{}], f"default answer() takes no reader_prompt: {ans_kw}")
    check("embed_task" not in out["config"] and "reader_prompt" not in out["config"],
          f"default config must not record the switches: {out['config']}")

    code, out, emb_kw, ans_kw = _eval_run(["--embed-task", "neutral", "--reader-prompt", "v2"])
    check(code == 0, f"exit {code}")
    check(emb_kw == [{"task": NEUTRAL}], f"neutral task reaches the Embedder: {emb_kw}")
    check(ans_kw == [{"reader_prompt": "v2"}], f"v2 reaches answer(): {ans_kw}")
    check(out["config"].get("embed_task") == "neutral"
          and out["config"].get("reader_prompt") == "v2", out["config"])

    code, out, emb_kw, ans_kw = _eval_run(["--embed-task", "neutral"], stage="retrieve")
    check(code == 0 and out is None and emb_kw == [{"task": NEUTRAL}] and ans_kw == [],
          f"the embed task acts at the retrieve stage: {code} {emb_kw} {ans_kw}")

    code, out, emb_kw, ans_kw = _eval_run(["--reader-prompt", "v2", "--answer-mode", "quote"])
    check(code == 2 and out is None and not emb_kw and not ans_kw,
          f"v2 with --answer-mode quote must exit 2 before anything runs: {code}")


def _ask_run(argv: list, answer):
    """Run ask.main() with stubs; return (exit code, Embedder kwargs)."""
    emb_kw: list = []

    class _Embedder(_Healthy):
        def __init__(self, *a, **kw):
            emb_kw.append(kw)

    class _Generator(_Healthy):
        pass

    _Generator.answer = answer

    class _Retriever:
        def __init__(self, db, **kw):
            pass

        def retrieve(self, question, k=5):
            return [{"chunk_id": "c1", "doc_id": "d1", "text": "text", "score": 0.9,
                     "rerank_score": 0.9}]

    names = ("Embedder", "Reranker", "Retriever", "Generator", "store")
    old = {n: getattr(ask, n) for n in names}
    old_argv = sys.argv
    try:
        ask.Embedder, ask.Reranker = _Embedder, _Healthy
        ask.Retriever, ask.Generator = _Retriever, _Generator
        ask.store = types.SimpleNamespace(connect=lambda path: "DB")
        sys.argv = ["ask.py"] + argv + ["how to install"]
        with contextlib.redirect_stdout(io.StringIO()):
            code = ask.main()
    finally:
        sys.argv = old_argv
        for n, v in old.items():
            setattr(ask, n, v)
    return code, emb_kw


def test_ask_flags_pass_through():
    # Default: the stub's answer() has NO **kw, so any unconditional reader_prompt
    # kwarg would raise TypeError here.
    code, emb_kw = _ask_run([], lambda self, question, hits, cite_grammar=True: "a [1]")
    check(code == 0 and emb_kw == [{}], f"default ask passes nothing extra: {code} {emb_kw}")

    seen: list = []

    def answer(self, question, hits, cite_grammar=True, **kw):
        seen.append(kw)
        return "a [1]"

    code, emb_kw = _ask_run(["--embed-task", "neutral", "--reader-prompt", "v2"], answer)
    check(code == 0, f"exit {code}")
    check(emb_kw == [{"task": NEUTRAL}], f"neutral task reaches the Embedder: {emb_kw}")
    check(seen == [{"reader_prompt": "v2"}], f"v2 reaches answer(): {seen}")


# --------------------------------------------------------------------------
# scripts/rerank_instruct.py
# --------------------------------------------------------------------------

OLD = "Given a web search query, retrieve relevant passages that answer the query"
NEW = ("Given a question from a user of Linux command-line tools or the GNU Emacs editor, "
       "possibly misspelt or in everyday words, judge whether the documentation passage answers it")
TEMPLATE = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the "
            'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
            "<|im_end|>\n<|im_start|>user\n<Instruct>: " + OLD +
            "\n<Query>: {query}\n<Document>: {document}<|im_end|>\n<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n")


def test_rerank_template_swap():
    check(rerank_instruct.OLD_INSTRUCTION == OLD, rerank_instruct.OLD_INSTRUCTION)
    check(rerank_instruct.KEY == "tokenizer.chat_template.rerank", rerank_instruct.KEY)
    out = rerank_instruct.swap_instruction(TEMPLATE, NEW)
    check(out == TEMPLATE.replace(OLD, NEW), "only the instruction changes")
    check("<Instruct>: " + NEW + "\n<Query>: {query}" in out and OLD not in out, out)
    check(rerank_instruct.swap_instruction(TEMPLATE, OLD) == TEMPLATE, "swapping in the same text is identity")
    for bad in ("no instruction here", TEMPLATE + OLD):
        try:
            rerank_instruct.swap_instruction(bad, NEW)
        except ValueError:
            continue
        raise AssertionError("the old instruction not occurring exactly once must raise")

    # main() refuses (exit 2) when --out exists, before touching --in.
    with tempfile.TemporaryDirectory() as td:
        existing = Path(td) / "existing.out"
        existing.write_text("keep me")
        old_argv = sys.argv
        try:
            sys.argv = ["rerank_instruct.py", "--in", str(Path(td) / "absent.in"),
                        "--out", str(existing), "--instruction", NEW]
            with contextlib.redirect_stderr(io.StringIO()):
                code = rerank_instruct.main()
        finally:
            sys.argv = old_argv
        check(code == 2, f"existing --out must exit 2: {code}")
        check(existing.read_text() == "keep me", "the existing --out is untouched")


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
