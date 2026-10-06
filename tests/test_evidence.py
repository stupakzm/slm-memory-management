"""Tests for src/smm/evidence.py and its two print sites in scripts/ask.py.
Hermetic: no servers, no GPU, no index; stubs as in tests/test_ask_defaults.py.
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

from smm import evidence, grammar  # noqa: E402

_spec = importlib.util.spec_from_file_location("ask", ROOT / "scripts" / "ask.py")
ask = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ask)

STATE: dict = {}


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


TAR = {"chunk_id": "t", "doc_id": "tar.1", "score": 0.9, "rerank_score": 0.9,
       "text": "Intro line\n  -z, --gzip  filter the archive through gzip\nlast"}
EMACS = {"chunk_id": "e", "doc_id": "emacs.emacs", "score": 0.9, "rerank_score": 0.9,
         "sec_id": "emacs.emacs#Saving", "text": "Saving files\nC-x C-s saves the buffer\nM-x save-buffer too"}


def test_finds_line_naming_a_flag():
    ev = evidence.evidence_lines("Use --gzip [1].", [TAR])
    check(len(ev) == 1 and ev[0]["n"] == 1, ev)
    check(ev[0]["line"] == "-z, --gzip  filter the archive through gzip", ev)
    ev = evidence.evidence_lines("Use -z [1].", [TAR])
    check(ev and "-z" in ev[0]["line"], ev)


def test_finds_line_naming_an_emacs_key_or_mx_command():
    ev = evidence.evidence_lines("Press C-x C-s [1].", [EMACS])
    check(ev and ev[0]["line"] == "C-x C-s saves the buffer", ev)
    ev = evidence.evidence_lines("Run M-x save-buffer. [1]", [EMACS])
    check(ev and "save-buffer" in ev[0]["line"], ev)


def test_no_identifier_in_answer_gives_no_evidence():
    check(evidence.evidence_lines("It compresses files [1].", [TAR]) == [], "got evidence")


def test_only_cited_extracts_are_searched():
    other = {"doc_id": "x.1", "text": "--gzip appears here too"}
    ev = evidence.evidence_lines("Use --gzip [2].", [other, TAR])
    check([e["n"] for e in ev] == [2], ev)
    check(evidence.evidence_lines("Use --gzip [3].", [TAR, TAR]) == [], "out of range")
    check(evidence.evidence_lines("Use --gzip.", [TAR]) == [], "no citation")


def test_man_pointer_from_doc_id():
    check(evidence.pointer_for({"doc_id": "tar.1"}) == "man 1 tar", "tar.1")
    check(evidence.pointer_for({"doc_id": "nl.1"}) == "man 1 nl", "nl.1")
    check(evidence.pointer_for({"doc_id": "foo.3p"}) == "man 3p foo", "foo.3p")


def test_info_pointer_from_sec_id():
    check(evidence.pointer_for(EMACS) == "Info manual emacs, section Saving", "sec_id")
    check(evidence.pointer_for({"doc_id": "elisp.elisp"})
          == "Info manual elisp, section elisp.elisp", "no sec_id")


def test_refusal_has_no_evidence():
    check(evidence.evidence_lines(grammar.REFUSAL, [TAR]) == [], "refusal")


class StubGenerator:
    def __init__(self, *a, **kw):
        pass

    def health(self):
        return True

    def answer(self, question, hits, cite_grammar=True):
        return STATE["answer"]


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
        return [dict(TAR, rerank_score=STATE["score"])]


class StubStore:
    @staticmethod
    def connect(path):
        return object()


def _run(argv, answer, score=0.9) -> str:
    STATE.update(answer=answer, score=score)
    saved = (ask.Generator, ask.Embedder, ask.Reranker, ask.Retriever, ask.store,
             sys.argv)
    ask.Generator, ask.Embedder, ask.Reranker = StubGenerator, StubEmbedder, StubReranker
    ask.Retriever, ask.store = StubRetriever, StubStore
    sys.argv = ["ask.py"] + argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = ask.main()
    finally:
        (ask.Generator, ask.Embedder, ask.Reranker, ask.Retriever, ask.store,
         sys.argv) = saved
    check(rc == 0, f"rc {rc}")
    return buf.getvalue()


def test_ask_prints_evidence_after_sources():
    out = _run(["how", "to", "gzip"], "Use --gzip [1].")
    want = ("sources: [1] tar.1\n"
            "evidence [1]: -z, --gzip  filter the archive through gzip\n"
            "read more: man 1 tar\n")
    check(out.endswith(want), repr(out))


def test_ask_no_evidence_flag_suppresses_it():
    out = _run(["--no-evidence", "how to gzip"], "Use --gzip [1].")
    check("evidence [" not in out and "read more" not in out, repr(out))
    check(out.endswith("sources: [1] tar.1\n"), repr(out))


def test_ask_refusal_by_gate_prints_no_evidence():
    out = _run(["how to gzip"], "Use --gzip [1].", score=0.1)
    check("evidence [" not in out and "read more" not in out, repr(out))
    check(grammar.REFUSAL in out, repr(out))
    out = _run(["how to gzip"], grammar.REFUSAL)
    check("evidence [" not in out and "read more" not in out, repr(out))


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
