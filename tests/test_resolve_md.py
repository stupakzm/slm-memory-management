"""Tests for scripts/resolve_gold.py's markdown-corpus mode (tsk_20260927_emacsq:
grow the eval set with an Emacs usage-question set validated against the
converted info-manual markdown corpus, not man.jsonl).

Hermetic by design (blk_test_env_constraints): this worktree has no .venv, no
data/, no models/. `scripts.resolve_gold` imports `smm.ingest` lazily, inside
`load_md_corpus`, and `smm.ingest` is stdlib-only, so no stub is needed. The
module is loaded via importlib (as tests/test_gold.py loads
scripts/derive_gold_aliases.py) so this file works with no changes to
sys.path beyond the src/ insert below. Every corpus and eval file used here is
a tiny fixture written to a tempfile directory; nothing reads the real
data/corpus/ or data/eval/ trees.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "resolve_gold", ROOT / "scripts" / "resolve_gold.py")
resolve_gold = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(resolve_gold)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


def _answerable_row(qid: str, question: str, doc: str, toks: list[str]) -> dict:
    return {
        "qid": qid, "question": question, "kind": "answerable", "doc": doc,
        "answer_contains": toks, "gold_sec_ids": [], "tags": [],
        "paraphrase_of": None, "gold_primary": None,
    }


def _run(argv: list[str]) -> tuple[int, str]:
    """Run resolve_gold.main() with argv, capturing its exit code and stdout."""
    old_argv = sys.argv
    sys.argv = ["resolve_gold.py"] + argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = resolve_gold.main()
    finally:
        sys.argv = old_argv
    return rc, buf.getvalue()


def test_md_mode_gold_resolution():
    """A markdown corpus's '## <node>' heading becomes exactly one section, and
    a question whose answer token appears there resolves gold_sec_ids/
    gold_primary to that section - the same contract as man.jsonl mode, just
    loaded through smm.ingest.from_file instead of a jsonl line."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        md_dir = tmp / "corpus"
        md_dir.mkdir()
        _write(md_dir / "foo.md",
               "## Greeting\n\nSay `hello-world` to greet the user warmly.\n")
        eval_path = _write_jsonl(tmp / "eval.jsonl", [
            _answerable_row("t1", "Which command greets the user?",
                             "foo.testdom", ["hello-world"]),
        ])

        rc, out = _run(["--md-corpus", str(md_dir), "--domain", "testdom",
                         "--eval", str(eval_path), "--write"])
        check(rc == 0, f"expected exit 0, got {rc}\n{out}")
        check("ERROR" not in out, f"unexpected error in output:\n{out}")

        rows = [json.loads(l) for l in eval_path.open(encoding="utf-8")]
        check(rows[0]["gold_sec_ids"] == ["foo.testdom#greeting"],
              f"expected gold_sec_ids to resolve to the Greeting section, got {rows[0]}")
        check(rows[0]["gold_primary"] == "foo.testdom#greeting",
              f"expected gold_primary written, got {rows[0]}")


def test_md_mode_tool_not_installed_detail_in_text_errors():
    """md-mode's tool-not-installed check is stricter than man.jsonl mode's:
    it errors if unanswerable_detail appears anywhere in the md corpus's
    section text (case-insensitively), not only if it is a doc name - the
    Emacs FAQ names third-party packages in prose without them being docs."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        md_dir = tmp / "corpus"
        md_dir.mkdir()
        _write(md_dir / "bar.md",
               "## Notes\n\nThis mentions Widgetizer as a related tool.\n")
        eval_path = _write_jsonl(tmp / "eval.jsonl", [
            {"qid": "t2", "question": "How do I use widgetizer here?",
             "kind": "unanswerable", "doc": None, "answer_contains": [],
             "gold_sec_ids": [], "tags": ["tool-not-installed"],
             "unanswerable_reason": "tool-not-installed",
             "unanswerable_detail": "widgetizer", "paraphrase_of": None},
        ])

        rc, out = _run(["--md-corpus", str(md_dir), "--domain", "testdom",
                         "--eval", str(eval_path)])
        check(rc == 1, f"expected exit 1 (mentioned in prose), got {rc}\n{out}")
        check("  x " in out and "widgetizer" in out.lower(),
              f"expected an error line naming widgetizer, got:\n{out}")


def test_leak_warning():
    """A question that contains its own answer token warns but does not
    error - matching man.jsonl mode's leak check exactly."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        md_dir = tmp / "corpus"
        md_dir.mkdir()
        _write(md_dir / "foo.md",
               "## Greeting\n\nSay `hello-world` to greet the user warmly.\n")
        eval_path = _write_jsonl(tmp / "eval.jsonl", [
            _answerable_row("t3", "Which command runs hello-world to greet the user?",
                             "foo.testdom", ["hello-world"]),
        ])

        rc, out = _run(["--md-corpus", str(md_dir), "--domain", "testdom",
                         "--eval", str(eval_path)])
        check(rc == 0, f"a leak is a warning, not an error - expected exit 0, got {rc}\n{out}")
        check("  ! " in out and "leaks answer token" in out,
              f"expected a leak warning, got:\n{out}")


def test_default_corpus_and_eval_flags():
    """--corpus/--eval point resolve_gold at a fixture man.jsonl and a
    fixture questions.jsonl instead of the real ones - the same code path
    the CLI uses with no flags at all, just pointed elsewhere."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        corpus_path = tmp / "man.jsonl"
        with corpus_path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "doc_id": "tool.1", "name": "tool", "aliases": [],
                "sections": [{"sec_id": "tool.1#OPTIONS",
                               "text": "Use --frobnicate to enable frobnication."}],
            }) + "\n")
        eval_path = _write_jsonl(tmp / "questions.jsonl", [
            _answerable_row("t4", "Which flag enables frobnication?",
                             "tool.1", ["--frobnicate"]),
        ])

        rc, out = _run(["--corpus", str(corpus_path), "--eval", str(eval_path)])
        check(rc == 0, f"expected exit 0, got {rc}\n{out}")
        check("ERROR" not in out, f"unexpected error in output:\n{out}")
        check("gold resolved      : 1" in out, f"expected 1 resolved, got:\n{out}")


def test_write_touches_only_the_eval_file():
    """--write must rewrite the --eval file and nothing else: not the md
    corpus files it read from, and no stray file anywhere near them."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        md_dir = tmp / "corpus"
        md_dir.mkdir()
        md_file = _write(md_dir / "foo.md",
                          "## Greeting\n\nSay `hello-world` to greet the user warmly.\n")
        eval_path = _write_jsonl(tmp / "eval.jsonl", [
            _answerable_row("t5", "Which command greets the user?",
                             "foo.testdom", ["hello-world"]),
        ])

        before_md = md_file.read_bytes()
        before_listing = sorted(p.relative_to(tmp) for p in tmp.rglob("*"))

        rc, out = _run(["--md-corpus", str(md_dir), "--domain", "testdom",
                         "--eval", str(eval_path), "--write"])
        check(rc == 0, f"expected exit 0, got {rc}\n{out}")
        check("wrote resolved gold ids" in out, f"expected a write confirmation, got:\n{out}")

        after_md = md_file.read_bytes()
        after_listing = sorted(p.relative_to(tmp) for p in tmp.rglob("*"))
        check(before_md == after_md, "the md corpus file must not be modified by --write")
        check(before_listing == after_listing,
              f"--write must not create or remove files: before={before_listing} "
              f"after={after_listing}")

        rows = [json.loads(l) for l in eval_path.open(encoding="utf-8")]
        check(rows[0]["gold_sec_ids"] == ["foo.testdom#greeting"],
              f"expected the eval file itself to carry the resolved gold id, got {rows[0]}")


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
