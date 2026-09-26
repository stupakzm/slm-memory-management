"""Tests for smm.gold and scripts/derive_gold_aliases.py (tsk_20260926_a51d0707:
gold aliases). Gold tokens in questions.jsonl are mostly long-form options
(`--no-clobber`); a correct answer using the documented short form (`-n`) was
scored wrong under plain substring matching. These tests check the three
derivation rules (synonym, argument, description) against a tiny synthetic
corpus, the alias-boundary regex in smm.gold, and that disabling aliases
reproduces strict scoring exactly.

Hermetic by design (blk_test_env_constraints): this worktree has no .venv, no
data/, no models/. smm.gold is stdlib-only and needs no stub.
scripts/derive_gold_aliases.py imports nothing beyond stdlib either, but is
loaded via importlib (as tests/test_eval_answers.py loads
scripts/eval_answers.py) so this file works with no changes to sys.path
beyond the src/ insert below. `sqlite_vec` is stubbed first anyway, matching
the repo's test convention, even though nothing here opens a database.
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

from smm import gold  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "derive_gold_aliases", ROOT / "scripts" / "derive_gold_aliases.py")
derive_gold_aliases = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(derive_gold_aliases)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _write_corpus(tmp: Path, docs: list[dict]) -> Path:
    path = tmp / "man.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d) + "\n")
    return path


def _write_questions(tmp: Path, rows: list[dict]) -> Path:
    path = tmp / "questions.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


def _cp_doc() -> dict:
    """A synthetic cp(1)-like doc: a synonym pair (-n/--no-clobber) and a
    second option whose long form carries an '=FILE' argument, to check that
    the argument suffix is stripped before the synonym match."""
    text = (
        "-n, --no-clobber\n"
        "       (deprecated) silently skip existing files\n"
        "\n"
        "-X, --exclude-from=FILE\n"
        "       skip files matching patterns from FILE\n"
    )
    return {
        "doc_id": "cp.1", "name": "cp", "section": "1", "path": "/cp.1",
        "title": "cp", "summary": "", "aliases": [], "see_also": [],
        "sections": [{"sec_id": "cp.1#OPTIONS", "heading": "OPTIONS",
                       "level": 1, "parent": None, "text": text}],
    }


def _scp_doc() -> dict:
    """A synthetic scp(1)-like doc: an option whose argument placeholder
    (identity_file) is itself a plausible gold token."""
    text = (
        "-i identity_file\n"
        "       selects the file from which the identity for public key\n"
        "       authentication is read\n"
    )
    return {
        "doc_id": "scp.1", "name": "scp", "section": "1", "path": "/scp.1",
        "title": "scp", "summary": "", "aliases": [], "see_also": [],
        "sections": [{"sec_id": "scp.1#OPTIONS", "heading": "OPTIONS",
                       "level": 1, "parent": None, "text": text}],
    }


def _ls_doc() -> dict:
    """A synthetic ls(1)-like doc: a bare flag whose description contains a
    phrase a question might use as its gold token."""
    text = (
        "-t\n"
        "       sort by time, newest first\n"
    )
    return {
        "doc_id": "ls.1", "name": "ls", "section": "1", "path": "/ls.1",
        "title": "ls", "summary": "", "aliases": [], "see_also": [],
        "sections": [{"sec_id": "ls.1#OPTIONS", "heading": "OPTIONS",
                       "level": 1, "parent": None, "text": text}],
    }


def test_synonym_rule():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        corpus = _write_corpus(tmp, [_cp_doc()])
        qs = _write_questions(tmp, [
            {"qid": "a1", "kind": "answerable", "doc": "cp.1",
             "answer_contains": ["--no-clobber", "--exclude-from"]},
        ])
        aliases, stats = derive_gold_aliases.derive(corpus, qs)

        got = aliases["a1"]["--no-clobber"]
        check(any(e["alias"] == "-n" and e["rule"] == "synonym" for e in got),
              f"--no-clobber should get alias -n via synonym, got {got}")

        got2 = aliases["a1"]["--exclude-from"]
        check(any(e["alias"] == "-X" and e["rule"] == "synonym" for e in got2),
              f"--exclude-from=FILE should strip =FILE and alias to -X, got {got2}")
        check(stats["per_rule"]["synonym"] >= 2, f"expected >=2 synonym matches: {stats}")


def test_argument_rule():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        corpus = _write_corpus(tmp, [_scp_doc()])
        qs = _write_questions(tmp, [
            {"qid": "a2", "kind": "answerable", "doc": "scp.1",
             "answer_contains": ["identity_file"]},
        ])
        aliases, _stats = derive_gold_aliases.derive(corpus, qs)

        got = aliases["a2"]["identity_file"]
        check(any(e["alias"] == "-i" and e["rule"] == "argument" for e in got),
              f"identity_file should get alias -i via argument rule, got {got}")


def test_description_rule():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        corpus = _write_corpus(tmp, [_ls_doc()])
        qs = _write_questions(tmp, [
            {"qid": "a3", "kind": "answerable", "doc": "ls.1",
             "answer_contains": ["sort by time"]},
        ])
        aliases, _stats = derive_gold_aliases.derive(corpus, qs)

        got = aliases["a3"]["sort by time"]
        check(any(e["alias"] == "-t" and e["rule"] == "description" for e in got),
              f"'sort by time' should get alias -t via description rule, got {got}")


def test_alias_boundary():
    aliases = [{"alias": "-r", "rule": "synonym", "sec_id": "x.1#OPTIONS",
                "line": "-r, --recursive"}]
    token = "placeholder-token-not-in-any-text"

    check(not gold.token_ok("descend with --recursive please", token, aliases),
          "-r must not match inside --recursive")
    check(not gold.token_ok("run with -rf to force", token, aliases),
          "-r must not match as a prefix of -rf")
    check(gold.token_ok("scp -r dir host:path", token, aliases),
          "-r must match as its own token in 'scp -r dir'")


def test_no_aliases_equals_strict():
    cases = [
        ("use --context=NUM to show lines around a match", ["--context"]),
        ("scp -r dir host:path", ["--recursive"]),
        ("nothing relevant here", ["--no-clobber"]),
        ("use -n to skip existing files", ["--no-clobber", "-n"]),
    ]
    for text, tokens in cases:
        strict = all(t in text for t in tokens)
        check(gold.is_correct(text, tokens, {}) == strict,
              f"empty aliases must reproduce strict scoring for {text!r}, {tokens!r}")


def test_strict_path_reproduces():
    """The --no-aliases code path is gold.is_correct(text, tokens, {}): it
    must equal strict scoring exactly, and real aliases must never turn a
    strict-correct answer wrong (aliasing only widens what counts, never
    narrows it)."""
    qid_aliases = {
        "--no-clobber": [{"alias": "-n", "rule": "synonym",
                           "sec_id": "cp.1#OPTIONS", "line": "-n, --no-clobber"}],
    }
    cases = [
        ("use -n to avoid overwriting files", ["--no-clobber"]),
        ("use --no-clobber to avoid overwriting files", ["--no-clobber"]),
        ("no relevant flag mentioned here", ["--no-clobber"]),
    ]
    for text, tokens in cases:
        strict = all(t in text for t in tokens)
        no_aliases_path = gold.is_correct(text, tokens, {})
        check(no_aliases_path == strict,
              f"--no-aliases path must equal strict for {text!r}: "
              f"{no_aliases_path} != {strict}")
        if strict:
            check(gold.is_correct(text, tokens, qid_aliases),
                  f"aliasing must never turn a strict-correct answer wrong: {text!r}")


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
