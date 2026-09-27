"""Tests for the keybinding alias rule (tsk_20260927_keyalias, R1 of
docs/robustness-plan.md): scripts/derive_gold_aliases.py --md-corpus.

The first Emacs eval under-credited answers that named the documented key
for a command question, or the command for a key question (`C-x k` for
"which command closes the buffer"). This rule derives that pairing
mechanically from the corpus text of a row's own gold section(s) alone -
never from an answer - via two shapes: an inline `‘KEY’ (‘COMMAND’)` pair,
and a definition-list header (or run of headers) whose description's first
`(‘COMMAND’)` names the command.

Hermetic by design (blk_test_env_constraints): this worktree has no .venv,
no models/. Fixtures here are tiny synthetic sections, not the real corpus.
sqlite_vec is stubbed first, matching the repo's test convention, even
though nothing here opens a database.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
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

_spec = importlib.util.spec_from_file_location(
    "derive_gold_aliases", ROOT / "scripts" / "derive_gold_aliases.py")
derive_gold_aliases = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(derive_gold_aliases)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _row(qid: str, doc: str, token: str, gold_sec_ids: list[str]) -> dict:
    return {"qid": qid, "kind": "answerable", "doc": doc,
            "answer_contains": [token], "gold_sec_ids": gold_sec_ids}


def test_inline_pair():
    """‘KEY’ (‘COMMAND’), close together in prose (real shape: emacs.emacs
    #appending-kills, "typing ‘C-M-w’ (‘append-next-kill’) right
    beforehand")."""
    text = (
        "If a kill command is separated from the last kill command by other\n"
        "commands, it starts a new entry.  But you can force it to combine\n"
        "with the last killed text, by typing ‘C-M-w’ "
        "(‘append-next-kill’) right beforehand.\n"
    )
    docs = {"t.emacs": {"sections": {"t.emacs#sec": text}}}
    rows = [_row("k1", "t.emacs", "append-next-kill", ["t.emacs#sec"])]

    aliases, stats = derive_gold_aliases.derive_keybinding(docs, rows)

    got = aliases["k1"]["append-next-kill"]
    check(any(e["alias"] == "C-M-w" and e["rule"] == "keybinding" for e in got),
          f"append-next-kill should get alias C-M-w via inline pair, got {got}")
    check(stats["per_rule"]["keybinding"] >= 1, f"expected >=1 match: {stats}")


def test_definition_list_multi_header():
    """One or more header lines that are exactly ‘...’ at column 0, followed
    by an indented description whose first (‘COMMAND’) names the entry's
    command - every header is a key of that command (real shape:
    emacs.emacs#marks-vs-flags: ‘% m REGEXP <RET>’ / ‘* % REGEXP <RET>’ both
    alias dired-mark-files-regexp)."""
    text = (
        "‘% m REGEXP <RET>’\n"
        "‘* % REGEXP <RET>’\n"
        "     Mark (with ‘*’) all files whose names match REGEXP\n"
        "     (‘dired-mark-files-regexp’).  This command is like\n"
        "     ‘% d’, except that it marks files with ‘*’.\n"
        "\n"
        "   Prose that mentions ‘% d’ elsewhere must not be swept in.\n"
    )
    docs = {"t.emacs": {"sections": {"t.emacs#sec": text}}}
    rows = [_row("k2", "t.emacs", "dired-mark-files-regexp", ["t.emacs#sec"])]

    aliases, _stats = derive_gold_aliases.derive_keybinding(docs, rows)

    got = {e["alias"] for e in aliases["k2"]["dired-mark-files-regexp"]}
    check(got == {"% m", "* %"},
          f"both headers should alias dired-mark-files-regexp, exactly, got {got}")


def test_placeholder_stripping():
    """Trailing ALL-CAPS placeholder words and `<RET>` are stripped; the key
    part itself is kept."""
    cases = [
        ("C-x k BUFFER <RET>", "C-x k"),
        ("% m REGEXP <RET>", "% m"),
        ("C-x r s R", "C-x r s"),
        ("C-x (", "C-x ("),  # nothing to strip
    ]
    for raw, want in cases:
        got = derive_gold_aliases.normalize_key(raw)
        check(got == want, f"normalize_key({raw!r}) == {got!r}, want {want!r}")


def test_bare_key_rejection():
    """A key is accepted only with a modifier, as a function key, or with a
    prefix character - never a bare single character or a plain word (would
    over-credit, per blk_emacs_eval_caveats-style reasoning)."""
    for bad in ("k", "g", "q", "kill", "m"):
        check(not derive_gold_aliases.is_key_shaped(bad),
              f"{bad!r} must not be key-shaped")
    for good in ("C-x (", "M-w", "s-a", "H-x", "A-f", "<F3>", "% m", "* %"):
        check(derive_gold_aliases.is_key_shaped(good),
              f"{good!r} must be key-shaped")
    check(not derive_gold_aliases.is_key_shaped("M-x some-command"),
          "'M-x <command>' is a run-by-name prefix, not a modifier keystroke")

    # End to end: a bare single-char header next to a real one (the actual
    # marks-vs-flags shape, "‘m’" / "‘* m’" both -> dired-mark) must only
    # ever license the real key.
    text = (
        "‘m’\n"
        "‘* m’\n"
        "     Mark the current file with ‘*’ (‘dired-mark’).\n"
    )
    docs = {"t.emacs": {"sections": {"t.emacs#sec": text}}}
    rows = [_row("k3", "t.emacs", "dired-mark", ["t.emacs#sec"])]
    aliases, _stats = derive_gold_aliases.derive_keybinding(docs, rows)
    got = {e["alias"] for e in aliases["k3"]["dired-mark"]}
    check(got == {"* m"}, f"the bare 'm' header must be rejected, got {got}")


def test_key_to_command_direction():
    """A key token's alias is the command (the reverse direction from a
    command token's alias being its key) - real shape: emacs.emacs
    #basic-keyboard-macro, "‘C-x (’ ... (‘kmacro-start-macro’)"."""
    text = (
        "‘C-x (’\n"
        "     Start defining a keyboard macro (old style)\n"
        "     (‘kmacro-start-macro’); with a prefix argument, append\n"
        "     keys to the last macro.\n"
    )
    docs = {"t.emacs": {"sections": {"t.emacs#sec": text}}}
    rows = [_row("k4", "t.emacs", "C-x (", ["t.emacs#sec"])]

    aliases, _stats = derive_gold_aliases.derive_keybinding(docs, rows)

    got = aliases["k4"]["C-x ("]
    check(any(e["alias"] == "kmacro-start-macro" for e in got),
          f"key token 'C-x (' should alias command kmacro-start-macro, got {got}")


def test_merge_leaves_other_qids_untouched():
    """--merge replaces/inserts only the qids of this --eval; every other
    qid in --out is byte-for-byte untouched."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        md_dir = tmp / "md"
        md_dir.mkdir()
        (md_dir / "fixture.md").write_text(
            "Intro.\n\n"
            "## Test Command\n\n"
            "Press ‘C-c C-t’ (‘test-command’) to run it.\n",
            encoding="utf-8")

        eval_path = tmp / "eval.jsonl"
        eval_path.write_text(json.dumps({
            "qid": "m01", "kind": "answerable", "doc": "fixture.emacs",
            "answer_contains": ["test-command"],
            "gold_sec_ids": ["fixture.emacs#test-command"],
        }) + "\n", encoding="utf-8")

        out_path = tmp / "gold_aliases.json"
        existing = {
            "other-qid": {"--no-clobber": [
                {"alias": "-n", "rule": "synonym",
                 "sec_id": "cp.1#OPTIONS", "line": "-n, --no-clobber"}]},
            "m01": {"stale": "must be replaced, not merged into"},
        }
        out_path.write_text(json.dumps(existing, indent=2, sort_keys=True))

        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "derive_gold_aliases.py"),
             "--md-corpus", str(md_dir), "--eval", str(eval_path),
             "--merge", "--out", str(out_path)],
            capture_output=True, text=True)
        check(result.returncode == 0,
              f"--merge should exit 0: {result.stdout}\n{result.stderr}")

        merged = json.loads(out_path.read_text())
        check(merged["other-qid"] == existing["other-qid"],
              f"a qid outside this --eval must stay byte-for-byte untouched, "
              f"got {merged.get('other-qid')}")
        check("stale" not in merged.get("m01", {}),
              f"this --eval's own qid must be replaced, not merged into, got {merged.get('m01')}")
        got = merged["m01"]["test-command"]
        check(any(e["alias"] == "C-c C-t" for e in got),
              f"m01 should be freshly derived with alias C-c C-t, got {got}")


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
