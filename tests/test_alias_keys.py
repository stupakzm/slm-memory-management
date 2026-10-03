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


_ALT_TEXT = (
    "To kill a buffer use ‘C-x k’ (‘kill-buffer’).\n"
    "To close it and its window together use ‘C-x 4 0’\n"
    "(‘kill-buffer-and-window’) instead.\n"
)
_ALT_SEC = "t.emacs#sec"


def _alt_fixture(question="how do I close a buffer and its window"):
    docs = {"t.emacs": {"sections": {_ALT_SEC: _ALT_TEXT}}}
    row = _row("a1", "t.emacs", "kill-buffer", [_ALT_SEC])
    row["question"] = question
    return docs, [row]


def _alt(**kw):
    e = {"token": "kill-buffer", "alias": "kill-buffer-and-window",
         "sec_id": _ALT_SEC,
         "line": "(‘kill-buffer-and-window’) instead.",
         "why": "closes the buffer together with its window"}
    e.update(kw)
    return e


def _check_alt(docs, rows, entries):
    alts = {"a1": entries}
    aliases, _ = derive_gold_aliases.derive_keybinding(docs, rows, alts)
    return aliases, derive_gold_aliases.check_alternatives(alts, aliases, docs, rows)


def test_alternative_command_licensed():
    docs, rows = _alt_fixture()
    aliases, viol = _check_alt(docs, rows, [_alt()])
    check(viol == [], f"a licensed alternative should pass, got {viol}")
    got = aliases["a1"]["kill-buffer"]
    rules = [(e["alias"], e["rule"]) for e in got]
    check(rules[0] == ("C-x k", "keybinding"),
          f"keybinding entries must come first, got {rules}")
    check(("kill-buffer-and-window", "alternative") in rules,
          f"alternative missing: {rules}")
    # alias equal to the token is dropped; duplicates deduped
    aliases2, _ = derive_gold_aliases.derive_keybinding(
        docs, rows, {"a1": [_alt(alias="kill-buffer"), _alt(), _alt()]})
    n = [e for e in aliases2["a1"]["kill-buffer"] if e["rule"] == "alternative"]
    check(len(n) == 1, f"expected one deduped alternative, got {n}")


def test_alternative_line_not_verbatim_rejected():
    docs, rows = _alt_fixture()
    _, viol = _check_alt(docs, rows, [_alt(line="made up (‘kill-buffer-and-window’)")])
    check(any("not verbatim" in v for v in viol), f"expected not-verbatim, got {viol}")
    _, viol = _check_alt(docs, rows, [_alt(alias="other-command")])
    check(any("not a substring" in v for v in viol), f"expected substring, got {viol}")
    _, viol = _check_alt(docs, rows, [_alt(why="")])
    check(any("no why" in v for v in viol), f"expected no-why, got {viol}")
    _, viol = _check_alt(docs, rows, [_alt(token="nope")])
    check(any("answer_contains" in v for v in viol), f"expected token, got {viol}")


def test_alternative_outside_gold_section_rejected():
    docs, rows = _alt_fixture()
    docs["t.emacs"]["sections"]["t.emacs#other"] = _ALT_TEXT
    _, viol = _check_alt(docs, rows, [_alt(sec_id="t.emacs#other")])
    check(any("not in gold_sec_ids" in v for v in viol), f"got {viol}")
    # a qid not in --eval, and a non-answerable row
    alts = {"zz": [_alt()]}
    check(any("not in this --eval" in v for v in
              derive_gold_aliases.check_alternatives(alts, {}, docs, rows)), "qid")
    rows2 = [dict(rows[0], kind="unanswerable")]
    check(any("non-answerable" in v for v in derive_gold_aliases.check_alternatives(
        {"a1": [_alt()]}, {}, docs, rows2)), "non-answerable")


def test_alternative_in_question_rejected():
    docs, rows = _alt_fixture(question="what does Kill-Buffer-And-Window do")
    _, viol = _check_alt(docs, rows, [_alt()])
    check(any("occurs in the question" in v for v in viol),
          f"an alias echoing the question must be rejected, got {viol}")
    # also for a derived alternative-key
    docs, rows = _alt_fixture(question="what is C-x 4 0 for")
    _, viol = _check_alt(docs, rows, [_alt()])
    check(any("alternative-key" in v and "C-x 4 0" in v for v in viol),
          f"alternative-key echo must be rejected, got {viol}")
    # an alias that is not command- or key-shaped
    _, viol = _check_alt(*_alt_fixture(), [_alt(alias="instead", line="instead.")])
    check(any("neither command- nor key-shaped" in v for v in viol), f"got {viol}")


def test_alternative_key_expansion():
    docs, rows = _alt_fixture()
    aliases, viol = _check_alt(docs, rows, [_alt()])
    keys = [e for e in aliases["a1"]["kill-buffer"] if e["rule"] == "alternative-key"]
    check(len(keys) == 1 and keys[0]["alias"] == "C-x 4 0"
          and keys[0]["via"] == "kill-buffer-and-window"
          and keys[0]["sec_id"] == _ALT_SEC,
          f"expected the alternative's documented key, got {keys}")
    check(keys[0]["line"] in _ALT_TEXT, "key line must be verbatim")
    _, stats = derive_gold_aliases.derive_keybinding(docs, rows, {"a1": [_alt()]})
    check(stats["per_rule"] == {"keybinding": 1, "alternative": 1,
                                "alternative-key": 1}, f"{stats}")
    # a key alternative expands nothing
    a2, _ = _check_alt(docs, rows, [_alt(alias="C-x 4 0", line="‘C-x 4 0’")])
    check(all(e["rule"] != "alternative-key" for e in a2["a1"]["kill-buffer"]),
          "a key-shaped alternative must not expand")


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "derive_gold_aliases.py"), *args],
        capture_output=True, text=True)


def _cli_fixture(td, question="how do I close a buffer and its window"):
    tmp = Path(td)
    md = tmp / "md"
    md.mkdir()
    (md / "fixture.md").write_text(
        "Intro.\n\n## Sec\n\n" + _ALT_TEXT, encoding="utf-8")
    ev = tmp / "eval.jsonl"
    ev.write_text(json.dumps({
        "qid": "a1", "kind": "answerable", "doc": "fixture.emacs",
        "question": question, "answer_contains": ["kill-buffer"],
        "gold_sec_ids": ["fixture.emacs#sec"]}) + "\n", encoding="utf-8")
    alt = tmp / "alt.json"
    alt.write_text(json.dumps({"a1": [_alt(sec_id="fixture.emacs#sec")]}))
    return md, ev, alt, tmp / "out.json"


def test_check_flags_stale_merge():
    with tempfile.TemporaryDirectory() as td:
        md, ev, alt, out = _cli_fixture(td)
        base = ["--md-corpus", str(md), "--eval", str(ev), "--out", str(out)]
        r = _run_cli(*base, "--merge")
        check(r.returncode == 0, f"{r.stdout}{r.stderr}")
        r = _run_cli(*base, "--check")
        check(r.returncode == 0, f"fresh no-flag merge should check clean: {r.stdout}")
        r = _run_cli(*base, "--alternatives", str(alt), "--check")
        check(r.returncode == 1 and "stale" in r.stdout,
              f"merge without alternatives is stale against them: {r.stdout}")
        r = _run_cli(*base, "--alternatives", str(alt), "--merge")
        check(r.returncode == 0 and "alternative-key aliases derived: 1" in r.stdout,
              f"{r.stdout}{r.stderr}")
        r = _run_cli(*base, "--alternatives", str(alt), "--check")
        check(r.returncode == 0, f"{r.stdout}")
        # no-flag check ignores the alternative entries
        r = _run_cli(*base, "--check")
        check(r.returncode == 0, f"no-flag check must compare keybinding only: {r.stdout}")
        # tamper with the merged file: stale
        data = json.loads(out.read_text())
        data["a1"]["kill-buffer"][0]["alias"] = "C-x z"
        out.write_text(json.dumps(data))
        r = _run_cli(*base, "--check")
        check(r.returncode == 1 and "stale" in r.stdout, f"{r.stdout}")


def test_no_alternatives_flag_is_unchanged():
    with tempfile.TemporaryDirectory() as td:
        md, ev, alt, out = _cli_fixture(td)
        base = ["--md-corpus", str(md), "--eval", str(ev), "--out", str(out)]
        r = _run_cli(*base)
        check(r.returncode == 0, f"{r.stdout}{r.stderr}")
        plain = out.read_bytes()
        check("alternative" not in r.stdout, f"no alternative output expected: {r.stdout}")
        r = _run_cli(*base, "--alternatives", str(alt))
        check(r.returncode == 0, f"{r.stdout}{r.stderr}")
        check(out.read_bytes() != plain, "alternatives should change the output")
        # an empty FILE changes nothing
        empty = Path(td) / "empty.json"
        empty.write_text(json.dumps({"a1": []}))
        _run_cli(*base, "--alternatives", str(empty))
        check(out.read_bytes() == plain, "empty alternatives must be byte-identical")
        aliases, stats = derive_gold_aliases.derive_keybinding(*_alt_fixture())
        check(stats["per_rule"] == {"keybinding": 1}, f"{stats}")


# --- tsk_20261003_altv2: man-mode --alternatives, --base, --variants ---

_MAN_SEC = "ss.8#OPTIONS"
_MAN_TEXT = (
    "       -l, --listening\n"
    "              Display only listening sockets.\n"
    "       -a, --all\n"
    "              Display both listening and non-listening sockets.\n"
)


def _man_fixture(td, extra_rows=()):
    """man.jsonl with one doc/section, an eval with q1 (+ extra rows), and the
    v1-style base derived from it (q1 gets `-l`; q2 is an unrelated row)."""
    tmp = Path(td)
    corpus = tmp / "man.jsonl"
    corpus.write_text(json.dumps({
        "doc_id": "ss.8", "sections": [{
            "sec_id": _MAN_SEC, "heading": "OPTIONS", "level": 1,
            "parent": None, "text": _MAN_TEXT}]}) + "\n", encoding="utf-8")
    rows = [
        {"qid": "q1", "kind": "answerable", "doc": "ss.8",
         "question": "show only the sockets waiting for connections",
         "answer_contains": ["--listening"], "gold_sec_ids": [_MAN_SEC]},
        {"qid": "q2", "kind": "answerable", "doc": "ss.8",
         "question": "show only the sockets waiting for connections please",
         "answer_contains": ["--listening"], "gold_sec_ids": [_MAN_SEC]},
        *extra_rows,
    ]
    ev = tmp / "eval.jsonl"
    ev.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return corpus, ev, rows


def _man_alt(**kw):
    e = {"token": "--listening", "alias": "--all", "sec_id": _MAN_SEC,
         "line": "-a, --all", "why": "lists listening sockets too"}
    e.update(kw)
    return e


def _man_check(rows, entries, qid="q1"):
    docs = {"ss.8": {"sections": {_MAN_SEC: "OPTIONS\n" + _MAN_TEXT}}}
    return derive_gold_aliases.check_alternatives(
        {qid: entries}, {}, docs, rows, man=True)


def test_man_alternative_licensed_by_option_line():
    with tempfile.TemporaryDirectory() as td:
        corpus, ev, rows = _man_fixture(td)
        docs = derive_gold_aliases.load_man_docs(corpus)
        text = docs["ss.8"]["sections"][_MAN_SEC]
        check(text == "OPTIONS\n" + _MAN_TEXT,
              f"section text is heading, newline, text: {text!r}")
        alts = {"q1": [_man_alt()]}
        check(derive_gold_aliases.check_alternatives(
            alts, {}, docs, rows, man=True) == [], "licensed alternative must pass")
        base, _ = derive_gold_aliases.derive(corpus, ev)
        check(base["q1"]["--listening"][0]["alias"] == "-l", f"{base}")
        merged, added = derive_gold_aliases.merge_man_alternatives(base, alts, rows)
        got = merged["q1"]["--listening"]
        check(got[:1] == base["q1"]["--listening"],
              f"the base entry must come first, unchanged: {got}")
        check(got[1:] == [{"alias": "--all", "rule": "alternative",
                           "sec_id": _MAN_SEC, "line": "-a, --all"}] and added == 1,
              f"one appended alternative expected: {got}")
        check(merged["q2"] == base["q2"], "a qid not in FILE must be unchanged")
        merged2, added2 = derive_gold_aliases.merge_man_alternatives(
            merged, {"q1": [_man_alt(), _man_alt(alias="--listening")]}, rows)
        check(merged2 == merged and added2 == 0,
              "re-merging dedupes on (alias, sec_id); alias == token is dropped")
    # rejections: each licensed field is policed
    cases = [
        (dict(line="-a, --everything"), "not verbatim"),
        (dict(alias="--other", line="-a, --all"), "not a substring"),
        (dict(sec_id="ss.8#OTHER"), "not in gold_sec_ids"),
        (dict(token="--tcp"), "answer_contains"),
        (dict(why=" "), "no why"),
    ]
    for kw, want in cases:
        viol = _man_check(rows, [_man_alt(**kw)])
        check(any(want in v for v in viol), f"{kw}: expected {want!r}, got {viol}")
    rows_q = [dict(rows[0], question="what does --ALL do")]
    check(any("occurs in the question" in v for v in _man_check(rows_q, [_man_alt()])),
          "an alias echoing the question must be rejected (case-insensitive)")
    check(any("not in this --eval" in v for v in _man_check(rows, [_man_alt()], "zz")),
          "a qid not in --eval must be rejected")


def test_man_alternative_shape_rules():
    shaped = derive_gold_aliases.is_man_alias_shaped
    for good in ("-l", "--listening", "--color=WHEN", "DenyUsers", "getfacl",
                 "/etc/shadow"):
        check(shaped(good), f"{good!r} must be accepted")
    for bad in ("ss -l", "x", "!!", "", "-", "--", "a b", "-l\n", "--x y"):
        check(not shaped(bad), f"{bad!r} must be rejected")
    # and through the checker: the shape rule is reported, whatever the line
    rows = [{"qid": "q1", "kind": "answerable", "doc": "ss.8", "question": "q",
             "answer_contains": ["--listening"], "gold_sec_ids": [_MAN_SEC]}]
    for alias, line in (("ss -l", "-l, --listening"), ("x", "Display"),
                        ("!!", "Display")):
        viol = _man_check(rows, [_man_alt(alias=alias, line=line)])
        check(any("neither flag- nor identifier-shaped" in v for v in viol),
              f"{alias!r} should violate the shape rule, got {viol}")
    check(_man_check(rows, [_man_alt(alias="-l", line="-l, --listening")]) == [],
          "a flag alias is accepted")


def _digest(entry):
    return json.dumps(entry, indent=2, sort_keys=True)


def test_base_flag_leaves_base_file_untouched():
    with tempfile.TemporaryDirectory() as td:
        corpus, ev, rows = _man_fixture(td)
        tmp = Path(td)
        base_obj, _ = derive_gold_aliases.derive(corpus, ev)
        base_obj["other"] = {"--foo": [{"alias": "-f", "rule": "synonym",
                                        "sec_id": "x#y", "line": "-f, --foo"}]}
        base = tmp / "base.json"
        base.write_text(json.dumps(base_obj, indent=2, sort_keys=True))
        before = base.read_bytes()
        alt = tmp / "alt.json"
        alt.write_text(json.dumps({"q1": [_man_alt()]}))
        out = tmp / "v2.json"
        args = ["--corpus", str(corpus), "--eval", str(ev), "--alternatives",
                str(alt), "--base", str(base), "--out", str(out)]
        r = _run_cli(*args, "--merge")
        check(r.returncode == 0, f"{r.stdout}{r.stderr}")
        check("1 alternatives merged" in r.stdout and "0 propagated" in r.stdout
              and "0 skipped as in question" in r.stdout, r.stdout)
        check(base.read_bytes() == before, "--base must never be written")
        got = json.loads(out.read_text())
        for qid in base_obj:
            if qid != "q1":
                check(_digest(got[qid]) == _digest(base_obj[qid]),
                      f"{qid} not in FILE must be byte-identical to --base")
        check(got["q1"]["--listening"][:1] == base_obj["q1"]["--listening"]
              and got["q1"]["--listening"][1]["alias"] == "--all",
              f"q1 is the base entry plus the alternative: {got['q1']}")
        r = _run_cli(*args, "--check")
        check(r.returncode == 0, f"fresh merge checks clean: {r.stdout}")
        got["q1"]["--listening"][1]["alias"] = "--tampered"
        out.write_text(json.dumps(got))
        r = _run_cli(*args, "--check")
        check(r.returncode == 1 and "stale" in r.stdout, f"{r.stdout}")
        # --check needs --base; --base must not be --out; no flags, no write
        r = _run_cli("--corpus", str(corpus), "--eval", str(ev), "--alternatives",
                     str(alt), "--out", str(out), "--check")
        check(r.returncode == 2, f"man --check without --base: {r.returncode}")
        r = _run_cli(*args[:-2], "--out", str(base), "--merge")
        check(r.returncode == 2 and base.read_bytes() == before,
              f"--base == --out must be refused: {r.returncode}")
        # a merge with violations writes nothing
        bad = tmp / "bad.json"
        bad.write_text(json.dumps({"q1": [_man_alt(alias="ss -l", line="x")]}))
        out2 = tmp / "never.json"
        r = _run_cli("--corpus", str(corpus), "--eval", str(ev), "--alternatives",
                     str(bad), "--base", str(base), "--out", str(out2), "--merge")
        check(r.returncode == 1 and not out2.exists()
              and base.read_bytes() == before, f"{r.stdout}{r.stderr}")


def _variant_rows():
    return [
        {"qid": "q1.t1", "kind": "answerable", "doc": "ss.8",
         "question": "show only the sockets waiting for connectoins",
         "answer_contains": ["--listening"], "gold_sec_ids": [_MAN_SEC],
         "variant_of": "q1"},
        {"qid": "q1.t1.p", "kind": "answerable", "doc": "ss.8",
         "question": "which sockets are in the listen state",
         "answer_contains": ["--listening", "--tcp"], "gold_sec_ids": [_MAN_SEC],
         "paraphrase_of": "q1.t1"},
        {"qid": "q1.t2", "kind": "answerable", "doc": "ss.8",
         "question": "sockets waiting for connections", "variant_of": "q1",
         "answer_contains": ["--listening"], "gold_sec_ids": [_MAN_SEC]},
    ]


def test_propagates_to_variants_with_own_entry():
    with tempfile.TemporaryDirectory() as td:
        corpus, ev, rows = _man_fixture(td)
        tmp = Path(td)
        var = tmp / "variants.jsonl"
        var.write_text("".join(json.dumps(r) + "\n" for r in _variant_rows()))
        base_obj, _ = derive_gold_aliases.derive(corpus, ev)
        own = [{"alias": "-l", "rule": "synonym", "sec_id": _MAN_SEC,
                "line": "-l, --listening"}]
        base_obj["q1.t1"] = {"--listening": list(own)}
        base_obj["q1.t1.p"] = {}  # explicit empty entry is still "its own"
        # q1.t2 has no entry: scoring falls back to q1, nothing is written for it
        base = tmp / "base.json"
        base.write_text(json.dumps(base_obj, indent=2, sort_keys=True))
        alt = tmp / "alt.json"
        alt.write_text(json.dumps({"q1": [_man_alt()]}))
        out = tmp / "v2.json"
        args = ["--corpus", str(corpus), "--eval", str(ev), "--variants", str(var),
                "--alternatives", str(alt), "--base", str(base), "--out", str(out)]
        r = _run_cli(*args, "--merge")
        check(r.returncode == 0, f"{r.stdout}{r.stderr}")
        check("1 alternatives merged, 2 propagated to variants, 0 skipped as in "
              "question" in r.stdout, r.stdout)
        got = json.loads(out.read_text())
        want = {"alias": "--all", "rule": "alternative", "sec_id": _MAN_SEC,
                "line": "-a, --all", "via_base": "q1"}
        check(got["q1.t1"]["--listening"] == own + [want],
              f"variant keeps its entry and gains the root's: {got['q1.t1']}")
        check(got["q1.t1.p"] == {"--listening": [want]},
              f"a grandchild's root is followed to the end, and only tokens in "
              f"its answer_contains: {got['q1.t1.p']}")
        check("q1.t2" not in got, "a row without its own entry gets none")
        check("via_base" not in json.dumps(got["q1"]), "the root itself is not a variant")
        r = _run_cli(*args, "--check")
        check(r.returncode == 0, f"{r.stdout}")
        got["q1.t1"]["--listening"].pop()
        out.write_text(json.dumps(got))
        r = _run_cli(*args, "--check")
        check(r.returncode == 1 and "q1.t1: --out entry is stale" in r.stdout,
              f"a missing propagated entry is stale: {r.stdout}")
    # a root that is not a loaded row is still the root
    target = {"v": {}}
    stats, touched = derive_gold_aliases.propagate_to_variants(
        target, [{"qid": "v", "variant_of": "ghost", "question": "q",
                  "answer_contains": ["--listening"]}],
        {"ghost": [_man_alt()]}, {"v"})
    check(stats["propagated"] == 1 and touched == {"v"}
          and target["v"]["--listening"][0]["via_base"] == "ghost", f"{target}")


def test_propagation_revalidates_variant_question():
    rows = [
        {"qid": "q1.t1", "question": "does --ALL show listening sockets",
         "answer_contains": ["--listening"], "variant_of": "q1"},
        {"qid": "q1.t2", "question": "which sockets are listening",
         "answer_contains": ["--listening"], "variant_of": "q1"},
    ]
    target = {"q1.t1": {}, "q1.t2": {}}
    stats, _ = derive_gold_aliases.propagate_to_variants(
        target, rows, {"q1": [_man_alt()]}, set(target))
    check(target["q1.t1"] == {}, f"alias in the variant's question: {target['q1.t1']}")
    check(target["q1.t2"]["--listening"][0]["alias"] == "--all", f"{target}")
    check(stats == {"propagated": 1, "skipped_in_question": 1}, f"{stats}")
    # md mode: same rule, end to end through the CLI
    with tempfile.TemporaryDirectory() as td:
        md, ev, alt, out = _cli_fixture(td)
        tmp = Path(td)
        var = tmp / "variants.jsonl"
        var.write_text(json.dumps({
            "qid": "a1.t", "kind": "answerable", "doc": "fixture.emacs",
            "question": "what does Kill-Buffer-And-Window do?",
            "answer_contains": ["kill-buffer"], "gold_sec_ids": ["fixture.emacs#sec"],
            "variant_of": "a1"}) + "\n")
        base = tmp / "base.json"
        base.write_text(json.dumps({"a1.t": {}}))
        r = _run_cli("--md-corpus", str(md), "--eval", str(ev), "--variants",
                     str(var), "--alternatives", str(alt), "--base", str(base),
                     "--out", str(out), "--merge")
        check(r.returncode == 0 and "0 propagated to variants, 1 skipped as in "
              "question" in r.stdout, f"{r.stdout}{r.stderr}")
        check(json.loads(out.read_text())["a1.t"] == {}, "nothing propagated")


def test_no_new_flags_output_unchanged():
    with tempfile.TemporaryDirectory() as td:
        corpus, ev, rows = _man_fixture(td)
        out = Path(td) / "out.json"
        r = _run_cli("--corpus", str(corpus), "--eval", str(ev), "--out", str(out))
        check(r.returncode == 0, f"{r.stdout}{r.stderr}")
        want = {qid: {"--listening": [{
            "alias": "-l", "rule": "synonym", "sec_id": _MAN_SEC,
            "line": "-l, --listening"}]} for qid in ("q1", "q2")}
        check(out.read_text() == json.dumps(want, indent=2, sort_keys=True),
              f"man mode without the new flags must write the derive() output: "
              f"{out.read_text()}")
        check(r.stdout.startswith(f"wrote {out}\nquestions touched: 2\n"
                                  "aliases per rule:\n")
              and "alternative" not in r.stdout and "propagated" not in r.stdout,
              f"stdout unchanged: {r.stdout}")
        # an md-mode --merge without the new flags is still the old merge
        md, ev2, alt, out2 = _cli_fixture(td)
        r = _run_cli("--md-corpus", str(md), "--eval", str(ev2), "--out", str(out2),
                     "--merge")
        check(r.returncode == 0 and "propagated" not in r.stdout, f"{r.stdout}")


def test_man_check_without_alternatives_never_writes():
    with tempfile.TemporaryDirectory() as td:
        corpus, ev, rows = _man_fixture(td)
        out = Path(td) / "curated.json"
        out.write_text('{"hand": "curated"}')
        before = out.read_bytes()
        r = _run_cli("--corpus", str(corpus), "--eval", str(ev), "--out", str(out),
                     "--check")
        check(r.returncode == 2, f"{r.returncode} {r.stdout}{r.stderr}")
        check("needs --alternatives" in r.stderr, r.stderr)
        check(out.read_bytes() == before, "--out must be byte-identical")


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
