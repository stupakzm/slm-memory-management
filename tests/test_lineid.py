"""Tests for --answer-mode line-id (tsk_20261006_lineid): like line mode, but only
lines naming an option or key (flag, Emacs chord, M-x command) are offered.
Hermetic (blk_test_env_constraints): stdlib only, sqlite_vec stubbed through
tests/test_instructions.py. The grammar is checked as text.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_instructions as ti  # noqa: E402  (stubs sqlite_vec, loads eval_answers)

generate = ti.generate
check = ti.check
grammar = sys.modules["smm.grammar"]


def _rules(g: str) -> dict:
    out = {}
    for ln in g.strip().splitlines():
        name, _, body = ln.partition("::=")
        out[name.strip()] = body.strip()
    return out


def _alts(body: str) -> list:
    return re.findall(r'"((?:[^"\\]|\\.)*)"', body)


class _Chat(ti._Chat):
    pass


EXTRACTS = [
    {"doc_id": "a.1", "prefix": "A\n",
     "text": "Display only lines which do NOT match.\n  -v, --invert-match  select non-matching lines\nplain prose here"},
    {"doc_id": "b.1", "prefix": "B\n", "text": "only plain prose, well-known and read-only"},
    {"doc_id": "c.1", "prefix": "C\n", "text": "Type C-x C-f to visit a file\nM-x find-file does the same"},
]


def test_identifier_line_matches_flags_chords_and_mx():
    f = grammar.is_identifier_line
    check(f("  -v, --invert-match  select"), "long and short flag")
    check(f("use --null to separate"), "long flag mid-line")
    check(f("print with -n only"), "short flag standing alone")
    check(f("Type C-x C-f to visit"), "chord")
    check(f("press M-f to move"), "meta chord")
    check(f("run M-x find-file now"), "M-x command")
    check(f("(s-a) super"), "chord after paren")


def test_identifier_line_rejects_plain_prose_and_hyphenated_words():
    f = grammar.is_identifier_line
    check(not f("null-separated, read-only, well-known"), "hyphenated English")
    check(not f("Display only lines which do NOT match."), "plain prose")
    check(not f("a pre-C-x thing"), "glued to a hyphen")
    check(not f("the -foo option"), "short flag followed by word characters")
    check(not f("rock-and-roll"), "dash inside words")


def test_extract_lines_identifier_only_keeps_only_those():
    got = grammar.extract_lines(EXTRACTS[0], identifier_only=True)
    check(got == ["-v, --invert-match  select non-matching lines"], got)
    check(len(grammar.extract_lines(EXTRACTS[0])) == 3, "default is unchanged")
    check(grammar.extract_lines(EXTRACTS[0], identifier_only=False) == grammar.extract_lines(EXTRACTS[0]),
          "False is the default")
    chunk = {"prefix": "", "text": "\n".join(f"opt --flag{i % 2} line" for i in range(10))}
    check(grammar.extract_lines(chunk, identifier_only=True, max_lines=1) == ["opt --flag0 line"],
          "dedupe and max_lines apply after the filter")
    check(grammar.extract_lines(chunk, identifier_only=True) == ["opt --flag0 line", "opt --flag1 line"],
          "dedupe after filter")


def test_line_answer_identifier_only_omits_extracts_without_such_lines():
    r = _rules(grammar.line_answer(EXTRACTS, identifier_only=True))
    check(r["claim"] == "claim1 | claim3", r["claim"])
    check("line2" not in r and "claim2" not in r, "no rule for the prose-only extract")
    check(_alts(r["line1"]) == ["-v, --invert-match  select non-matching lines"], r["line1"])
    check(_alts(r["line3"]) == ["Type C-x C-f to visit a file", "M-x find-file does the same"], r["line3"])
    check(r["claim3"].endswith('" [3]"'), r["claim3"])
    prose = [EXTRACTS[1]]
    g = grammar.line_answer(prose, identifier_only=True)
    check(g == 'root    ::= refusal\nrefusal ::= "I don\'t know."\n', g)


def test_generator_line_id_mode_uses_identifier_grammar():
    g = _Chat()
    g.answer("q?", EXTRACTS, mode="line-id")
    msgs, gram = g.calls[0]
    check(msgs == generate.build_prompt("q?", EXTRACTS, system=generate.QUOTE_SYSTEM), "quote prompt")
    check(gram == grammar.line_answer(EXTRACTS, identifier_only=True), "identifier grammar")
    check(gram != grammar.line_answer(EXTRACTS), "differs from line grammar")
    g.answer("q?", EXTRACTS, mode="line-id", reader_prompt="v1", cite_grammar=True)
    check(g.calls[1] == g.calls[0], "cite_grammar is irrelevant")


def test_eval_answers_accepts_line_id_and_records_it():
    code, out, _, _ = ti._eval_run(["--answer-mode", "line-id"])
    check(code == 0, f"exit {code}")
    check(out["config"].get("answer_mode") == "line-id", out["config"])
    rows = out["results"] if "results" in out else out["records"]
    rec = rows[0]
    check(not [k for k in rec if k.startswith("quote_")] and "answer_raw" not in rec,
          f"line-id runs no verify_quotes: {sorted(rec)}")


def test_line_id_rejects_reader_prompt_v2_v3():
    for v in ("v2", "v3"):
        code, out, _, ans_kw = ti._eval_run(["--reader-prompt", v, "--answer-mode", "line-id"])
        check(code == 2 and out is None and not ans_kw, f"{v} with line-id must exit 2: {code}")
        try:
            _Chat().answer("q?", ti.CHUNKS, mode="line-id", reader_prompt=v)
        except ValueError:
            continue
        raise AssertionError(f"line-id mode with {v} must raise ValueError")


def test_line_mode_unchanged_by_the_new_parameter():
    g = _Chat()
    g.answer("q?", EXTRACTS, mode="line")
    check(g.calls[0] == (generate.build_prompt("q?", EXTRACTS, system=generate.QUOTE_SYSTEM),
                         grammar.line_answer(EXTRACTS)), "line mode")
    check(len(_alts(_rules(grammar.line_answer(EXTRACTS))["line2"])) == 1, "prose extract still offered in line mode")
    check(grammar.line_answer(EXTRACTS) == grammar.line_answer(EXTRACTS, identifier_only=False), "default")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  pass  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
