"""Tests for --answer-mode line (tsk_20261006_linemode): the answer's opening
quotation must be a real line of the extract it cites, enforced by a per-question
grammar listing each extract's lines as literal alternatives. cite and quote modes
must stay byte-identical.

Hermetic (blk_test_env_constraints): stdlib only; sqlite_vec is stubbed and the
eval_answers harness comes from tests/test_instructions.py. No GBNF engine is
available here, so the grammar is checked as text: rule names, literals, numbers.
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
    """The literal alternatives of a rule body made only of GBNF string literals."""
    return re.findall(r'"((?:[^"\\]|\\.)*)"', body)


def test_extract_lines_strip_dedupe_and_cap():
    chunk = {"prefix": "PFX", "text": "  alpha beta  \nabc\n\n\tgamma delta\nalpha beta\n" + "x" * 200
             + "\n" + "\n".join(f"line number {i}" for i in range(60))}
    got = grammar.extract_lines(chunk)
    check(got[:3] == ["alpha beta", "gamma delta", "x" * 160], f"strip/short/dedupe/cut: {got[:3]}")
    check("abc" not in got, "a line under 4 characters is dropped")
    check(got.count("alpha beta") == 1, "first occurrence only")
    check(len(got) == 40, f"at most 40 lines by default, got {len(got)}")
    check(got[3] == "line number 0", f"order is kept: {got[3]}")
    check(grammar.extract_lines(chunk, max_lines=2) == ["alpha beta", "gamma delta"], "max_lines")
    check(grammar.extract_lines({"prefix": "", "text": "abcdefgh"}, max_len=5) == ["abcde"], "max_len")
    check(grammar.extract_lines({"prefix": "", "text": "abcd\nabc"}) == ["abcd"], "4 characters is kept")


def test_extract_lines_ignore_prefix():
    chunk = {"prefix": "NAME malloc - allocate memory\n", "text": "the body line"}
    check(grammar.extract_lines(chunk) == ["the body line"], grammar.extract_lines(chunk))


def test_gbnf_literal_escapes_quote_and_backslash():
    check(grammar.gbnf_literal("plain") == '"plain"', grammar.gbnf_literal("plain"))
    check(grammar.gbnf_literal('say "hi"') == '"say \\"hi\\""', grammar.gbnf_literal('say "hi"'))
    check(grammar.gbnf_literal("a\\b") == '"a\\\\b"', grammar.gbnf_literal("a\\b"))
    check(grammar.gbnf_literal("a\tb\rc\x01d") == '"a b c d"', grammar.gbnf_literal("a\tb\rc\x01d"))
    check(grammar.gbnf_literal("café →") == '"café →"', "non-ASCII stays")


EXTRACTS = [
    {"doc_id": "a.1", "prefix": "A\n", "text": "first line of one\nsecond line of one"},
    {"doc_id": "b.1", "prefix": "B\n", "text": "only line of two"},
    {"doc_id": "c.1", "prefix": "C\n", "text": "x\n\n  \n"},
    {"doc_id": "d.1", "prefix": "D\n", "text": "line of four"},
]


def test_line_answer_has_one_alternative_per_nonempty_extract():
    g = grammar.line_answer(EXTRACTS)
    r = _rules(g)
    check(r["root"] == "refusal | claim", r["root"])
    check(r["claim"] == "claim1 | claim2 | claim4", f"empty extract 3 gets no alternative: {r['claim']}")
    check("line3" not in r and "claim3" not in r, "no rule for the empty extract")
    check(_alts(r["line1"]) == ["first line of one", "second line of one"], r["line1"])
    check(_alts(r["line2"]) == ["only line of two"], r["line2"])
    check(r["text"] == r"[^\[\]\n]+", r["text"])
    check("CLAIMS" not in g, "placeholder substituted")
    many = [{"prefix": "", "text": f"extract line {i}"} for i in range(12)]
    rm = _rules(grammar.line_answer(many))
    check("claim9" in rm and "claim10" not in rm, "at most 9 extracts")


def test_line_answer_citation_matches_its_extract():
    r = _rules(grammar.line_answer(EXTRACTS))
    for n in (1, 2, 4):
        body = r[f"claim{n}"]
        check(f"line{n}" in body and f'" [{n}]"' in body, f"claim{n} pairs line{n} with [{n}]: {body}")
        check(body.startswith(r'"\"" ') and r'"\" "' in body, f"quote framing: {body}")
        others = [m for m in re.findall(r"line(\d+)", body) if int(m) != n]
        check(not others, f"claim{n} mentions another extract's lines: {body}")
        check(re.findall(r'" \[(\d+)\]"', body) == [str(n)], body)
    # the claim shape matches what verify_quotes parses
    line = grammar.extract_lines(EXTRACTS[1])[0]
    answer = f'"{line}" it says so [2]'
    check(grammar.verify_quotes(answer, EXTRACTS)["quote_verified"] is True, "verify_quotes accepts it")


def test_line_answer_refusal_is_allowed():
    g = grammar.line_answer(EXTRACTS)
    r = _rules(g)
    check("refusal" in r["root"] and r["refusal"] == '"I don\'t know."', r["root"])
    check(r["refusal"] == '"' + grammar.REFUSAL + '"', r["refusal"])


def test_line_answer_without_lines_is_refusal_only():
    for ex in ([], [{"prefix": "", "text": ""}, {"prefix": "p p p p", "text": "ab\n c "}]):
        g = grammar.line_answer(ex)
        r = _rules(g)
        check(r["root"] == "refusal", f"refusal alone: {g!r}")
        check("claim" not in g and r["refusal"] == '"I don\'t know."', g)


class _LineChat(ti._Chat):
    pass


def test_generator_line_mode_uses_quote_system_and_line_grammar():
    g = _LineChat()
    g.answer("q?", EXTRACTS, mode="line")
    msgs, gram = g.calls[0]
    check(msgs == generate.build_prompt("q?", EXTRACTS, system=generate.QUOTE_SYSTEM), "quote prompt")
    check(gram == grammar.line_answer(EXTRACTS), "line grammar")
    check(gram != grammar.quoted_answer(len(EXTRACTS)), "not the free-quote grammar")
    g.answer("q?", EXTRACTS, mode="line", reader_prompt="v1", cite_grammar=True)
    check(g.calls[1] == g.calls[0], "cite_grammar is irrelevant in line mode")


def test_eval_answers_accepts_line_mode_and_records_it():
    code, out, _, ans_kw = ti._eval_run(["--answer-mode", "line"])
    check(code == 0, f"exit {code}")
    check(out["config"].get("answer_mode") == "line", out["config"])
    rows = out["results"] if isinstance(out, dict) and "results" in out else out["records"]
    rec = rows[0]
    check(rec["answer"] == "apt-get install [1]", f"line mode does not rewrite the answer: {rec}")
    check(not [k for k in rec if k.startswith("quote_")] and "answer_raw" not in rec,
          f"line mode runs no verify_quotes: {sorted(rec)}")
    code, out, _, _ = ti._eval_run(["--answer-mode", "quote"])
    check(code == 0 and out["config"].get("answer_mode") == "quote", "quote mode still verifies")
    rows = out["results"] if "results" in out else out["records"]
    check(rows[0].get("quote_failed") is True, "stub answer has no quote, so quote mode converts it")


def test_line_mode_rejects_reader_prompt_v2_v3():
    for v in ("v2", "v3"):
        code, out, _, ans_kw = ti._eval_run(["--reader-prompt", v, "--answer-mode", "line"])
        check(code == 2 and out is None and not ans_kw, f"{v} with line must exit 2: {code}")
        try:
            _LineChat().answer("q?", ti.CHUNKS, mode="line", reader_prompt=v)
        except ValueError:
            continue
        raise AssertionError(f"Generator.answer line mode with {v} must raise ValueError")


def test_cite_and_quote_modes_unchanged():
    g = _LineChat()
    g.answer("q?", ti.CHUNKS, cite_grammar=True)
    check(g.calls[0] == (generate.build_prompt("q?", ti.CHUNKS), grammar.cited_answer(len(ti.CHUNKS))),
          "cite mode")
    g.answer("q?", ti.CHUNKS)
    check(g.calls[1] == (generate.build_prompt("q?", ti.CHUNKS), None), "cite mode, no grammar")
    g.answer("q?", ti.CHUNKS, mode="quote")
    check(g.calls[2] == (generate.build_prompt("q?", ti.CHUNKS, system=generate.QUOTE_SYSTEM),
                         grammar.quoted_answer(len(ti.CHUNKS))), "quote mode")
    for kw in (dict(mode="quote", reader_prompt="v2"), dict(mode="quote", reader_prompt="v3")):
        try:
            g.answer("q?", ti.CHUNKS, **kw)
        except ValueError:
            continue
        raise AssertionError(f"{kw} must still raise ValueError")


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
