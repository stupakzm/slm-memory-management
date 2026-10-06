"""Tests for the bracket-tolerant text rule in smm.grammar.TEMPLATE.

Hermetic (blk_test_env_constraints): stdlib only. There is no GBNF interpreter
here, so the tests pin the grammar TEXT and check a Python regex mirror of the
text rule that documents the intended language.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

if "sqlite_vec" not in sys.modules:
    stub = types.ModuleType("sqlite_vec")
    stub.load = lambda *a, **kw: None
    sys.modules["sqlite_vec"] = stub

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smm import grammar  # noqa: E402

TEXT_RULE = "text    ::= unit+"
UNIT_RULE = r'unit    ::= [^\[\]\n] | "["+ [^1-9\[\n] | "]"'
OLD_RULE = r"text    ::= [^\[\]\n]+"

# Mirror of the unit/text rules.
TEXT_RE = re.compile(r"(?:[^\[\]\n]|\[+[^1-9\[\n]|\])+")


def test_template_text_allows_brackets():
    assert TEXT_RULE in grammar.TEMPLATE
    assert UNIT_RULE in grammar.TEMPLATE
    g = grammar.cited_answer(3)
    assert TEXT_RULE in g and UNIT_RULE in g
    for s in ["[:alpha:]", "[[:alpha:]]", "[a-z]", "[0-9]", "[ ]", "]x[ ", "plain", "]"]:
        assert TEXT_RE.fullmatch(s), s


def test_template_text_forbids_open_bracket_digit():
    assert r"[^1-9\[\n]" in grammar.TEMPLATE
    assert OLD_RULE not in grammar.TEMPLATE
    for s in ["[1]", "[[1]", "[9]", "a[", "x [5", "[", "a\nb", ""]:
        assert not TEXT_RE.fullmatch(s), s


def test_template_still_ends_in_citation():
    g = grammar.cited_answer(3)
    assert 'claim   ::= text " [" ref "]"' in g
    assert 'refusal ::= "I don\'t know."' in g
    assert 'ref     ::= "1" | "2" | "3"' in g


def test_cited_answer_refs_substituted():
    for n, last in [(1, '"1"'), (4, '"4"'), (20, '"9"')]:
        g = grammar.cited_answer(n)
        assert "REFS" not in g
        assert g.endswith("\n")
        ref = [l for l in g.splitlines() if l.startswith("ref ")][0]
        assert ref.endswith(last), ref


def test_quote_template_unchanged():
    q = grammar.QUOTE_TEMPLATE
    assert OLD_RULE in q
    assert 'quote   ::= [^"\\n]+' in q
    assert "unit" not in q


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"pass {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if failed else 0)
