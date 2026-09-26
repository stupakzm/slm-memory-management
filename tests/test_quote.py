"""Tests for the phase 10 opt-in quote-grounded answer mode (smm.grammar's
QUOTE_TEMPLATE/quoted_answer/verify_quotes, smm.generate's QUOTE_SYSTEM and
build_prompt's system= parameter).

Motivation (blk_phase9_parametric_knowledge_failure): the 30B reader gains
correctness by answering from parametric knowledge, not by reading, and the
reranker gate cannot catch this because a parametric answer still scores
0.94-0.999 (blk_phase9_30b_gate_threshold_failure) - so the fix has to be in
the reader/claim itself. Requiring each claim to open with an exact quotation
from the extract it cites makes that mechanically checkable
(blk_phase6_verifier_failure: this is a substring check, not a judged one).

Hermetic by design (blk_test_env_constraints): this worktree has no .venv, no
data/, no models/. `smm.grammar` and `smm.generate` are stdlib-only, but
importing `scripts.eval_answers` (not exercised here, but kept as the house
convention in case future tests in this file grow to need it) pulls in
`smm.store`, which imports the third-party `sqlite_vec`. Stub it in
sys.modules first anyway, exactly as tests/test_eval_answers.py does, so this
file stays safe to extend without re-deriving the convention.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

if "sqlite_vec" not in sys.modules:
    stub = types.ModuleType("sqlite_vec")
    stub.load = lambda *a, **kw: None  # never called: no real db is opened here
    sys.modules["sqlite_vec"] = stub

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import grammar  # noqa: E402
from smm.generate import SYSTEM, build_prompt  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_quote_verbatim_in_cited_extract_verifies():
    chunks = [{"doc_id": "a.1", "prefix": "", "text": "The malloc function returns NULL on failure."}]
    g = grammar.quoted_answer(len(chunks))
    check("REFS" not in g, f"quoted_answer must substitute REFS, got {g!r}")
    answer = '"returns NULL on failure" NULL is returned. [1]'
    got = grammar.verify_quotes(answer, chunks)
    check(got["quote_verified"] is True, f"exact-substring quote must verify, got {got}")
    check(got["quote_failed"] is False, f"a verified quote must not be failed, got {got}")
    check(got["n_quotes"] == 1, f"expected 1 quote, got {got}")


def test_quote_present_only_in_a_different_extract_fails():
    chunks = [
        {"doc_id": "a.1", "prefix": "", "text": "malloc returns NULL on failure."},
        {"doc_id": "b.1", "prefix": "", "text": "free() takes a pointer previously returned by malloc."},
    ]
    # quote is real text, but lifted from extract 1 while the claim cites [2]
    answer = '"returns NULL on failure" it fails. [2]'
    got = grammar.verify_quotes(answer, chunks)
    check(got["quote_failed"] is True, f"a quote cited against the wrong extract must fail, got {got}")
    check(got["quote_verified"] is False, f"got {got}")


def test_quote_absent_from_every_extract_fails():
    chunks = [{"doc_id": "a.1", "prefix": "", "text": "malloc returns NULL on failure."}]
    answer = '"this text is nowhere in the corpus" some claim. [1]'
    got = grammar.verify_quotes(answer, chunks)
    check(got["quote_failed"] is True, f"an absent quote must fail, got {got}")
    check(got["quote_verified"] is False, f"got {got}")


def test_quote_matches_after_whitespace_collapse():
    chunks = [{"doc_id": "a.1", "prefix": "NAME\n       malloc\n\n",
               "text": "   malloc  returns\n   NULL   on failure.  "}]
    answer = '"malloc returns NULL on failure" it returns NULL. [1]'
    got = grammar.verify_quotes(answer, chunks)
    check(got["quote_verified"] is True,
          f"a quote that matches only after whitespace collapse must verify, got {got}")
    check(got["quote_failed"] is False, f"got {got}")


def test_refusal_passes_through_unverified_and_unfailed():
    chunks = [{"doc_id": "a.1", "prefix": "", "text": "whatever"}]
    got = grammar.verify_quotes(grammar.REFUSAL, chunks)
    check(got["quote_verified"] is False, f"the refusal must not be quote_verified, got {got}")
    check(got["quote_failed"] is False, f"the refusal must not be quote_failed, got {got}")
    check(got["n_quotes"] == 0, f"the refusal carries no claims, got {got}")


def test_default_path_unchanged():
    question = "how do I list files?"
    chunks = [{"doc_id": "ls.1", "prefix": "", "text": "ls lists directory contents."}]

    got = build_prompt(question, chunks)
    want = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Manual page extracts:\n\n[1] ls.1\nls lists directory contents."
                                     f"\n\nQuestion: {question}"},
    ]
    check(got == want, f"build_prompt with no system= must build SYSTEM messages, got {got}")

    got_default = grammar.cited_answer(3)
    refs = " | ".join(f'"{i}"' for i in (1, 2, 3))
    want_template = grammar.TEMPLATE.replace("REFS", refs).strip() + "\n"
    check(got_default == want_template,
          f"cited_answer(3) must equal TEMPLATE with REFS '1'|'2'|'3', got {got_default!r}")

    # quoted_answer mirrors cited_answer's own REFS substitution - exercised
    # here rather than as a 7th test function (the task caps this file at 6).
    g2 = grammar.quoted_answer(2)
    check('"1" | "2"' in g2, f"quoted_answer(2) must inline refs 1 and 2, got {g2!r}")
    check("REFS" not in g2, f"REFS must be fully substituted, got {g2!r}")
    check(g2.startswith("root"), f"quoted_answer must return the grammar text, got {g2!r}")


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
