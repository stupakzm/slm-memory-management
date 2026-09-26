"""Tests for scripts/eval_answers.py's abstention and evidence scoring
(tsk_20260926_0354593a).

Two bugs fixed here:
  - abstained(text) used to fire if ABSTAIN_RE matched ANYWHERE in the text,
    so a hedged answer that states a real, cited claim and then appends
    "I don't know. [n]" was scored as a refusal (blk_abstain_re_false_positives).
    Fixed: abstained() now strips citation markers, splits into sentences, and
    returns True only if EVERY sentence is a refusal.
  - evidence_retrieved used a strict substring check while `correct` is scored
    via gold.is_correct (which also matches gold-token aliases), so an answer
    could be correct via an alias whose evidence was in context yet be scored
    "no evidence" (blk_evidence_retrieved_alias_mismatch). Fixed: evidence_in()
    returns (aliased, strict) so both can be inspected.

Hermetic by design (blk_test_env_constraints): this worktree has no .venv, no
data/, no models/, and `scripts.eval_answers` imports `smm.store`, which
imports the third-party `sqlite_vec` (absent here). `sqlite_vec` is stubbed in
sys.modules before the import below, exactly as tests/test_eval_answers.py
does; nothing here ever opens a real sqlite3 connection or calls into the
stub, so its contents don't matter, only its presence. The module is then
loaded via importlib so this file works with no changes to sys.path beyond
that stub.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

if "sqlite_vec" not in sys.modules:
    stub = types.ModuleType("sqlite_vec")
    stub.load = lambda *a, **kw: None  # never called: no real db is opened here
    sys.modules["sqlite_vec"] = stub

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "eval_answers", ROOT / "scripts" / "eval_answers.py")
eval_answers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_answers)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_cited_claim_then_i_dont_know_is_not_abstained():
    """A real, cited claim followed by a hedged "I don't know. [1]" is still
    an answer - not every sentence is a refusal."""
    text = "Use `-r` [2] to recurse into subdirectories. I don't know. [1]"
    check(not eval_answers.abstained(text),
          "a cited claim followed by a hedge must not count as an abstention")


def test_bare_i_dont_know_is_abstained():
    """A text with no other sentence, only the refusal itself, must abstain."""
    text = "I don't know."
    check(eval_answers.abstained(text),
          "a bare 'I don't know.' must count as an abstention")


def test_uncited_extracts_do_not_contain_is_abstained():
    text = "The extracts do not contain the answer."
    check(eval_answers.abstained(text),
          "an uncited 'the extracts do not contain' sentence must count as an abstention")


def test_evidence_in_via_alias_only():
    """Gold token '--human-readable' is absent verbatim, but its alias '-h'
    appears in a hit's text on option-token boundaries: aliased is True,
    strict is False."""
    toks = ["--human-readable"]
    token_aliases = {"--human-readable": [{"alias": "-h"}]}
    hits = [{"text": "ls -h file lists sizes in human readable form"}]

    aliased, strict = eval_answers.evidence_in(hits, toks, token_aliases)

    check(aliased is True, f"alias '-h' present on boundaries should give aliased=True, got {aliased}")
    check(strict is False, f"'--human-readable' is absent verbatim, strict should be False, got {strict}")


def test_evidence_in_verbatim_token():
    """Gold token present verbatim in a hit: both aliased and strict are True."""
    toks = ["--verbose"]
    hits = [{"text": "Use --verbose to enable detailed output"}]

    aliased, strict = eval_answers.evidence_in(hits, toks, {})

    check(aliased is True, f"verbatim token should give aliased=True, got {aliased}")
    check(strict is True, f"verbatim token should give strict=True, got {strict}")


def test_evidence_in_no_evidence():
    """Gold token absent everywhere, no matching alias either: both False."""
    toks = ["--human-readable"]
    token_aliases = {"--human-readable": [{"alias": "-h"}]}
    hits = [{"text": "this hit is about something else entirely"}]

    aliased, strict = eval_answers.evidence_in(hits, toks, token_aliases)

    check(aliased is False, f"no evidence anywhere should give aliased=False, got {aliased}")
    check(strict is False, f"no evidence anywhere should give strict=False, got {strict}")


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
