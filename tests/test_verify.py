"""Tests for smm.verify (phase 6: the verifier) and the pure, unit-testable
pieces of scripts/eval_verifier.py's --stage replay.

Hermetic by design (blk_test_env_constraints): this worktree has no .venv, no
data/, no models/. smm.verify itself only imports stdlib + smm.grammar, so it
needs no stub - but scripts/eval_verifier.py is loaded via importlib exactly
as tests/test_eval_answers.py loads scripts/eval_answers.py, and `sqlite_vec`
is stubbed in sys.modules first, the way tests/test_retrieve.py does, so that
convention holds even though this file's own imports don't require it.
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

from smm import grammar  # noqa: E402
from smm.verify import anchors, claims, lexical_support, score_answer  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "eval_verifier", ROOT / "scripts" / "eval_verifier.py")
eval_verifier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_verifier)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_claims_split():
    got = claims("Use -C. [2] Or `grep -A 3`. [1]")
    check(got == [("Use -C.", [2]), ("Or `grep -A 3`.", [1])],
          f"expected 2 claims citing [2] then [1], got {got}")
    check(claims(grammar.REFUSAL) == [], "the refusal must carry no claims")


def test_anchors_extracted():
    text = "Use `grep -A 3` or --context=NUM to show lines around a match, please."
    got = anchors(text)
    check("grep -A 3" in got, f"backtick span missing: {got}")
    check("--context" in got, f"--context=NUM must strip to --context: {got}")
    check("-A" in got, f"the flag inside the backtick span must also be found: {got}")
    for prose in ("Use", "or", "to", "show", "lines", "around", "match", "please"):
        check(prose not in got, f"plain prose word wrongly extracted as an anchor: {prose}")


def test_lexical_support():
    check(lexical_support("Use `-C` and `-3`.", "the flags are -C and -3 here") == 1.0,
          "all anchors present should score 1.0")
    check(lexical_support("Use `-C` and `-3`.", "nothing relevant here") == 0.0,
          "no anchors present should score 0.0")
    check(lexical_support("Use `-C` and `-3`.", "only -C appears here") == 0.5,
          "half the anchors present should score 0.5")
    check(lexical_support("just prose, no anchors here", "anything") is None,
          "a claim with no anchors has no opinion (None)")


def test_answer_score_is_min_over_claims():
    answer = "First claim. [1] Second claim. [2] Third claim. [3]"
    chunks = [{"text": "e1"}, {"text": "e2"}, {"text": "e3"}]
    fake_scores = {"First claim.": 0.9, "Second claim.": 0.2, "Third claim.": None}

    def fake_scorer(claim_text, extract_text):
        return fake_scores[claim_text]

    got = score_answer(answer, chunks, fake_scorer)
    check(got == 0.2, f"answer score must be the min over non-None claims, got {got}")


def test_out_of_range_citation_scores_zero():
    answer = "A claim citing an extract that was never shown. [5]"
    chunks = [{"text": "e1"}, {"text": "e2"}, {"text": "e3"}]
    got = score_answer(answer, chunks, lambda c, e: 1.0)
    check(got == 0.0, f"an all-out-of-range citation must score 0.0, got {got}")


def test_folds_keep_variants_together():
    fold_of = eval_verifier.fold_of
    for base in ("a11", "a04", "u09", "p03"):
        variants = [base, f"{base}.t", f"{base}.n", f"{base}.z"]
        folds = {fold_of(q) for q in variants}
        check(len(folds) == 1, f"{base}'s variants must share one fold, got {folds}")
    check(fold_of("a11") == fold_of("a11"), "fold_of must be stable across calls")
    check(fold_of("a11.t") in (0, 1), "fold_of must return 0 or 1")


def test_replay_counts():
    replay = eval_verifier.replay
    # 2 good (answerable + correct), 2 bad (one wrong answerable, one
    # unanswerable the model spoke on). Scores chosen so t=0.5 removes exactly
    # the two bad ones and keeps both good ones.
    records = [
        {"qid": "a01", "kind": "answerable", "correct": True},
        {"qid": "a02", "kind": "answerable", "correct": True},
        {"qid": "a03", "kind": "answerable", "correct": False},
        {"qid": "u01", "kind": "unanswerable", "correct": False},
    ]
    scores = {"a01": 0.9, "a02": 0.8, "a03": 0.3, "u01": 0.1}

    got = replay(records, scores, 0.5)
    check(got == {"kept_good": 2, "removed_good": 0, "kept_bad": 0, "removed_bad": 2},
          f"threshold 0.5 should keep both good and remove both bad, got {got}")

    off = replay(records, scores, "off")
    check(off == {"kept_good": 2, "removed_good": 0, "kept_bad": 2, "removed_bad": 0},
          f"'off' must remove nothing, got {off}")


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
