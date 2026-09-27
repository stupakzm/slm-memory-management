"""Tests for scripts/robustness_report.py (tsk_20260927_robreport, R3a):
turning one eval_answers.py run into the robustness map - kind assignment,
re-scored correctness with the base's alias fallback (blk_gold_aliases_procedure),
paired kept/lost/gained/net, the exact McNemar p, unanswerable abstention
pairing, the per-domain split, and the missing/duplicate qid guards.

Hermetic by design (blk_test_env_constraints): scripts/robustness_report.py
imports only smm.gold (stdlib-only, no sqlite_vec) beyond the stdlib, so no
stub is needed. Loaded via importlib, matching tests/test_gold.py and
tests/test_eval_answers.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "robustness_report", ROOT / "scripts" / "robustness_report.py")
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_kind_assignment():
    clean = {"qid": "a01", "variant_of": None, "paraphrase_of": None, "variant_kind": None}
    paraphrase = {"qid": "a01.p", "variant_of": None, "paraphrase_of": "a01",
                   "variant_kind": None}
    typo = {"qid": "a01.y1", "variant_of": "a01", "paraphrase_of": None,
            "variant_kind": "typo1"}
    no_name = {"qid": "a01.n", "variant_of": "a01", "paraphrase_of": None,
               "variant_kind": "no-name"}
    check(rr.classify_kind(clean) == "clean", "no variant_of/paraphrase_of must be clean")
    check(rr.classify_kind(paraphrase) == "paraphrase", "paraphrase_of set must be paraphrase")
    check(rr.classify_kind(typo) == "typo1", "variant_of set must use variant_kind as-is")
    check(rr.classify_kind(no_name) == "no-name", "variant_kind passes through unchanged")


def test_correctness_recomputed_with_alias_fallback():
    """A stored correct=False must never be trusted: the answer is rescored
    from its own text, and a typo variant with no alias entry of its own
    falls back to its base's aliases (gold.aliases_for) - a base alias that
    matches the variant's answer text must turn it correct even though the
    stored flag said otherwise."""
    pool = {
        "a01": {"qid": "a01", "kind": "answerable", "answer_contains": ["--no-clobber"],
                "variant_of": None, "paraphrase_of": None, "variant_kind": None},
        "a01.y1": {"qid": "a01.y1", "kind": "answerable", "answer_contains": ["--no-clobber"],
                   "variant_of": "a01", "paraphrase_of": None, "variant_kind": "typo1"},
    }
    # aliases keyed only by the base qid; a01.y1 has no entry of its own.
    aliases = {"a01": {"--no-clobber": [{"alias": "-n", "rule": "synonym"}]}}
    results = [
        {"qid": "a01", "kind": "answerable", "answer": "use -n to skip existing files",
         "abstained": False, "evidence_retrieved": True, "correct": False},
        {"qid": "a01.y1", "kind": "answerable", "answer": "use -n here too",
         "abstained": False, "evidence_retrieved": True, "correct": False},
    ]
    report = rr.build_report(pool, results, aliases)
    clean = report["kinds"]["clean"]["overall"]["answerable"]
    typo1 = report["kinds"]["typo1"]["overall"]["answerable"]
    check(clean["correct"] == 1, f"base should be recomputed correct via -n alias: {clean}")
    check(typo1["correct"] == 1,
          f"variant with no alias entry of its own should fall back to base's alias: {typo1}")


def test_paired_kept_lost_gained_counts():
    pairs = [
        (True, True),   # kept
        (True, True),   # kept
        (True, False),  # lost
        (False, True),  # gained
        (False, False),  # both_wrong
    ]
    got = rr.paired_counts(pairs)
    check(got["kept"] == 2, got)
    check(got["lost"] == 1, got)
    check(got["gained"] == 1, got)
    check(got["both_wrong"] == 1, got)
    check(got["net"] == 0, got)
    check(got["p"] == 1.0, got)


def test_mcnemar_p_known_values():
    check(abs(rr.mcnemar_p(0, 6) - 0.03125) < 1e-9,
          f"lost=0, gained=6 must give p=0.03125, got {rr.mcnemar_p(0, 6)}")
    check(rr.mcnemar_p(3, 3) == 1.0, f"lost==gained must give p=1.0, got {rr.mcnemar_p(3, 3)}")
    check(rr.mcnemar_p(0, 0) == 1.0, "no discordant pairs must give p=1.0")


def test_unanswerable_abstention_pairing():
    """lost = base abstained, variant answered (the typo broke abstention);
    gained = base answered, variant abstained."""
    pool = {
        "u01": {"qid": "u01", "kind": "unanswerable", "answer_contains": [],
                "variant_of": None, "paraphrase_of": None, "variant_kind": None},
        "u01.y1": {"qid": "u01.y1", "kind": "unanswerable", "answer_contains": [],
                   "variant_of": "u01", "paraphrase_of": None, "variant_kind": "typo1"},
    }
    results = [
        {"qid": "u01", "kind": "unanswerable", "answer": "I don't know.", "abstained": True},
        {"qid": "u01.y1", "kind": "unanswerable", "answer": "here is a made-up answer",
         "abstained": False},
    ]
    report = rr.build_report(pool, results, {})
    p = report["kinds"]["typo1"]["overall"]["paired"]["abstained"]
    check(p["lost"] == 1, f"base abstained, variant answered must be lost: {p}")
    check(p["gained"] == 0, p)
    check(p["kept"] == 0, p)


def test_per_domain_split():
    pool = {
        "l01": {"qid": "l01", "kind": "answerable", "answer_contains": ["x"], "domain": "linux",
                "variant_of": None, "paraphrase_of": None, "variant_kind": None},
        "e01": {"qid": "e01", "kind": "answerable", "answer_contains": ["y"], "domain": "emacs",
                "variant_of": None, "paraphrase_of": None, "variant_kind": None},
        "n01": {"qid": "n01", "kind": "answerable", "answer_contains": ["z"],
                "variant_of": None, "paraphrase_of": None, "variant_kind": None},
    }
    results = [
        {"qid": "l01", "kind": "answerable", "answer": "x here", "abstained": False,
         "evidence_retrieved": True},
        {"qid": "e01", "kind": "answerable", "answer": "no match", "abstained": False,
         "evidence_retrieved": False},
        {"qid": "n01", "kind": "answerable", "answer": "z here", "abstained": False,
         "evidence_retrieved": True},
    ]
    report = rr.build_report(pool, results, {})
    domains = report["kinds"]["clean"]["domains"]
    check(set(domains) == {"linux", "emacs"},
          f"a missing domain field must default to linux, got {sorted(domains)}")
    check(domains["linux"]["answerable"]["n"] == 2,
          f"l01 and n01 (defaulted) both fall under linux: {domains['linux']}")
    check(domains["emacs"]["answerable"]["n"] == 1, domains["emacs"])
    overall = report["kinds"]["clean"]["overall"]["answerable"]
    check(overall["n"] == 3, overall)
    check(overall["correct"] == 2, overall)


def test_missing_and_duplicate_qid_errors():
    pool = {"a01": {"qid": "a01", "kind": "answerable", "answer_contains": ["x"],
                     "variant_of": None, "paraphrase_of": None, "variant_kind": None}}
    missing_results = [{"qid": "does-not-exist", "kind": "answerable", "answer": "x",
                         "abstained": False, "evidence_retrieved": True}]
    try:
        rr.build_report(pool, missing_results, {})
        raise AssertionError("a qid missing from the pool must error out")
    except SystemExit:
        pass

    dup_results = [
        {"qid": "a01", "kind": "answerable", "answer": "x", "abstained": False,
         "evidence_retrieved": True},
        {"qid": "a01", "kind": "answerable", "answer": "x", "abstained": False,
         "evidence_retrieved": True},
    ]
    try:
        rr.build_report(pool, dup_results, {})
        raise AssertionError("a duplicate qid in the results must error out")
    except SystemExit:
        pass

    dup_pool_lines = [
        '{"qid": "x1", "kind": "answerable", "answer_contains": ["a"]}\n',
        '{"qid": "x1", "kind": "answerable", "answer_contains": ["b"]}\n',
    ]
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "pool.jsonl"
        p.write_text("".join(dup_pool_lines), encoding="utf-8")
        try:
            rr.load_pool([str(p)])
            raise AssertionError("a duplicate qid across pool files must error out")
        except SystemExit:
            pass


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
