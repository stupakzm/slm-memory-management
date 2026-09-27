"""Tests for scripts/make_variants.py (typo generation, R2a,
tsk_20260927_typos) and smm.gold.aliases_for, the variant-aware alias
lookup it introduces.

Hermetic by design (blk_test_env_constraints): this worktree's test-running
environment has no .venv, no models/, and must not assume data/eval/*.jsonl
is present either - every test builds its own tiny synthetic question set
in a tempdir. make_variants.py and smm.gold are both stdlib-only, so no
sqlite_vec stub is needed for either import (nothing here opens a database),
but one is installed anyway, matching the repo's test convention.
"""

from __future__ import annotations

import importlib.util
import json
import random
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
    "make_variants", ROOT / "scripts" / "make_variants.py")
mv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mv)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _write_jsonl(path: Path, rows: list) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


def _row(qid: str, question: str, **extra) -> dict:
    base = {
        "qid": qid, "question": question, "kind": "answerable",
        "doc": "tar.1", "answer_contains": ["--exclude-from"],
        "gold_sec_ids": ["tar.1#OPTIONS"], "gold_primary": "tar.1#OPTIONS",
        "tags": ["flag"], "paraphrase_of": None,
    }
    base.update(extra)
    return base


# A question with plenty of >= 4 letter words, none overlapping the gold
# token/hint, so every test below has an ample, unconstrained word pool.
_Q = ("I have a text file listing filenames I want left out of an archive. "
      "How do I build a tarball that skips all of them?")


def _sample_rows() -> list:
    return [
        _row("a01", _Q),
        _row("a02", "How do I compress an archive with gzip while creating it?",
             answer_contains=["--gzip"]),
        _row("u01", "How do I mount a host directory into a docker container?",
             kind="unanswerable", answer_contains=[], gold_sec_ids=[],
             doc=None, unanswerable_reason="tool-not-installed",
             unanswerable_detail="docker"),
    ]


def test_each_edit_op_is_a_real_distinct_change():
    """Every op in mv._APPLY, invoked directly, must change the word - and
    must change it the way its name promises: substitution keeps length and
    swaps one QWERTY-adjacent letter, deletion shortens by 1, doubling
    lengthens by 1, transposition swaps an adjacent pair without changing
    the multiset of letters."""
    rng = random.Random(0)
    word = "mistake"

    sub = mv._apply_sub(rng, word)
    check(len(sub) == len(word), f"sub must preserve length: {sub!r}")
    diffs = [i for i in range(len(word)) if sub[i] != word[i]]
    check(len(diffs) == 1, f"sub must change exactly one letter: {word!r} -> {sub!r}")
    check(diffs[0] != 0, "sub must never touch the first letter")
    i = diffs[0]
    check(sub[i] in mv.QWERTY_ADJ[word[i]],
          f"substituted letter {sub[i]!r} must be QWERTY-adjacent to {word[i]!r}")

    deleted = mv._apply_delete(rng, word)
    check(len(deleted) == len(word) - 1, f"delete must shorten by 1: {deleted!r}")
    check(deleted != word, "delete must change the word")

    doubled = mv._apply_double(rng, word)
    check(len(doubled) == len(word) + 1, f"double must lengthen by 1: {doubled!r}")
    check(doubled != word, "double must change the word")

    transposed = mv._apply_transpose(rng, "mistake")
    check(len(transposed) == len(word), "transpose must preserve length")
    check(sorted(transposed) == sorted(word), "transpose must preserve the letter multiset")
    check(transposed != word, "transpose must change the word")

    # QWERTY_ADJ never lists a letter as its own neighbour.
    for letter, neighbors in mv.QWERTY_ADJ.items():
        check(letter not in neighbors, f"{letter!r} must not neighbour itself")


def test_first_letter_and_short_words_never_edited():
    """Across every edit make_edits ever produces for a battery of rows and
    seeds, the first letter is untouched and the edited ('before') word is
    never shorter than the 2-letter fallback floor."""
    rows = _sample_rows()
    for row in rows:
        for seed_kind, n in (("typo1", 1), ("typo3", 3)):
            edits = mv.make_edits(row, n, seed_kind)
            for e in edits:
                check(e["before"][0] == e["after"][0],
                      f"{row['qid']}/{seed_kind}: first letter changed "
                      f"{e['before']!r} -> {e['after']!r}")
                check(len(e["before"]) >= 2,
                      f"{row['qid']}/{seed_kind}: edited word {e['before']!r} "
                      f"below the 2-letter floor")


def test_answer_token_and_gold_hint_words_never_edited():
    """A word that is a sub-word of an answer_contains token or a gold_hint
    string must never be chosen, even when it is otherwise the most
    tempting (long, early) candidate."""
    row = _row(
        "g01",
        "How do I exclude files listed in a manifest from the archive build?",
        answer_contains=["--exclude-from"], gold_hint=["skip files listed"])
    protected = mv._protected_words(row)
    check(protected == {"exclude", "from", "skip", "files", "listed"},
          f"unexpected protected set: {protected}")
    for seed_kind, n in (("typo1", 1), ("typo3", 3)):
        edits = mv.make_edits(row, n, seed_kind)
        for e in edits:
            check(e["before"].lower() not in protected,
                  f"{seed_kind}: edited a protected word {e['before']!r}")


def test_determinism_same_seed_same_output_order_independent():
    """Same (qid, question, answer_contains, gold_hint) gives byte-identical
    edits regardless of where the row sits in a list, or which unrelated
    rows surround it - the seed depends only on the row's own qid and the
    variant kind, never on position."""
    row_a = _row("a01", _Q)
    edits1 = mv.make_edits(row_a, 3, "typo3")
    edits2 = mv.make_edits(row_a, 3, "typo3")
    check(edits1 == edits2, "same row, same seed must give identical edits")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rows_order1 = [_row("a01", _Q), _row("a02", "How do I list files?")]
        rows_order2 = [_row("a02", "How do I list files?"), _row("a01", _Q)]
        q1 = _write_jsonl(tmp / "q1.jsonl", rows_order1)
        e1 = _write_jsonl(tmp / "e1.jsonl", [])
        q2 = _write_jsonl(tmp / "q2.jsonl", rows_order2)
        e2 = _write_jsonl(tmp / "e2.jsonl", [])
        out1 = {r["qid"]: r for r in mv.generate([q1, e1])}
        out2 = {r["qid"]: r for r in mv.generate([q2, e2])}
        check(out1["a01.y3"]["edits"] == out2["a01.y3"]["edits"],
              "row order must not affect a01's generated edits")
        check(out1["a01.y3"]["question"] == out2["a01.y3"]["question"],
              "row order must not affect a01's generated question")


def test_replay_reproduces_the_variant_question():
    row = _row("a01", _Q)
    variant = mv.build_variant(row, "y3", "typo3", "variant-typo3", 3)
    replayed = mv.apply_edits(row["question"], variant["edits"])
    check(replayed == variant["question"],
          f"replay must reproduce the variant exactly: {replayed!r} != "
          f"{variant['question']!r}")


def test_check_catches_a_tampered_row():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rows = _sample_rows()
        q_path = _write_jsonl(tmp / "questions.jsonl",
                               [r for r in rows if r["qid"].startswith("a")])
        e_path = _write_jsonl(tmp / "emacs_questions.jsonl",
                               [r for r in rows if r["qid"].startswith("u")])
        variants = mv.generate([q_path, e_path])
        out_path = _write_jsonl(tmp / "variations.jsonl", variants)

        ok = mv.check(out_path, bases=[q_path, e_path])
        check(ok == 0, "check must pass on an untampered file")

        tampered = json.loads(out_path.read_text().splitlines()[0])
        tampered["question"] = tampered["question"] + " extra"
        lines = out_path.read_text().splitlines()
        lines[0] = json.dumps(tampered)
        out_path.write_text("\n".join(lines) + "\n")

        bad = mv.check(out_path, bases=[q_path, e_path])
        check(bad == 1, "check must fail (exit 1) on a tampered question")


def test_aliases_for_own_entry_fallback_and_none():
    aliases = {"a01": {"--exclude-from": [{"alias": "-X", "rule": "synonym"}]},
               "a02": {}}
    # own entry present (non-empty) -> returned, base ignored.
    check(gold.aliases_for(aliases, "a01", "a99") == aliases["a01"],
          "a qid with its own entry must never fall back")
    # own entry present but EMPTY -> still its own (no fallback for a
    # present-but-empty key).
    check(gold.aliases_for(aliases, "a02", "a01") == {},
          "a present, empty entry must not fall back to the base")
    # missing own entry, base present -> base's entry.
    check(gold.aliases_for(aliases, "a01.y1", "a01") == aliases["a01"],
          "a variant with no entry of its own must fall back to its base's")
    # missing own entry, no base given / base also missing -> {}.
    check(gold.aliases_for(aliases, "zzz") == {},
          "no qid, no base must give {}")
    check(gold.aliases_for(aliases, "zzz", "also-missing") == {},
          "missing base entry must give {}, not raise")


def test_y3_edits_land_in_three_distinct_words():
    rows = _sample_rows()
    for row in rows:
        edits = mv.make_edits(row, 3, "typo3")
        check(len(edits) == 3, f"{row['qid']}: typo3 must have exactly 3 edits")
        indices = {e["word_index"] for e in edits}
        check(len(indices) == 3,
              f"{row['qid']}: typo3 edits must land in 3 distinct words, got {edits}")


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
