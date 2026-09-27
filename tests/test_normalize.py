"""Tests for src/smm/normalize.py and its two CLI-adjacent call sites
(tsk_20260927_spellnorm, phase 11 R4a build): the spelling-normalisation
rules pre-registered in docs/phase11-results.md's "R4a pre-registration" -
distance bands, tie-break order, the unprotected first letter, untouchable
flag/path/digit/KEY=VALUE tokens, in-vocab and short-word passthrough, case
and punctuation preservation, `build_vocab`'s min-count and cache freshness,
the eval-path helper's `--normalize off` identity, and
scripts/robustness_report.py's `--overlay` merge.

Hermetic by design (blk_test_env_constraints): src/smm/normalize.py is
stdlib-only (no sqlite_vec - it opens the index with plain `sqlite3`, see
the module docstring), so no stub is needed; the one sqlite db this file
touches is a tiny fixture built in a tempdir. scripts/robustness_report.py
is likewise stdlib-only (it imports only smm.gold beyond the stdlib), loaded
via importlib exactly like tests/test_robustness_report.py.

Distances used below were computed once with the real `osa_distance` and
are asserted as fixed facts here, not re-derived at test time - see the
task report for how each was checked:
  corl/cork = 1     cozy/cork = 2     conputir/computer = 2
  conpuzir/computer = 3
"""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import normalize  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "robustness_report", ROOT / "scripts" / "robustness_report.py")
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_distance_bands():
    """4-7 letters: distance <= 1 only. 8+ letters: distance <= 2. A word
    one band-width over its cap must be left untouched."""
    vocab4_7 = {"cork": 5}
    out, edits = normalize.normalize("a corl b", vocab4_7)
    check(out == "a cork b", f"4-7 band: distance 1 must correct: {out!r}")
    check(edits == [{"from": "corl", "to": "cork", "distance": 1}], edits)

    out, edits = normalize.normalize("a cozy b", vocab4_7)
    check(out == "a cozy b",
          f"4-7 band: distance 2 (cozy/cork) exceeds max_dist=1, must be left alone: {out!r}")
    check(edits == [], edits)

    vocab8 = {"computer": 5}
    out, edits = normalize.normalize("my conputir works", vocab8)
    check(out == "my computer works", f"8+ band: distance 2 must correct: {out!r}")
    check(edits == [{"from": "conputir", "to": "computer", "distance": 2}], edits)

    out, edits = normalize.normalize("my conpuzir works", vocab8)
    check(out == "my conpuzir works",
          f"8+ band: distance 3 (conpuzir/computer) exceeds max_dist=2, must be left alone: {out!r}")
    check(edits == [], edits)


def test_tie_break_order():
    """(distance, NOT keyboard-adjacent substitution, -frequency, word):
    keyboard adjacency outranks frequency, and frequency outranks the final
    alphabetical tie-break."""
    # 'l' is QWERTY-adjacent to 'k' (cork) but not to 'n' (corn) or 'b' (corb).
    vocab_adjacency = {"corn": 50, "corb": 100, "cork": 3}
    out, _ = normalize.normalize("a corl b", vocab_adjacency)
    check(out == "a cork b",
          f"a keyboard-adjacent substitution must win over two more frequent, "
          f"non-adjacent candidates at the same distance: {out!r}")

    # 'x' is adjacent to neither 'n' (corn) nor 't' (cort): frequency decides.
    vocab_freq = {"corn": 5, "cort": 50}
    out, _ = normalize.normalize("a corx b", vocab_freq)
    check(out == "a cort b",
          f"with no keyboard-adjacent candidate, the more frequent word must win: {out!r}")

    # equal frequency, neither adjacent: alphabetically smaller word wins.
    vocab_alpha = {"corn": 50, "cort": 50}
    out, _ = normalize.normalize("a corx b", vocab_alpha)
    check(out == "a corn b",
          f"equal distance, adjacency and frequency must fall back to the word itself: {out!r}")


def test_first_letter_not_protected():
    """The typo generator never touches a word's first letter
    (blk_make_variants_typo_procedure), but the normaliser must still fix
    one there - protecting it would tune the mechanism to the generator's
    own habits, not to people (docs/phase11-results.md)."""
    vocab = {"computer": 5}
    out, edits = normalize.normalize("vomputer science", vocab)
    check(out == "computer science", f"a first-letter typo must still be corrected: {out!r}")
    check(edits == [{"from": "vomputer", "to": "computer", "distance": 1}], edits)


def test_untouchable_tokens():
    """A digit, '-', '/', '=', '.', '_' or '~' anywhere in a token's core
    (after stripping only ordinary sentence punctuation) makes it exact:
    flags, paths, KEY=VALUE and versions are never touched, even when a
    plausible correction sits right next to them in the vocabulary."""
    vocab = {"clobber": 5, "passwd": 5, "value": 5, "version": 5}
    for question, unchanged_token in [
        ("run --no-clobbar now", "--no-clobbar"),   # embedded '-': flag
        ("open /etc/passwd/ file", "/etc/passwd/"),  # embedded '/': path
        ("set KEY=VALUR now", "KEY=VALUR"),          # embedded '=': KEY=VALUE
        ("using v1.2.3 today", "v1.2.3"),            # digits and '.': version
    ]:
        out, edits = normalize.normalize(question, vocab)
        check(out == question,
              f"{unchanged_token!r} must never be touched, got {out!r}")
        check(edits == [], f"{unchanged_token!r} must produce no edits: {edits}")


def test_in_vocab_and_short_words_untouched():
    vocab = {"computer": 5, "cat": 9}  # 'cat' is 3 letters: never a candidate regardless
    out, edits = normalize.normalize("the computer and a cat sat", vocab)
    check(out == "the computer and a cat sat", out)
    check(edits == [], edits)

    out2, edits2 = normalize.normalize("a big dog ran", vocab)
    check(out2 == "a big dog ran",
          f"'dog'/'ran' are short (< 4 letters) and must never be candidates: {out2!r}")
    check(edits2 == [], edits2)


def test_case_and_punctuation_preserved():
    vocab = {"computer": 5}
    out, _ = normalize.normalize("Vomputer, science?", vocab)
    check(out == "Computer, science?",
          f"Capitalised case and trailing punctuation must both be preserved: {out!r}")

    out2, _ = normalize.normalize("VOMPUTER!", vocab)
    check(out2 == "COMPUTER!", f"ALL-CAPS must be preserved: {out2!r}")

    out3, _ = normalize.normalize('"vomputer"', vocab)
    check(out3 == '"computer"',
          f"surrounding quotes (ordinary punctuation) must be preserved: {out3!r}")


def test_build_vocab_min_count_and_cache():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "tiny.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE chunks (chunk_id TEXT, doc_id TEXT, prefix TEXT, "
            "text TEXT, domain TEXT)")
        conn.executemany(
            "INSERT INTO chunks (chunk_id, doc_id, prefix, text, domain) VALUES (?,?,?,?,?)",
            [
                ("c1", "d1", "", "Computer computer tarball", "linux"),
                ("c2", "d1", "", "an id ox", "linux"),
            ],
        )
        conn.commit()
        conn.close()

        cache_path = Path(td) / "tiny.db.vocab.json"
        vocab = normalize.build_vocab(db_path, cache_path=cache_path)
        check(vocab.get("computer") == 2,
              f"lower-cased, count >= 2 across chunks must be kept: {vocab}")
        check("tarball" not in vocab, f"count == 1 must be dropped: {vocab}")
        check(all(len(w) >= 3 for w in vocab), f"words shorter than 3 letters excluded: {vocab}")
        check(cache_path.exists(), "the cache file must be written next to the db")

        # A cache NEWER than the db must be read as-is, not rescanned.
        cache_path.write_text(json.dumps({"planted": 99}), encoding="utf-8")
        newer = db_path.stat().st_mtime + 10
        os.utime(cache_path, (newer, newer))
        reloaded = normalize.build_vocab(db_path, cache_path=cache_path)
        check(reloaded == {"planted": 99},
              f"a cache newer than the db must be read verbatim: {reloaded}")

        # A cache OLDER than the db must be rescanned.
        older = db_path.stat().st_mtime - 10
        os.utime(cache_path, (older, older))
        rescanned = normalize.build_vocab(db_path, cache_path=cache_path)
        check("computer" in rescanned and "planted" not in rescanned,
              f"a cache older than the db must be rescanned: {rescanned}")


def test_normalize_query_eval_path_helper():
    """scripts/eval_answers.py and scripts/ask.py's single call site:
    mode='off' must return the question completely unchanged, with no
    edits - byte-identical to before this feature existed."""
    vocab = {"computer": 5}
    q = "vomputer science now"
    out, edits = normalize.normalize_query(q, "off", vocab)
    check(out == q, f"'off' must return the question unchanged: {out!r}")
    check(edits == [], f"'off' must return no edits: {edits}")

    # sanity: 'spell' against the very same vocab DOES change it, so 'off'
    # above isn't just an accident of a no-op normalize().
    out2, edits2 = normalize.normalize_query(q, "spell", vocab)
    check(out2 == "computer science now", f"'spell' must actually normalise: {out2!r}")
    check(len(edits2) == 1, edits2)

    try:
        normalize.normalize_query(q, "spell", None)
        raise AssertionError("mode='spell' without a vocab must raise")
    except ValueError:
        pass


def test_robustness_report_overlay():
    """--overlay REPLACES --answers results with the same qid, in the
    base's own order; an overlay qid absent from --answers must error."""
    base_results = [
        {"qid": "a01", "answer": "orig-a"},
        {"qid": "a02", "answer": "orig-b"},
    ]
    overlay_results = [{"qid": "a01", "answer": "new-a"}]
    merged = rr.apply_overlay(base_results, overlay_results)
    check(merged == [{"qid": "a01", "answer": "new-a"}, {"qid": "a02", "answer": "orig-b"}],
          f"only the matching qid must be replaced, base order preserved: {merged}")

    try:
        rr.apply_overlay(base_results, [{"qid": "does-not-exist", "answer": "x"}])
        raise AssertionError("an overlay qid missing from --answers must error")
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
