"""Tests for smm.extract_view and scripts/eval_answers.py --read-view repaired.

Hermetic by design (blk_test_env_constraints): stdlib only, no db - the
previous-window lookup is a dict. `sqlite_vec` is stubbed before eval_answers
is imported, exactly as tests/test_eval_answers.py does.

The scenario throughout is a man page cut into windows: window n+1 starts
inside a line of window n, so it has lost the option header it describes.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import re
import sys
import types
from contextlib import redirect_stderr
from pathlib import Path

if "sqlite_vec" not in sys.modules:
    stub = types.ModuleType("sqlite_vec")
    stub.load = lambda *a, **kw: None  # never called: no real db is opened here
    sys.modules["sqlite_vec"] = stub

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import extract_view  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "eval_answers", ROOT / "scripts" / "eval_answers.py")
eval_answers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_answers)

PREV = (
    "-n, --lines=[-]NUM\n"
    "       output the first NUM lines instead of the first 10; with the leading '-',\n"
    "       print all but the last NUM lines of each file\n"
    "\n"
    "-q, --quiet, --silent\n"
    "       never print headers giving file names, and nothing else, whatever the number of files"
)
EXTRA = "\n\n-v, --verbose\n       always print headers giving file names, for every file"


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def hit(ord_, text, doc="head.1", **kw):
    h = {"chunk_id": f"{doc}:{ord_}", "doc_id": doc, "ord": str(ord_),
         "prefix": f"{doc}(1) - output the first part of files\n", "text": text,
         "sec_id": "s1", "tag": "man", "rerank_score": 0.9}
    h.update(kw)
    return h


def next_window(cut_at):
    """The window after PREV that starts at PREV[cut_at:] (stripped, like the index)."""
    return (PREV[cut_at:] + EXTRA).strip()


def lookup_for(prev_text, doc="head.1", ord_=7):
    table = {(doc, ord_): prev_text}
    return lambda d, o: table.get((d, o))


def test_window_at_line_start_unchanged():
    t = next_window(PREV.index("-q, --quiet"))
    h = hit(8, t)
    out = extract_view.repair_extract(h, PREV)
    check(out == h, f"a window opening on a line start must be unchanged: {out['text']!r}")


def test_mid_line_fragment_is_completed_from_previous_window():
    cut = PREV.index("headers giving")
    out = extract_view.repair_extract(hit(8, next_window(cut)), PREV)
    check("never print headers giving file names" in out["text"], out["text"])
    check(out["text"].endswith(next_window(cut)), "the original window must be kept whole")
    check(out["text"].count("never print") == 1, out["text"])


def test_option_line_is_put_above_an_indented_description():
    cut = PREV.index("never print")  # stripped window: indentation lost, line start kept
    out = extract_view.repair_extract(hit(8, next_window(cut)), PREV)
    first, second = out["text"].split("\n")[:2]
    check(first == "-q, --quiet, --silent", f"option line must be on top: {first!r}")
    check(second.startswith("       never print"), f"indentation must be kept: {second!r}")
    mid = extract_view.repair_extract(hit(8, next_window(PREV.index("headers giving"))), PREV)
    check(mid["text"].startswith("-q, --quiet, --silent\n       never print headers"),
          mid["text"])


def test_section_heading_used_when_no_option_line():
    prev = ("DESCRIPTION\n"
            "       Print the first 10 lines of each FILE to standard output.  With more than\n"
            "       one FILE, precede each with a header giving the file name.")
    cut = prev.index("one FILE")
    t = (prev[cut:] + "\n       With no FILE, or when FILE is -, read standard input.").strip()
    out = extract_view.repair_extract(hit(8, t), prev)
    check(out["text"] == "DESCRIPTION\n       " + t, f"heading + indentation + window: {out['text']!r}")


def test_alignment_failure_leaves_extract_unchanged():
    h = hit(8, "this text does not occur anywhere in the previous window at all")
    check(extract_view.repair_extract(h, PREV) == h, "not found must be unchanged")
    check(extract_view.repair_extract(h, None) == h, "unknown previous must be unchanged")
    short = hit(8, "never print")  # fewer than 20 characters cannot be located
    check(extract_view.repair_extract(short, PREV) == short, "too short must be unchanged")


def test_first_window_of_a_page_unchanged():
    calls = []

    def lookup(d, o):
        calls.append((d, o))
        return PREV

    h = hit(0, next_window(PREV.index("headers giving")))
    out = extract_view.repair_hits([h], lookup)
    check(out == [h], f"ord 0 has no previous window: {out}")
    check(calls == [], f"ord 0 must not be looked up: {calls}")


def test_consecutive_windows_merge_without_repeating_overlap():
    w8 = next_window(PREV.index("headers giving"))
    out = extract_view.repair_hits([hit(7, PREV), hit(8, w8)],
                                   lookup_for(PREV))
    check(len(out) == 1, f"consecutive windows must merge: {len(out)}")
    check(out[0]["text"] == PREV + EXTRA, f"overlap repeated or text lost: {out[0]['text']!r}")
    check(out[0]["text"].count("never print headers giving file names") == 1,
          "the overlap must appear once")
    # plain windows (no repair needed): the overlap is dropped from the second
    merged = extract_view.merge_consecutive([hit(7, "A" * 30 + "x" * 40),
                                             hit(8, "x" * 40 + "B" * 30)])
    check(merged[0]["text"] == "A" * 30 + "x" * 40 + "B" * 30, merged[0]["text"])
    # no overlap at all: joined by a newline
    far = extract_view.merge_consecutive([hit(7, "first window text here"),
                                          hit(8, "second window other words")])
    check(far[0]["text"] == "first window text here\nsecond window other words", far[0]["text"])


def test_non_consecutive_windows_stay_separate():
    a, b, c = hit(7, "a" * 40), hit(9, "b" * 40), hit(8, "c" * 40, doc="tail.1")
    out = extract_view.merge_consecutive([a, b, c])
    check([h["chunk_id"] for h in out] == ["head.1:7", "head.1:9", "tail.1:8"],
          f"order or membership changed: {[h['chunk_id'] for h in out]}")
    check([h["text"] for h in out] == [a["text"], b["text"], c["text"]], "texts must be kept")


def test_merge_keeps_rank_order_of_first_member():
    hits = [hit(8, "x" * 30 + "B" * 30, rerank_score=0.9),
            hit(1, "other page text", doc="other.1", rerank_score=0.8),
            hit(7, "A" * 30 + "x" * 30, rerank_score=0.7)]
    out = extract_view.merge_consecutive(hits)
    check([h["doc_id"] for h in out] == ["head.1", "other.1"], f"{[h['doc_id'] for h in out]}")
    check(out[0]["chunk_id"] == "head.1:8" and out[0]["rerank_score"] == 0.9,
          f"keys must be the earlier-ranked member's: {out[0]}")
    check(out[0]["text"] == "A" * 30 + "x" * 30 + "B" * 30,
          f"the lower ord must come first: {out[0]['text']!r}")


def test_prefix_and_other_keys_preserved():
    h = hit(8, next_window(PREV.index("headers giving")))
    out = extract_view.repair_extract(h, PREV)
    check(out["text"] != h["text"], "this case must be repaired")
    for k in h:
        if k != "text":
            check(out[k] == h[k], f"key {k} changed: {out[k]!r}")
    check(set(out) == set(h), "no key may be added or dropped")
    merged = extract_view.repair_hits([hit(7, PREV), h], lookup_for(PREV))
    check(merged[0]["prefix"] == h["prefix"] and merged[0]["sec_id"] == "s1"
          and merged[0]["tag"] == "man", f"merge dropped keys: {merged[0]}")


def test_repair_does_not_mutate_input():
    hits = [hit(7, PREV), hit(8, next_window(PREV.index("headers giving")))]
    before = copy.deepcopy(hits)
    extract_view.repair_extract(hits[1], PREV)
    extract_view.merge_consecutive(hits)
    out = extract_view.repair_hits(hits, lookup_for(PREV))
    check(hits == before, "input hits were mutated")
    check(out[0] is not hits[0], "output must be new dicts")


def test_read_view_flag_default_raw_and_config_key():
    check(eval_answers.read_view_config("raw") == {}, "raw must add no config key")
    check(eval_answers.read_view_config("repaired") == {"read_view": "repaired"},
          "repaired must record read_view")
    hits = [hit(8, next_window(PREV.index("headers giving")))]
    check(eval_answers.read_view_hits(hits, "raw", None) is hits, "raw must return hits itself")
    check(eval_answers.read_view_hits(hits, "repaired", lookup_for(PREV))[0]["text"]
          != hits[0]["text"], "repaired must change the text")
    try:
        eval_answers.read_view_hits(hits, "bogus", None)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown view must raise ValueError")
    src = (ROOT / "scripts" / "eval_answers.py").read_text()
    check(re.search(r'"--read-view",\s*choices=\("raw", "repaired"\),\s*default="raw"', src),
          "--read-view must default to raw")
    old, err = sys.argv, io.StringIO()
    sys.argv = ["eval_answers.py", "--read-view", "repaired", "--cascade"]
    try:
        with redirect_stderr(err):
            rc = eval_answers.main()
    finally:
        sys.argv = old
    check(rc == 2, f"repaired + --cascade must return 2, got {rc}")
    check("--read-view" in err.getvalue() and "--cascade" in err.getvalue(),
          f"message must name both flags: {err.getvalue()!r}")


def test_evidence_scoring_uses_original_hits():
    # the answer token only exists in the header the repair puts on top
    hits = [hit(8, next_window(PREV.index("headers giving")))]
    toks = ["--silent"]
    read = eval_answers.read_view_hits(hits, "repaired", lookup_for(PREV))
    check(eval_answers.evidence_in(read, toks, {})[1] is True, "the read view must carry it")
    check(eval_answers.evidence_in(hits, toks, {})[1] is False,
          "the original hits must not (evidence is scored on them)")
    # the generate loop scores the selection, not the reader's view
    src = (ROOT / "scripts" / "eval_answers.py").read_text()
    loop = src.split("# --- stage 2: generation only", 1)[1].split("return write_report", 1)[0]
    check("evidence_in(scored_hits," in loop and "evidence_in(hits," not in loop,
          "evidence_in must read the original selection in the generate loop")
    check('[h["doc_id"] for h in scored_hits]' in loop, "retrieved_docs must read the original")
    check("gate_score(gate_hits)" in loop, "the gate must still read gate_hits")


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
