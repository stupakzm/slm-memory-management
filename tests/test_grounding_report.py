"""Tests for scripts/grounding_report.py (ungrounded-identifier scorer).

Hermetic (blk_test_env_constraints): stdlib only, tempdir fixtures, never
data/eval/results/. sqlite_vec is stubbed first, matching the repo convention.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import types
from pathlib import Path

if "sqlite_vec" not in sys.modules:
    stub = types.ModuleType("sqlite_vec")
    stub.load = lambda *a, **kw: None
    sys.modules["sqlite_vec"] = stub

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "grounding_report", ROOT / "scripts" / "grounding_report.py")
gr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gr)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def erow(qid, kind="answerable", tok=("alpha",)):
    return {"qid": qid, "kind": kind, "answer_contains": list(tok)}


def hit(text, prefix="", ord_="0", doc_id="d"):
    return {"prefix": prefix, "text": text, "ord": ord_, "doc_id": doc_id}


def arow(qid, answer, abstained=False):
    return {"qid": qid, "answer": answer, "abstained": abstained}


def setup(tmp, eval_rows, runs, caches, config=None):
    tmp = Path(tmp)
    ev = tmp / "eval.jsonl"
    ev.write_text("\n".join(json.dumps(r) for r in eval_rows) + "\n")
    for name, rows in runs.items():
        (tmp / f"{name}-answers.json").write_text(
            json.dumps({"config": config or {}, "results": rows}))
    for name, cache in caches.items():
        (tmp / f"{name}-retrieved.json").write_text(json.dumps(cache))
    al = tmp / "aliases.json"
    al.write_text("{}")
    return ev, al


def run_main(tmp, ev, al, runs, extra=()):
    argv = ["--eval", str(ev), "--aliases", str(al)]
    for r in runs:
        argv += ["--run", r]
    argv += list(extra)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = gr.main(argv, results_dir=tmp)
    return code, out.getvalue()


def test_draft_pattern_flags_hyphenated_english():
    check("null-separated" in gr.draft_idents("Use null-separated output."), "draft")
    check("read-only" in gr.draft_idents("It is read-only."), "draft")


def test_tight_pattern_ignores_hyphenated_english():
    got = gr.tight_idents("Use null-separated, colon-separated and read-only output.")
    check(got == set(), got)


def test_tight_pattern_finds_flags_keys_and_mx_commands():
    got = gr.tight_idents("Run --max-depth=2 with -n, press C-x C-f, or M-x dired-jump.")
    for want in ("--max-depth", "-n", "C-x", "C-f", "M-x", "dired-jump"):
        check(want in got, (want, got))


def test_tight_pattern_backticked_long_single_dash_flag():
    check("-name" in gr.tight_idents("Use `-name` to match."), "ticked")
    check("-name" not in gr.tight_idents("Use -name to match."), "unticked")


def test_citations_stripped_before_extraction():
    check(gr.tight_idents("Use -n[1] here.") == {"-n"}, gr.tight_idents("Use -n[1] here."))
    check("[1]" not in "".join(gr.draft_idents("see C-x[2] now")), "draft")


def test_grounded_means_substring_of_prefix_plus_text():
    with tempfile.TemporaryDirectory() as tmp:
        ev, al = setup(tmp, [erow("q1", tok=("zzz",)), erow("q2", tok=("zzz",))],
                       {"r": [arow("q1", "Use --frob."), arow("q2", "Use --frob.")]},
                       {"r": {"q1": [hit("body", prefix="the --frob flag. ")],
                              "q2": [hit("body")]}})
        code, out = run_main(tmp, ev, al, ["r"])
        check(code == 0, out)
        check("r [tight] correct 0/0 wrong 1/2 unanswerable-answered 0/0" in out, out)


def test_buckets_exclude_abstained_rows():
    with tempfile.TemporaryDirectory() as tmp:
        ev, al = setup(tmp, [erow("q1"), erow("q2")],
                       {"r": [arow("q1", "alpha --x", True), arow("q2", "--x")]},
                       {"r": {"q1": [hit("b")], "q2": [hit("b")]}})
        code, out = run_main(tmp, ev, al, ["r"])
        check("r [tight] correct 0/0 wrong 1/1 unanswerable-answered 0/0" in out, out)


def test_unanswerable_answered_bucket():
    with tempfile.TemporaryDirectory() as tmp:
        ev, al = setup(tmp, [erow("q1", kind="unanswerable"), erow("q2")],
                       {"r": [arow("q1", "Use --x"), arow("q2", "alpha")]},
                       {"r": {"q1": [hit("b")], "q2": [hit("b")]}})
        code, out = run_main(tmp, ev, al, ["r"])
        check("r [tight] correct 0/1 wrong 0/0 unanswerable-answered 1/1" in out, out)


def test_extract_starts_counts_lowercase_ord_gt0():
    rows = [arow("q1", "x")]
    cache = {"q1": [hit("lower", ord_="3"), hit("Upper", ord_="3"),
                    hit("lower", ord_="0"), hit("9abc", ord_="10"),
                    {"text": ".dot", "ord": 4}, hit("lateness", ord_="1")]}
    got = gr.extract_starts(rows, cache)
    check(got == {"n": 6, "mid": 2}, got)
    dict_cache = {"q1": {"hits": [hit("low", ord_=2)], "gate_hits": [hit("low", ord_=2)]}}
    check(gr.extract_starts(rows, dict_cache) == {"n": 1, "mid": 1}, "dict entry")


def test_run_with_cap_per_doc_is_skipped_not_scored():
    with tempfile.TemporaryDirectory() as tmp:
        ev, al = setup(tmp, [erow("q1")], {"r": [arow("q1", "x")]},
                       {"r": {"q1": [hit("b")]}}, config={"cap_per_doc": 2, "read_k": 5})
        code, out = run_main(tmp, ev, al, ["r"])
        check(code == 0, code)
        check(out == "r: skipped (config cap_per_doc nonzero)\n", out)


def test_read_k_selects_first_k_hits():
    with tempfile.TemporaryDirectory() as tmp:
        hits = [hit("one"), hit("two"), hit("three --frob")]
        ev, al = setup(tmp, [erow("q1", tok=("zzz",))], {"r": [arow("q1", "--frob")]},
                       {"r": {"q1": hits}}, config={"read_k": 2})
        _, out = run_main(tmp, ev, al, ["r"])
        check("r [tight] correct 0/0 wrong 1/1" in out, out)
        check("extract-starts: lowercase fragment (ord>0) 0/2" in out, out)
        ev, al = setup(tmp, [erow("q1", tok=("zzz",))], {"r": [arow("q1", "--frob")]},
                       {"r": {"q1": hits}}, config={"read_k": 0})
        _, out = run_main(tmp, ev, al, ["r"])
        check("r [tight] correct 0/0 wrong 0/1" in out, out)


def test_missing_cache_is_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        ev, al = setup(tmp, [erow("q1")], {"a": [arow("q1", "x")], "b": [arow("q1", "x")]},
                       {"b": {"q1": [hit("t")]}})
        code, out = run_main(tmp, ev, al, ["a", "b", "c:b"])
        check(code == 0, code)
        check("a: skipped (missing a-retrieved.json)\n" in out, out)
        check("b [draft]" in out, out)
        check("c: skipped (missing c-answers.json)\n" in out, out)
        ev, al = setup(tmp, [erow("q9")], {"d": [arow("q1", "x")]}, {"d": {}})
        _, out = run_main(tmp, ev, al, ["d"])
        check(out == "d: skipped (qid not in eval)\n", out)


def test_report_line_format():
    with tempfile.TemporaryDirectory() as tmp:
        ev, al = setup(tmp, [erow("q1"), erow("q2", tok=("zzz",))],
                       {"r": [arow("q1", "alpha"), arow("q2", "--x")]},
                       {"rc": {"q1": [hit("lower", ord_="2")], "q2": [hit("Up")]}})
        _, out = run_main(tmp, ev, al, ["r:rc"])
        lines = out.splitlines()
        check(lines == [
            "r [draft] correct 0/1 wrong 1/1 unanswerable-answered 0/0",
            "r [draft] catch: wrong flagged 1/1, correct flagged 0/1",
            "r [tight] correct 0/1 wrong 1/1 unanswerable-answered 0/0",
            "r [tight] catch: wrong flagged 1/1, correct flagged 0/1",
            "rc extract-starts: lowercase fragment (ord>0) 1/2"], lines)


def test_json_output_lists_input_sha256():
    with tempfile.TemporaryDirectory() as tmp:
        ev, al = setup(tmp, [erow("q1")], {"r": [arow("q1", "alpha")]},
                       {"r": {"q1": [hit("t")]}})
        js = Path(tmp) / "out.json"
        run_main(tmp, ev, al, ["r", "zz"], extra=["--json", str(js)])
        data = json.loads(js.read_text())
        h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
        check(data["sha256"]["eval"] == h(ev), data)
        check(data["sha256"]["aliases"] == h(al), data)
        check(data["sha256"]["answers"] == {"r": h(Path(tmp) / "r-answers.json")}, data)
        check(data["sha256"]["caches"] == {"r": h(Path(tmp) / "r-retrieved.json")}, data)
        check(data["runs"]["r"]["tight"]["correct"] == {"n": 1, "flagged": 0}, data)
        check(data["runs"]["r"]["extract_starts"] == {"n": 1, "mid": 0}, data)
        check("zz" in data["skipped"], data)


def test_runs_are_independent_of_input_order():
    with tempfile.TemporaryDirectory() as tmp:
        ev, al = setup(tmp, [erow("q1")],
                       {"a": [arow("q1", "--x")], "b": [arow("q1", "alpha")]},
                       {"a": {"q1": [hit("t")]}, "b": {"q1": [hit("t")]}})
        _, o1 = run_main(tmp, ev, al, ["a", "b"])
        _, o2 = run_main(tmp, ev, al, ["b", "a"])
        check(sorted(o1.splitlines()) == sorted(o2.splitlines()), (o1, o2))
        check(len(set(o1.splitlines())) == len(o1.splitlines()), o1)


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
