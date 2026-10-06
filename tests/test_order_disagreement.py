"""Tests for scripts/order_disagreement.py (answer change between extract orders).

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
    "order_disagreement", ROOT / "scripts" / "order_disagreement.py")
od = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(od)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def erow(qid, kind="answerable", tok=("alpha",)):
    return {"qid": qid, "kind": kind, "answer_contains": list(tok)}


def arow(qid, answer, abstained=False):
    return {"qid": qid, "answer": answer, "abstained": abstained}


def setup(tmp, eval_rows, runs):
    tmp = Path(tmp)
    ev = tmp / "eval.jsonl"
    ev.write_text("\n".join(json.dumps(r) for r in eval_rows) + "\n")
    for name, rows in runs.items():
        (tmp / f"{name}-answers.json").write_text(
            json.dumps({"config": {}, "results": rows}))
    al = tmp / "aliases.json"
    al.write_text("{}")
    return ev, al


def run_main(tmp, ev, al, extra=(), a="a", b="b"):
    argv = ["--eval", str(ev), "--aliases", str(al), "--a", a, "--b", b, *extra]
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = od.main(argv, results_dir=tmp)
    return code, out.getvalue(), err.getvalue()


def one(a_answer, b_answer, b_abstained=False, kind="answerable"):
    """Run a single-row pair; return (code, stdout)."""
    with tempfile.TemporaryDirectory() as tmp:
        ev, al = setup(tmp, [erow("q1", kind=kind)],
                       {"a": [arow("q1", a_answer)],
                        "b": [arow("q1", b_answer, b_abstained)]})
        code, out, _ = run_main(tmp, ev, al)
        return code, out


def test_same_identifiers_agree_despite_prose():
    _, out = one("alpha: use -n[1] here", "Different words, but -n and alpha.")
    check("correct 0/1" in out, out)


def test_different_identifiers_disagree():
    _, out = one("alpha: use -n", "alpha: use -m")
    check("correct 1/1" in out, out)


def test_second_run_abstaining_is_disagreement():
    _, out = one("alpha: use -n", "", b_abstained=True)
    check("correct 1/1" in out, out)


def test_first_run_abstained_rows_are_excluded():
    with tempfile.TemporaryDirectory() as tmp:
        ev, al = setup(tmp, [erow("q1"), erow("q2")],
                       {"a": [arow("q1", "", True), arow("q2", "alpha -n")],
                        "b": [arow("q1", "alpha -n"), arow("q2", "alpha -n")]})
        _, out, _ = run_main(tmp, ev, al)
        check(out.splitlines()[0] ==
              "a vs b: answered 1 correct 0/1 wrong 0/0 unanswerable-answered 0/0", out)


def test_buckets_follow_first_run_correctness():
    with tempfile.TemporaryDirectory() as tmp:
        ev, al = setup(tmp, [erow("q1"), erow("q2"), erow("q3", kind="unanswerable")],
                       {"a": [arow("q1", "alpha -n"), arow("q2", "zzz -n"),
                              arow("q3", "alpha -n")],
                        "b": [arow("q1", "alpha -m"), arow("q2", "zzz -n"),
                              arow("q3", "alpha -m")]})
        _, out, _ = run_main(tmp, ev, al)
        check(out.splitlines()[0] ==
              "a vs b: answered 3 correct 1/1 wrong 0/1 unanswerable-answered 1/1", out)


def test_rows_with_no_identifiers_compare_normalised_text():
    _, out = one("alpha  is   Fine.[1]", "ALPHA is fine")
    check("correct 0/1" in out, out)
    _, out = one("alpha is fine", "alpha is not fine")
    check("correct 1/1" in out, out)


def test_qid_sets_must_match_exit_2():
    with tempfile.TemporaryDirectory() as tmp:
        ev, al = setup(tmp, [erow("q1"), erow("q2")],
                       {"a": [arow("q1", "alpha")],
                        "b": [arow("q1", "alpha"), arow("q2", "alpha")]})
        code, out, err = run_main(tmp, ev, al)
        check(code == 2 and out == "" and err, (code, out, err))
        code, _, err = run_main(tmp, ev, al, b="missing")
        check(code == 2 and "missing-answers.json" in err, (code, err))


def test_report_line_format():
    with tempfile.TemporaryDirectory() as tmp:
        ev, al = setup(tmp, [erow("q1"), erow("q2", tok=("zzz",)),
                             erow("q3", kind="unanswerable")],
                       {"a": [arow("q1", "alpha -n"), arow("q2", "-x"),
                              arow("q3", "-y")],
                        "b": [arow("q1", "alpha -m"), arow("q2", "", True),
                              arow("q3", "-y")]})
        code, out, _ = run_main(tmp, ev, al)
        check(code == 0, code)
        check(out.splitlines() == [
            "a vs b: answered 3 correct 1/1 wrong 1/1 unanswerable-answered 0/1",
            "a vs b: refusing every disagreement would lose 1 correct answers "
            "and catch 1 wrong and 0 unanswerable-answered rows"], out)


def test_json_output_lists_input_sha256():
    with tempfile.TemporaryDirectory() as tmp:
        ev, al = setup(tmp, [erow("q1")],
                       {"a": [arow("q1", "alpha -n")], "b": [arow("q1", "alpha -m")]})
        js = Path(tmp) / "out.json"
        run_main(tmp, ev, al, extra=["--json", str(js)])
        data = json.loads(js.read_text())
        h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
        check(data["counts"]["correct"] == {"n": 1, "disagree": 1}, data)
        check(data["sha256"] == {"eval": h(ev), "aliases": h(al),
                                 "a": h(Path(tmp) / "a-answers.json"),
                                 "b": h(Path(tmp) / "b-answers.json")}, data)


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
