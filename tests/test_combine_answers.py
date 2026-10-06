"""Tests for scripts/combine_answers.py.

Hermetic (blk_test_env_constraints): stdlib only, synthetic answers files in a
tempdir, never data/eval/results/.
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


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ca = _load("combine_answers", "combine_answers.py")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def row(qid, answer="x", abstained=False, gated=False, top=0.9, evidence=True):
    return {"qid": qid, "answer": answer, "abstained": abstained, "gated": gated,
            "top_score": top, "evidence_retrieved": evidence}


def write(tmp, name, rows, config=None):
    (Path(tmp) / f"{name}-answers.json").write_text(json.dumps(
        {"name": name, "k": 5, "n": len(rows), "config": config or {"gate": 0.5},
         "results": rows}))


def combine(tmp, base="b", alt="a", t=0.8, out="o"):
    o, e = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
        code = ca.main(["--base", base, "--alt", alt, "--min-score", str(t),
                        "--out", out], results_dir=tmp)
    return code, o.getvalue(), e.getvalue()


def result(tmp, out="o"):
    return json.loads((Path(tmp) / f"{out}-answers.json").read_text())


def test_takes_alt_row_when_base_refused_above_threshold():
    with tempfile.TemporaryDirectory() as t:
        write(t, "b", [row("q1", "no", abstained=True, top=0.9)])
        write(t, "a", [row("q1", "alpha", top=0.9)])
        code, out, _ = combine(t, t=0.8)
        res = result(t)["results"][0]
        check(code == 0, code)
        check(res["answer"] == "alpha" and res["abstained"] is False, res)
        check(res["combined_from"] == "alt", res)
        check(out == "o: replaced 1 of 1 rows (base b, alt a, min score 0.8)\n", out)


def test_keeps_base_row_below_threshold():
    with tempfile.TemporaryDirectory() as t:
        write(t, "b", [row("q1", "no", abstained=True, top=0.79)])
        write(t, "a", [row("q1", "alpha")])
        combine(t, t=0.8)
        res = result(t)["results"][0]
        check(res["answer"] == "no" and "combined_from" not in res, res)


def test_keeps_base_row_when_base_answered():
    with tempfile.TemporaryDirectory() as t:
        write(t, "b", [row("q1", "beta", abstained=False, top=0.95)])
        write(t, "a", [row("q1", "alpha")])
        combine(t, t=0.8)
        res = result(t)["results"][0]
        check(res["answer"] == "beta" and "combined_from" not in res, res)


def test_gated_rows_are_never_replaced():
    with tempfile.TemporaryDirectory() as t:
        write(t, "b", [row("q1", "no", abstained=True, gated=True, top=0.95)])
        write(t, "a", [row("q1", "alpha")])
        combine(t, t=0.0)
        res = result(t)["results"][0]
        check(res["answer"] == "no" and "combined_from" not in res, res)
        base = row("q2", "no", abstained=True)
        del base["gated"]
        nots = row("q3", "no", abstained=True)
        del nots["top_score"]
        write(t, "b2", [base, nots])
        write(t, "a2", [row("q2", "alpha"), row("q3", "alpha")])
        combine(t, base="b2", alt="a2", t=0.0, out="o2")
        check(all(r["answer"] == "no" for r in result(t, "o2")["results"]),
              "rows missing gated or top_score must not be replaced")


def test_output_is_scoreable_by_screen_report():
    with tempfile.TemporaryDirectory() as t:
        write(t, "b", [row("q1", "no", abstained=True), row("q2", "beta")])
        write(t, "a", [row("q1", "alpha"), row("q2", "alpha")])
        combine(t)
        sr = _load("screen_report", "screen_report.py")
        ev = {"q1": {"qid": "q1", "kind": "answerable", "answer_contains": ["alpha"]},
              "q2": {"qid": "q2", "kind": "answerable", "answer_contains": ["alpha"]}}
        run = sr.score_run(Path(t) / "o-answers.json", ev, {})
        check(run["q1"]["correct"] is True, run)
        check(run["q2"]["correct"] is False, run)


def test_config_records_combination():
    with tempfile.TemporaryDirectory() as t:
        write(t, "b", [row("q1", "no", abstained=True), row("q2", "beta")],
              config={"gate": 0.5, "read_k": 5})
        write(t, "a", [row("q1", "alpha"), row("q2", "alpha")])
        combine(t, t=0.8)
        out = result(t)
        check(out["config"]["gate"] == 0.5 and out["config"]["read_k"] == 5, out["config"])
        check(out["config"]["combined"] ==
              {"base": "b", "alt": "a", "min_score": 0.8, "replaced": 1}, out["config"])
        check(out["name"] == "o" and out["k"] == 5 and out["n"] == 2, out)
        side = json.loads((Path(t) / "o-combine.json").read_text())
        check(side["replaced_qids"] == ["q1"], side)


def test_qid_sets_must_match_exit_2():
    with tempfile.TemporaryDirectory() as t:
        write(t, "b", [row("q1"), row("q2")])
        write(t, "a", [row("q1"), row("q3")])
        code, _, err = combine(t)
        check(code == 2 and err, (code, err))
        check(not (Path(t) / "o-answers.json").exists(), "nothing may be written")
        code, _, err = combine(t, alt="missing")
        check(code == 2 and "missing-answers.json" in err, (code, err))


def test_existing_output_file_is_not_overwritten():
    with tempfile.TemporaryDirectory() as t:
        write(t, "b", [row("q1")])
        write(t, "a", [row("q1")])
        (Path(t) / "o-answers.json").write_text("keep")
        code, _, err = combine(t)
        check(code == 2 and err, (code, err))
        check((Path(t) / "o-answers.json").read_text() == "keep", "overwritten")


def test_writes_sha256_of_inputs():
    with tempfile.TemporaryDirectory() as t:
        write(t, "b", [row("q2", "no", abstained=True), row("q1", "no", abstained=True)])
        write(t, "a", [row("q1", "alpha"), row("q2", "alpha")])
        combine(t, t=0.8)
        side = json.loads((Path(t) / "o-combine.json").read_text())
        for key, name in (("base", "b"), ("alt", "a")):
            want = hashlib.sha256((Path(t) / f"{name}-answers.json").read_bytes()).hexdigest()
            check(side[key]["sha256"] == want, side)
        check(side["replaced_qids"] == ["q1", "q2"], side)
        check([r["qid"] for r in result(t)["results"]] == ["q2", "q1"], "base order")


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
