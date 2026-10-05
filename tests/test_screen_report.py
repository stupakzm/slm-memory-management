"""Tests for scripts/screen_report.py (paired screen scorer).

Hermetic (blk_test_env_constraints): stdlib only, synthetic tiny eval and
answers files in a tempdir, never data/eval/results/. sqlite_vec is stubbed
first, matching the repo's test convention.
"""

from __future__ import annotations

import contextlib
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
    "screen_report", ROOT / "scripts" / "screen_report.py")
sr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sr)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def row(qid, answer="", abstained=False, evidence=True):
    return {"qid": qid, "answer": answer, "abstained": abstained,
            "evidence_retrieved": evidence}


def erow(qid, kind="answerable", tok=("alpha",), **kw):
    return {"qid": qid, "kind": kind, "answer_contains": list(tok), **kw}


def setup(tmp, eval_rows, runs, aliases=None):
    tmp = Path(tmp)
    ev = tmp / "eval.jsonl"
    ev.write_text("\n".join(json.dumps(r) for r in eval_rows) + "\n")
    for name, rows in runs.items():
        (tmp / f"{name}-answers.json").write_text(json.dumps({"results": rows}))
    al = tmp / "aliases.json"
    al.write_text(json.dumps(aliases or {}))
    return ev, al


def run_main(tmp, ev, al, control, arms, extra=()):
    argv = ["--eval", str(ev), "--control", control, "--aliases", str(al)]
    for a in arms:
        argv += ["--arm", a]
    argv += list(extra)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = sr.main(argv, results_dir=tmp)
    return code, out.getvalue(), err.getvalue()


def scored(tmp, ev, al, name):
    return sr.score_run(Path(tmp) / f"{name}-answers.json",
                        sr.load_eval(ev), sr.gold.load_aliases(al))


def test_sign_test_p_values():
    check(sr.sign_test_p(4, 0) == 0.125, sr.sign_test_p(4, 0))
    check(sr.sign_test_p(4, 1) == 0.375, sr.sign_test_p(4, 1))
    check(sr.sign_test_p(0, 0) == 1.0, sr.sign_test_p(0, 0))
    check(f"{sr.sign_test_p(0, 0):.3g}" == "1", "p=1 must print as 1")
    check(f"{sr.sign_test_p(4, 1):.3g}" == "0.375", "format")


def test_abstained_never_correct():
    with tempfile.TemporaryDirectory() as t:
        ev, al = setup(t, [erow("q1")],
                       {"c": [row("q1", "alpha", abstained=True)]})
        check(scored(t, ev, al, "c")["q1"]["correct"] is False,
              "an abstained row containing the token must not be correct")


def test_aliases_widen_correct():
    with tempfile.TemporaryDirectory() as t:
        aliases = {"q1": {"alpha": [{"alias": "beta"}]}}
        ev, al = setup(t, [erow("q1"), erow("q1.y1", variant_of="q1")],
                       {"c": [row("q1", "use beta"), row("q1.y1", "use beta")]},
                       aliases)
        run = scored(t, ev, al, "c")
        check(run["q1"]["correct"], "alias should credit q1")
        check(run["q1.y1"]["correct"], "variant should fall back to base aliases")
        al2 = Path(t) / "none.json"
        al2.write_text("{}")
        check(not scored(t, ev, al2, "c")["q1"]["correct"],
              "without aliases the answer is wrong")


def _verdict_fixture(t, n_gain, n_lose):
    n = n_gain + n_lose + 1
    ev_rows = [erow(f"q{i}") for i in range(n)]
    ctl = [row(f"q{i}", "alpha" if i < n_lose else "x") for i in range(n)]
    arm = [row(f"q{i}", "x" if i < n_lose else
               ("alpha" if i < n_lose + n_gain else "x")) for i in range(n)]
    return ev_rows, ctl, arm


def test_verdict_thresholds():
    def verdict(gain, lose, abst_arm=True, **kw):
        with tempfile.TemporaryDirectory() as t:
            ev_rows, ctl, arm = _verdict_fixture(t, gain, lose)
            ev_rows.append(erow("u1", kind="unanswerable"))
            ctl.append(row("u1", "no", abstained=True))
            arm.append(row("u1", "no", abstained=abst_arm))
            ev, al = setup(t, ev_rows, {"c": ctl, "a": arm})
            return sr.compare(scored(t, ev, al, "c"), scored(t, ev, al, "a"), **kw)

    check(verdict(5, 1)["verdict"] == "PASS", "net 4, lost 1 should pass")
    check(verdict(3, 0)["verdict"] == "FAIL", "net below min-net fails alone")
    check(verdict(6, 2)["verdict"] == "FAIL", "lost above max-lost fails alone")
    check(verdict(4, 0, abst_arm=False)["verdict"] == "FAIL",
          "negative abstention net fails alone")
    check(verdict(4, 0, abst_arm=False, min_abstention_net=-1)["verdict"] == "PASS",
          "threshold is configurable")


def test_abstention_net():
    with tempfile.TemporaryDirectory() as t:
        ev_rows = [erow(f"u{i}", kind="unanswerable") for i in range(4)]
        ctl = [row("u0", abstained=True), row("u1", abstained=True),
               row("u2"), row("u3")]
        arm = [row("u0", abstained=False), row("u1", abstained=True),
               row("u2", abstained=True), row("u3", abstained=True)]
        ev, al = setup(t, ev_rows, {"c": ctl, "a": arm})
        r = sr.compare(scored(t, ev, al, "c"), scored(t, ev, al, "a"))
        check(r["abstention_net"] == 1, r["abstention_net"])  # +2 gained, -1 lost
        check(r["abstained"] == 3 and r["unanswerable"] == 4, r)


def test_mismatched_qids_exit_2():
    with tempfile.TemporaryDirectory() as t:
        ev, al = setup(t, [erow("q1"), erow("q2")],
                       {"c": [row("q1"), row("q2")], "a": [row("q1")],
                        "x": [row("q1"), row("q3")]})
        code, out, err = run_main(t, ev, al, "c", ["a"])
        check(code == 2 and err and not out, (code, out, err))
        code, out, err = run_main(t, ev, al, "c", ["x"])
        check(code == 2 and err, (code, err))
        # a qid missing from the eval file also exits 2
        ev1, al1 = setup(t, [erow("q1")], {})
        code, out, err = run_main(t, ev1, al1, "c", ["c"])
        check(code == 2, code)


def test_line_format():
    with tempfile.TemporaryDirectory() as t:
        ev_rows = [erow("q1"), erow("q2"), erow("q3"), erow("u1", kind="unanswerable")]
        ctl = [row("q1", "alpha", evidence=False), row("q2", "x"),
               row("q3", "alpha"), row("u1", abstained=True)]
        arm = [row("q1", "x"), row("q2", "alpha"), row("q3", "x"),
               row("u1", abstained=True)]
        ev, al = setup(t, ev_rows, {"c": ctl, "a": arm, "b": ctl})
        out_json = Path(t) / "out.json"
        code, out, err = run_main(t, ev, al, "c", ["a", "b"], ["--json", str(out_json)])
        want = ("control c: correct 2/3 evidence 2/3 abstained 1/1\n"
                "a vs c: correct 1/3 lost 2 gained 1 net -1 p 1 evidence 3/3 "
                "abstained 1/1 abstention_net +0 verdict FAIL\n"
                "  lost: q1 q3\n"
                "  gained: q2\n"
                "b vs c: correct 2/3 lost 0 gained 0 net +0 p 1 evidence 2/3 "
                "abstained 1/1 abstention_net +0 verdict FAIL\n")
        check(code == 0 and out == want, f"\n{out}\n!=\n{want}")
        data = json.loads(out_json.read_text())
        check(data["arms"]["a"]["lost_qids"] == ["q1", "q3"], data)
        check(len(data["sha256"]["eval"]) == 64 and set(data["sha256"]["answers"]) == {"c", "a", "b"},
              data["sha256"])
        check(data["thresholds"]["min_net"] == 4, data["thresholds"])


def test_repeatable_eval_merges_files():
    with tempfile.TemporaryDirectory() as t:
        ev1, al = setup(t, [erow("q1"), erow("q2")],
                        {"c": [row("q1", "alpha"), row("q2", "x")],
                         "a": [row("q1", "x"), row("q2", "alpha")]})
        ev2 = Path(t) / "eval2.jsonl"
        ev2.write_text(json.dumps(erow("q3")) + "\n")
        for n in ("c", "a"):
            p = Path(t) / f"{n}-answers.json"
            rows = json.loads(p.read_text())["results"] + [row("q3", "alpha")]
            p.write_text(json.dumps({"results": rows}))
        argv = ["--eval", str(ev1), "--eval", str(ev2), "--control", "c",
                "--arm", "a", "--aliases", str(al)]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = sr.main(argv, results_dir=t)
        check(code == 0 and out.getvalue().startswith("control c: correct 2/3 "),
              (code, out.getvalue(), err.getvalue()))
        # the same qid in two files exits 2
        argv = ["--eval", str(ev1), "--eval", str(ev1), "--control", "c",
                "--arm", "a", "--aliases", str(al)]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = sr.main(argv, results_dir=t)
        check(code == 2 and err.getvalue() and not out.getvalue(), (code, out.getvalue()))


def test_group_lines_partition_rows():
    with tempfile.TemporaryDirectory() as t:
        ev_rows = [erow("q1", domain="x"), erow("q2", domain="x"), erow("q3"),
                   erow("u1", kind="unanswerable", domain="x"),
                   erow("u2", kind="unanswerable")]
        ctl = [row("q1", "alpha"), row("q2", "x"), row("q3", "alpha"),
               row("u1", abstained=True), row("u2")]
        arm = [row("q1", "x"), row("q2", "alpha"), row("q3", "alpha"),
               row("u1"), row("u2", abstained=True)]
        ev, al = setup(t, ev_rows, {"c": ctl, "a": arm})
        out_json = Path(t) / "out.json"
        code, out, err = run_main(t, ev, al, "c", ["a"],
                                  ["--group", "domain", "--group-default", "dflt",
                                   "--json", str(out_json)])
        want = ("  [dflt] correct 1/1 lost 0 gained 0 net +0 p 1 evidence 1/1 "
                "abstained 1/1 abstention_net +1\n"
                "  [x] correct 1/2 lost 1 gained 1 net +0 p 1 evidence 2/2 "
                "abstained 0/1 abstention_net -1\n")
        check(code == 0 and out.endswith(want), f"\n{out}")
        r = json.loads(out_json.read_text())["arms"]["a"]
        gs = r["groups"].values()
        for k in ("correct", "evidence", "abstained", "answerable", "unanswerable",
                  "lost", "gained", "net", "abstention_net"):
            check(sum(g[k] for g in gs) == r[k], (k, r))


def test_no_group_output_unchanged():
    with tempfile.TemporaryDirectory() as t:
        ev, al = setup(t, [erow("q1", domain="x")],
                       {"c": [row("q1", "alpha")], "a": [row("q1", "x")]})
        out_json = Path(t) / "out.json"
        code, out, err = run_main(t, ev, al, "c", ["a"], ["--json", str(out_json)])
        want = ("control c: correct 1/1 evidence 1/1 abstained 0/0\n"
                "a vs c: correct 0/1 lost 1 gained 0 net -1 p 1 evidence 1/1 "
                "abstained 0/0 abstention_net +0 verdict FAIL\n"
                "  lost: q1\n")
        check(code == 0 and out == want, f"\n{out}")
        check("groups" not in json.loads(out_json.read_text())["arms"]["a"], "no groups")


def _ma_fixture(t, arm_gate=None):
    ev_rows = [erow("q1"), erow("q2"),
               erow("u1", kind="unanswerable"), erow("u2", kind="unanswerable"),
               erow("u3", kind="unanswerable")]
    ctl = [row("q1", "alpha", evidence=True), row("q2", "alpha"),
           row("u1", abstained=True), row("u2", abstained=True), row("u3")]
    arm = [dict(row("q1", "alpha"), top_score=0.9), dict(row("q2", "alpha"), top_score=0.5),
           dict(row("u1"), top_score=0.7), dict(row("u2"), top_score=0.6),
           dict(row("u3"), top_score=0.95)]
    ev, al = setup(t, ev_rows, {"c": ctl, "a": arm})
    if arm_gate is not None:
        (Path(t) / "a-answers.json").write_text(
            json.dumps({"config": {"gate": arm_gate}, "results": arm}))
    return ev, al


def test_match_abstention_picks_lowest_reaching_gate():
    with tempfile.TemporaryDirectory() as t:
        ev, al = _ma_fixture(t)
        # control abstains on 2 unanswerable; arm needs top_score < t for u2 and u1
        code, out, err = run_main(t, ev, al, "c", ["a"], ["--match-abstention"])
        check(code == 0, err)
        check("  matched-abstention gate 0.7000:" in out, f"\n{out}")
        out_json = Path(t) / "o.json"
        run_main(t, ev, al, "c", ["a"], ["--match-abstention", "--json", str(out_json)])
        m = json.loads(out_json.read_text())["arms"]["a"]["matched"]
        check(abs(m["threshold"] - 0.7) < 1e-6 and m["abstained"] == 2, m)
        # the arm's own gate floors the candidates
        ev, al = _ma_fixture(t, arm_gate=0.8)
        code, out, err = run_main(t, ev, al, "c", ["a"], ["--match-abstention"])
        check("  matched-abstention gate 0.8000:" in out, f"\n{out}")


def test_match_abstention_replays_refusals():
    with tempfile.TemporaryDirectory() as t:
        ev, al = _ma_fixture(t)
        code, out, err = run_main(t, ev, al, "c", ["a"], ["--match-abstention"])
        # q2 (0.5 < 0.7) becomes a refusal and loses; q1 (0.9) stays correct
        want = ("  matched-abstention gate 0.7000: correct 1/2 lost 1 gained 0 "
                "net -1 p 1 evidence 2/2 abstained 2/3 abstention_net +0 "
                "verdict FAIL\n")
        check(want in out, f"\n{out}")
        run = sr.score_run(Path(t) / "a-answers.json", sr.load_eval(ev),
                           sr.gold.load_aliases(al))
        rp = sr.replay_gate(run, 0.7)
        check(rp["q2"]["abstained"] and not rp["q2"]["correct"] and rp["q2"]["evidence"],
              rp["q2"])
        check(not rp["q1"]["abstained"] and rp["q1"]["correct"], rp["q1"])
        run["q1"]["top_score"] = float("-inf")
        check(not sr.replay_gate(run, 5.0)["q1"]["abstained"], "no score, never gated")


def test_match_abstention_unreachable():
    with tempfile.TemporaryDirectory() as t:
        ev_rows = [erow("u1", kind="unanswerable"), erow("u2", kind="unanswerable")]
        ev, al = setup(t, ev_rows, {
            "c": [row("u1", abstained=True), row("u2", abstained=True)],
            "a": [row("u1"), row("u2")]})  # no top_score at all
        code, out, err = run_main(t, ev, al, "c", ["a"], ["--match-abstention"])
        check(code == 0 and out.endswith(
            "  matched-abstention: unreachable (max abstained 0/2)\n"), f"\n{out}")


def test_match_abstention_line_format():
    with tempfile.TemporaryDirectory() as t:
        ev, al = _ma_fixture(t)
        out_json = Path(t) / "o.json"
        code, out, err = run_main(t, ev, al, "c", ["a"],
                                  ["--match-abstention", "--group", "domain",
                                   "--json", str(out_json)])
        lines = out.splitlines()
        i = next(k for k, l in enumerate(lines) if l.startswith("  matched-abstention"))
        check(lines[i].startswith("  matched-abstention gate 0.7000: correct 1/2 "), lines[i])
        check(lines[i].endswith("verdict FAIL"), lines[i])
        check(lines[i + 1].startswith("    [-] correct "), lines[i + 1:])
        check(lines[i - 1].startswith("  [-] "), lines[i - 1])
        r = json.loads(out_json.read_text())["arms"]["a"]
        check("threshold" in r["matched"] and "groups" in r["matched"], r["matched"])


def test_no_flag_output_unchanged_with_top_score():
    with tempfile.TemporaryDirectory() as t:
        ev, al = _ma_fixture(t)
        out_json = Path(t) / "o.json"
        code, out, err = run_main(t, ev, al, "c", ["a"], ["--json", str(out_json)])
        check(code == 0 and "matched" not in out, f"\n{out}")
        check("matched" not in json.loads(out_json.read_text())["arms"]["a"], "json")
        want = ("control c: correct 2/2 evidence 2/2 abstained 2/3\n"
                "a vs c: correct 2/2 lost 0 gained 0 net +0 p 1 evidence 2/2 "
                "abstained 0/3 abstention_net -2 verdict FAIL\n")
        check(out == want, f"\n{out}")


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
