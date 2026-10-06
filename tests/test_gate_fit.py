"""Tests for scripts/gate_fit.py (calibrated-gate fit and held-out replay).

Hermetic (blk_test_env_constraints): stdlib only, synthetic tiny eval, answers,
cache and vocab files in a tempdir, never data/. sqlite_vec is stubbed first,
matching the repo's test convention.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import re
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
    "gate_fit", ROOT / "scripts" / "gate_fit.py")
gf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gf)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

N = 60


def toy_eval():
    rows = []
    for i in range(N):
        if i % 3 == 0:
            rows.append({"qid": f"t{i:02d}", "kind": "unanswerable",
                         "question": "tell me about xyzzy plugh", "answer_contains": []})
        else:
            rows.append({"qid": f"t{i:02d}", "kind": "answerable",
                         "question": "what is alpha", "answer_contains": ["alpha"]})
    return rows


def toy_top1(i):
    return 0.1 + 0.003 * i if i % 3 == 0 else 0.8 + 0.003 * i


def toy_cache():
    return {f"t{i:02d}": [{"doc_id": f"d{i}", "score": str(toy_top1(i)),
                           "distance": "0.5"}] for i in range(N)}


def toy_answers(plain_gate_works):
    """Answerable rows are correct with evidence. Unanswerable rows are refused by
    the plain gate iff `plain_gate_works`, else answered (no evidence)."""
    rows = []
    for i in range(N):
        q = f"t{i:02d}"
        if i % 3 == 0:
            rows.append({"qid": q, "answer": "" if plain_gate_works else "alpha",
                         "abstained": plain_gate_works, "evidence_retrieved": False})
        else:
            rows.append({"qid": q, "answer": "alpha is it", "abstained": False,
                         "evidence_retrieved": True})
    return rows


def write_world(tmp, eval_rows=None, runs=None, caches=None, vocab=None):
    tmp = Path(tmp)
    ev = tmp / "eval.jsonl"
    ev.write_text("\n".join(json.dumps(r) for r in (eval_rows or toy_eval())) + "\n")
    al = tmp / "aliases.json"
    al.write_text("{}")
    runs = runs if runs is not None else {"ctl": toy_answers(True), "arm": toy_answers(False)}
    for name, rows in runs.items():
        (tmp / f"{name}-answers.json").write_text(json.dumps({"results": rows}))
    caches = caches if caches is not None else {"ctl": toy_cache()}
    for name, c in caches.items():
        (tmp / f"{name}-retrieved.json").write_text(json.dumps(c))
    vp = None
    if vocab is not None:
        vp = tmp / "vocab.json"
        vp.write_text(json.dumps(vocab))
    return ev, al, vp


def run_main(tmp, ev, al, control="ctl", arms=("arm:ctl",), extra=()):
    argv = ["--eval", str(ev), "--aliases", str(al), "--control", control]
    for a in arms:
        argv += ["--arm", a]
    argv += list(extra)
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = gf.main(argv, results_dir=tmp)
    return code, out.getvalue(), err.getvalue()


def scored(tmp, ev, al, name):
    return gf.sr.score_run(Path(tmp) / f"{name}-answers.json",
                           gf.sr.load_eval(ev), gf.gold.load_aliases(al))


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------

def test_split_is_deterministic_and_pinned():
    # buckets computed once with int(hashlib.sha256(key).hexdigest(), 16) % 10
    pinned = {"q1": 3, "q2": 1, "q3": 9, "q4": 6, "q5": 1, "u1": 7, "u2": 8,
              "fam-a": 3, "fam-b": 1, "fam-c": 5}
    for key, b in pinned.items():
        check(gf.bucket_of(key) == b, f"{key}: {gf.bucket_of(key)} != {b}")
    check(gf.split_of({"qid": "q4"}) == "fit", "bucket 6 is fit")
    check(gf.split_of({"qid": "u1"}) == "held", "bucket 7 is held")
    check(gf.split_of({"qid": "q3"}) == "held", "bucket 9 is held")
    check(gf.split_of({"qid": "q1"}) == "fit", "bucket 3 is fit")
    check([gf.split_of({"qid": "q3"}) for _ in range(3)] == ["held"] * 3, "deterministic")


def test_split_keeps_variant_families_together():
    own = [f"v{i}" for i in range(30)]
    # the member qids alone would land on both sides; the family key must win
    check({gf.split_of({"qid": q}) for q in own} == {"fit", "held"}, "fixture must straddle")
    for field in ("variant_of", "paraphrase_of"):
        got = {gf.split_of({"qid": q, field: "fam-c"}) for q in own}
        check(got == {"fit"}, f"{field}: {got}")  # fam-c is bucket 5
    check({gf.split_of({"qid": q, "variant_of": "q3"}) for q in own} == {"held"},
          "family of q3 (bucket 9) is held")
    # variant_of takes precedence over paraphrase_of
    check(gf.split_of({"qid": "x", "variant_of": "fam-c", "paraphrase_of": "q3"}) == "fit",
          "variant_of first")


# --------------------------------------------------------------------------
# features and label
# --------------------------------------------------------------------------

def test_features_from_cache_entry():
    gate_hits = [{"doc_id": "a", "score": 0.5, "rerank_score": 0.9, "distance": 0.2},
                 {"doc_id": "a", "score": 0.6, "rerank_score": 0.4},
                 {"doc_id": "b", "score": 0.3},
                 {"doc_id": "a", "score": 0.2, "rerank_score": 0.2},
                 {"doc_id": "c", "score": 0.1, "rerank_score": 0.1}]
    other = [{"doc_id": "z", "score": 5.0, "distance": 9.0}]
    want = [0.9, 0.9 - 0.4, 3.0, 0.2, (0.9 + 0.4 + 0.3 + 0.2 + 0.1) / 5, 0.0]
    got = gf.features({"hits": other, "gate_hits": gate_hits}, "hello", None)
    check(len(got) == len(gf.FEATURES) == 6, got)
    check(all(close(g, w) for g, w in zip(got, want)), f"{got} != {want}")
    check(gf.features(gate_hits, "hello", None) == got, "plain list is both lists")
    check(gf.features({"hits": gate_hits, "gate_hits": other}, "x", None)[0] == 5.0,
          "features read gate_hits, not hits")
    check(gf.FEATURES == ("top1", "gap", "same_doc", "dist1", "mean5", "unk"), gf.FEATURES)


def test_features_tolerate_string_numbers_and_short_lists():
    one = [{"doc_id": "a", "score": "0.5877", "distance": "0.31"}]
    got = gf.features(one, "q", None)
    check(got == [0.5877, 0.5877, 1.0, 0.31, 0.5877, 0.0], got)
    got = gf.features([{"doc_id": "a", "score": "0.5", "rerank_score": "0.8"},
                       {"doc_id": "b", "score": "0.9", "rerank_score": "0.3"}], "q", None)
    check(close(got[0], 0.8) and close(got[1], 0.5) and got[2] == 1.0 and got[3] == 0.0,
          got)
    seven = [{"doc_id": "a" if i < 5 else "a", "score": "0.5"} for i in range(7)]
    seven[4]["doc_id"] = "b"
    check(gf.features(seven, "q", None)[2] == 4.0, "only the first five hits count")
    check(gf.features([], "q", None) is None and gf.features({"hits": [], "gate_hits": []},
                                                              "q", None) is None,
          "no hits -> no features")


def test_unknown_word_share_uses_vocab():
    vocab = {"quick": 1, "brown": 3}
    h = [{"doc_id": "a", "score": 0.5}]
    # 4+ letter words: quick, xyzzy, plugh (the, foo are too short) -> 2 of 3 unknown
    got = gf.features(h, "The QUICK xyzzy foo plugh", vocab)[5]
    check(close(got, 2 / 3), got)
    check(gf.features(h, "the quick xyzzy", None)[5] == 0.0, "no vocab -> 0")
    check(gf.features(h, "a an the", vocab)[5] == 0.0, "no long words -> 0")
    check(gf.features(h, "quick brown", vocab)[5] == 0.0, "all known -> 0")


def test_label_is_answerable_and_evidence():
    r = {"answerable": True, "evidence": True}
    check(gf.label_of(r) == 1, "answerable with evidence")
    check(gf.label_of({**r, "evidence": False}) == 0, "answerable without evidence")
    check(gf.label_of({"answerable": False, "evidence": True}) == 0,
          "unanswerable is 0 even with evidence")
    check(gf.label_of({"answerable": False, "evidence": False}) == 0, "unanswerable")


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def _toy_xy():
    X, y = [], []
    for i in range(40):
        pos = i % 2
        top1 = (0.9 if pos else 0.2) + 0.002 * i
        X.append([top1, top1, 1.0, 0.0, top1, 0.0])
        y.append(pos)
    return X, y


def test_fit_is_deterministic():
    X, y = _toy_xy()
    m, s = gf.standardiser(X)
    Z = [gf.standardise(x, m, s) for x in X]
    a, b = gf.fit_logistic(Z, y), gf.fit_logistic(Z, y)
    check(a == b, "same inputs, same weights")
    check(any(abs(w) > 0.05 for w in a[0]), f"model must learn something: {a}")
    check(gf.fit_logistic([], []) == ([0.0] * 6, 0.0), "empty fit set -> zero model")


def test_fit_separates_a_separable_toy_set():
    with tempfile.TemporaryDirectory() as tmp:
        ev, al, _ = write_world(tmp)
        eval_rows = gf.sr.load_eval(ev)
        ctl, arm = scored(tmp, ev, al, "ctl"), scored(tmp, ev, al, "arm")
        cache = json.loads((Path(tmp) / "ctl-retrieved.json").read_text())
        feats = {q: gf.features(cache[q], "", None) for q in ctl}
        target = sum(r["abstained"] for q, r in ctl.items()
                     if gf.split_of(eval_rows[q]) == "fit" and not r["answerable"])
        check(target > 0, "fixture: control refuses some fit rows")
        held = [q for q in ctl if gf.split_of(eval_rows[q]) == "held"]
        una_held = {q for q in held if not ctl[q]["answerable"]}
        check(una_held and len(una_held) < len(held), "fixture: both classes held out")
        res = gf.fit_run(arm, feats, eval_rows, target)
        check(res["reachable"], "target reachable on a separable set")
        check(res["model"]["weights"][0] > 0, f"top1 weight positive: {res['model']}")
        refused = {q for q in held if res["calibrated"][q]["abstained"]}
        check(refused == una_held, f"refused {sorted(refused)} != {sorted(una_held)}")
        for q in held:
            if q not in una_held:
                check(res["calibrated"][q]["correct"], f"{q} must stay correct")


def test_standardisation_uses_fit_rows_only():
    means, stds = gf.standardiser([[1.0, 5, 0, 0, 0, 0], [3.0, 5, 0, 0, 0, 0]])
    check(means[0] == 2.0 and stds[0] == 1.0, (means, stds))  # population std
    check(means[1] == 5.0 and stds[1] == 1.0, "std 0 -> 1")
    check(gf.standardise([4.0, 5, 0, 0, 0, 0], means, stds)[:2] == [2.0, 0.0], "z-score")
    # end to end: moving the held rows' cache scores must not move the fitted model
    with tempfile.TemporaryDirectory() as tmp:
        ev, al, _ = write_world(tmp)
        eval_rows = gf.sr.load_eval(ev)
        ctl = scored(tmp, ev, al, "ctl")
        cache = toy_cache()

        def model_for(c):
            feats = {q: gf.features(c[q], "", None) for q in ctl}
            return gf.fit_run(ctl, feats, eval_rows, 3)["model"]

        base = model_for(cache)
        moved = {q: ([{**h, "score": "0.01"} for h in v]
                     if gf.split_of(eval_rows[q]) == "held" else v)
                 for q, v in cache.items()}
        check(model_for(moved) == base, "held rows leaked into the fit")
        moved_fit = {q: ([{**h, "score": "0.99"} for h in v]
                         if gf.split_of(eval_rows[q]) == "fit" else v)
                     for q, v in cache.items()}
        check(model_for(moved_fit) != base, "fit rows must move the model")


def test_threshold_matches_control_abstention_on_fit_rows():
    rows = [(0.1, False), (0.2, False), (0.3, False), (0.4, False)]
    t, ok = gf.choose_threshold(rows, 2)
    check(ok and close(t, 0.2 + 1e-12, 1e-15), (t, ok))
    check(sum(1 for p, a in rows if p < t) == 2, "a row AT the p is refused")
    t, ok = gf.choose_threshold([(0.1, False), (0.2, True), (0.3, False), (0.4, False)], 2)
    check(ok and close(t, 0.1 + 1e-12, 1e-15), "stored abstentions count toward the target")
    t, ok = gf.choose_threshold(rows, 0)
    check(ok and close(t, 0.1 + 1e-12, 1e-15), "lowest candidate")
    t, ok = gf.choose_threshold(rows, 5)
    check(not ok and close(t, 0.4 + 1e-12, 1e-15), "unreachable -> largest candidate")
    t, ok = gf.choose_threshold([(None, False), (0.5, False)], 2)
    check(ok and close(t, 0.5 + 1e-12, 1e-15), "no-hit rows are always refused")
    # end to end: the printed fit line reaches the control's abstention
    with tempfile.TemporaryDirectory() as tmp:
        ev, al, _ = write_world(tmp)
        code, out, err = run_main(tmp, ev, al)
        check(code == 0, err)
        m = re.search(r"^target control refuses (\d+)/(\d+) unanswerable fit rows at its own"
                      r" gate$", out, re.M)
        check(m and m.group(1) == m.group(2), out)
        for name in ("ctl", "arm"):
            f = re.search(rf"^{name} fit threshold (\d+\.\d{{6}}) refused (\d+)/(\d+) "
                          rf"correct (\d+)/(\d+)$", out, re.M)
            check(f and int(f.group(2)) >= int(m.group(1)), f"{name}: {out}")


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------

def test_replay_turns_low_probability_rows_into_refusals():
    run = {"a": {"correct": True, "evidence": True, "abstained": False, "answerable": True,
                 "top_score": 0.9},
           "b": {"correct": True, "evidence": True, "abstained": False, "answerable": True,
                 "top_score": 0.9},
           "c": {"correct": False, "evidence": False, "abstained": False, "answerable": False,
                 "top_score": 0.9}}
    got = gf.replay(run, {"a": 0.2, "b": 0.8, "c": 0.5}, 0.5)
    check(got["a"]["abstained"] and not got["a"]["correct"], "p < t is refused")
    check(got["a"]["evidence"] is True, "evidence unchanged")
    check(got["b"] == run["b"], "p >= t keeps stored values")
    check(got["c"] == run["c"], "p == t is not refused")
    check(run["a"]["abstained"] is False, "input not mutated")


def test_no_hits_row_is_always_refused():
    run = {"a": {"correct": True, "evidence": True, "abstained": False, "answerable": True,
                 "top_score": 0.9}}
    check(gf.replay(run, {"a": None}, 0.0)["a"]["abstained"], "no p -> refused at any t")
    with tempfile.TemporaryDirectory() as tmp:
        cache = toy_cache()
        cache["t01"] = []
        cache.pop("t02")  # a qid absent from the cache has no hits either
        ev, al, _ = write_world(tmp, caches={"ctl": cache})
        eval_rows = gf.sr.load_eval(ev)
        ctl = scored(tmp, ev, al, "ctl")
        feats = {q: gf.features(cache.get(q), "", None) for q in ctl}
        check(feats["t01"] is None and feats["t02"] is None, "no features")
        res = gf.fit_run(ctl, feats, eval_rows, 0)
        for q in ("t01", "t02"):
            check(res["probs"][q] is None, "no probability")
            check(res["calibrated"][q]["abstained"] and not res["calibrated"][q]["correct"], q)
        code, out, err = run_main(tmp, ev, al, arms=())
        check(code == 0, err)


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

FIELDS = (r"correct \d+/\d+ lost \d+ gained \d+ net [+-]\d+ p \S+ evidence \d+/\d+ "
          r"abstained \d+/\d+ abstention_net [+-]\d+ verdict (?:PASS|FAIL)")


def test_heldout_lines_format():
    with tempfile.TemporaryDirectory() as tmp:
        ev, al, _ = write_world(tmp)
        code, out, err = run_main(tmp, ev, al, arms=("arm:ctl",))
        check(code == 0, err)
        lines = out.splitlines()
        eval_rows = gf.sr.load_eval(ev)
        nf = sum(1 for r in eval_rows.values() if gf.split_of(r) == "fit")
        check(re.fullmatch(rf"split fit {nf} rows \(\d+ answerable, \d+ unanswerable\) held "
                           rf"{N - nf} rows \(\d+ answerable, \d+ unanswerable\)", lines[0]),
              lines[0])
        check(re.fullmatch(r"target control refuses \d+/\d+ unanswerable fit rows at its own"
                           r" gate", lines[1]), lines[1])
        num = r"[+-]\d+\.\d{4}"
        for name in ("ctl", "arm"):
            check(any(re.fullmatch(rf"{name} weights bias {num} top1 {num} gap {num} "
                                   rf"same_doc {num} dist1 {num} mean5 {num} unk {num}", l)
                      for l in lines), f"{name} weights line\n{out}")
            check(any(re.fullmatch(rf"{name} fit threshold \d+\.\d{{6}} refused \d+/\d+ "
                                   rf"correct \d+/\d+", l) for l in lines), f"{name} fit line")
        check(any(re.fullmatch(rf"ctl held calibrated vs ctl plain: {FIELDS}", l)
                  for l in lines), f"control line\n{out}")
        check(any(re.fullmatch(rf"arm held plain vs ctl: {FIELDS}", l) for l in lines),
              f"plain line\n{out}")
        check(any(re.fullmatch(rf"arm held calibrated vs ctl: {FIELDS}", l) for l in lines),
              f"calibrated line\n{out}")
        check(not any(l.startswith("ctl held plain") for l in lines), "control has no plain line")
        # the arm's plain gate answers every unanswerable row, the calibrated gate refuses them
        pl = next(l for l in lines if l.startswith("arm held plain"))
        cl = next(l for l in lines if l.startswith("arm held calibrated"))
        uh = int(re.search(r"split .* held \d+ rows \(\d+ answerable, (\d+) unanswerable",
                           lines[0]).group(1))
        check(f"abstained 0/{uh} abstention_net -{uh}" in pl, pl)
        check(f"abstained {uh}/{uh} abstention_net +0" in cl, cl)


def test_json_output_lists_weights_and_input_sha256():
    with tempfile.TemporaryDirectory() as tmp:
        ev, al, vp = write_world(tmp, vocab={"alpha": 1})
        js = Path(tmp) / "out.json"
        code, out, err = run_main(tmp, ev, al, extra=["--vocab", str(vp), "--json", str(js),
                                                      "--min-net", "2"])
        check(code == 0, err)
        d = json.loads(js.read_text())

        def h(p):
            return hashlib.sha256(Path(p).read_bytes()).hexdigest()

        check(d["split_rule"] == gf.SPLIT_RULE and d["features"] == list(gf.FEATURES), "rule")
        check(d["thresholds"]["min_net"] == 2, d["thresholds"])
        for n in ("ctl", "arm"):
            r = d["runs"][n]
            check(len(r["weights"]) == 6 and len(r["means"]) == 6 and len(r["stds"]) == 6, n)
            check(isinstance(r["bias"], float) and isinstance(r["threshold"], float), n)
            check("held_calibrated" in r, n)
        check("held_plain" in d["runs"]["arm"] and "held_plain" not in d["runs"]["ctl"],
              "plain comparison for arms only")
        s = d["sha256"]
        check(s["eval"] == {str(ev): h(ev)}, s["eval"])
        check(s["aliases"] == h(al) and s["vocab"] == h(vp), s)
        check(s["answers"] == {"ctl": h(Path(tmp) / "ctl-answers.json"),
                               "arm": h(Path(tmp) / "arm-answers.json")}, s["answers"])
        check(s["caches"] == {"ctl": h(Path(tmp) / "ctl-retrieved.json"),
                              "arm": h(Path(tmp) / "ctl-retrieved.json")}, s["caches"])


def test_missing_cache_exits_2():
    with tempfile.TemporaryDirectory() as tmp:
        ev, al, _ = write_world(tmp)
        code, out, err = run_main(tmp, ev, al, arms=("arm:nocache",))
        check(code == 2, code)
        check(str(Path(tmp) / "nocache-retrieved.json") in err, err)
        check(out == "", "nothing printed to stdout")
        (Path(tmp) / "ctl-retrieved.json").unlink()
        code, out, err = run_main(tmp, ev, al, arms=())
        check(code == 2 and "ctl-retrieved.json" in err, (code, err))
        code, out, err = run_main(tmp, ev, al, control="ghost", arms=())
        check(code == 2 and "ghost-answers.json" in err, (code, err))


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"pass {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if failed else 0)
