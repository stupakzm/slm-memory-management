"""Tests for scripts/judge_pack.py and scripts/judge_score.py (blind judge tools).

Hermetic (blk_test_env_constraints): stdlib only, synthetic eval/answers files, a
tiny man.jsonl-shaped corpus and a markdown dir in a tempdir, never data/.
sqlite_vec is stubbed first, matching the repo's test convention.
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


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


jp = _load("judge_pack", "scripts/judge_pack.py")
js = _load("judge_score", "scripts/judge_score.py")
sr = jp.sr


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def row(qid, answer="", abstained=False, evidence=True):
    return {"qid": qid, "answer": answer, "abstained": abstained,
            "evidence_retrieved": evidence}


def erow(qid, kind="answerable", tok=("alpha",), **kw):
    return {"qid": qid, "kind": kind, "question": f"question {qid}?",
            "answer_contains": list(tok), "gold_sec_ids": [], **kw}


def sec(doc_id, name, text):
    return {"sec_id": f"{doc_id}#{name}", "heading": name, "level": 1,
            "parent": None, "text": text}


def corpus(tmp):
    """(man.jsonl path, markdown dir, md sec_id): one man doc, one markdown doc."""
    tmp = Path(tmp)
    man = tmp / "man.jsonl"
    doc = {"doc_id": "d.1", "name": "d", "section": "1", "summary": "a doc",
           "sections": [sec("d.1", "OPTIONS", "alpha does the thing"),
                        sec("d.1", "LONG", "x" * 7000),
                        sec("d.1", "OTHER", "beta does another")]}
    man.write_text(json.dumps(doc) + "\n")
    md = tmp / "md"
    md.mkdir(exist_ok=True)
    (md / "notes.md").write_text("# Intro\n\nmarkdown gamma text\n")
    docs = jp.rh.load_docs(man, [(md, "emacs")])
    md_sec = docs["notes.emacs"]["sections"][0]["sec_id"]
    return man, md, md_sec


def write_eval(tmp, rows):
    ev = Path(tmp) / "eval.jsonl"
    ev.write_text("".join(json.dumps(r) + "\n" for r in rows))
    al = Path(tmp) / "aliases.json"
    al.write_text("{}")
    return ev, al


def write_runs(tmp, runs):
    for name, rows in runs.items():
        (Path(tmp) / f"{name}-answers.json").write_text(json.dumps({"results": rows}))


def call(fn, argv, tmp):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = fn(argv, results_dir=tmp)
    return code, out.getvalue(), err.getvalue()


def pack(tmp, eval_rows, runs, pairs=("c:a",), controls=0, extra=(), seed=16):
    """Run judge_pack; returns (code, stdout, stderr, items, key)."""
    tmp = Path(tmp)
    man, md, _ = corpus(tmp)
    ev, al = write_eval(tmp, eval_rows)
    write_runs(tmp, runs)
    out_dir, key = tmp / "blind", tmp / "key.jsonl"
    argv = ["--eval", str(ev), "--aliases", str(al), "--man", str(man),
            "--md", f"{md}:emacs", "--controls", str(controls), "--seed", str(seed),
            "--out-dir", str(out_dir), "--key", str(key), *extra]
    for p in pairs:
        argv += ["--pair", p]
    code, out, err = call(jp.main, argv, tmp)
    items = []
    for f in sorted(out_dir.glob("items-*.jsonl")):
        items += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    keys = ([json.loads(l) for l in key.read_text().splitlines() if l.strip()]
            if key.exists() else [])
    return code, out, err, items, keys


def basic():
    ev = [erow("q1"), erow("q2"), erow("q3"), erow("q4"),
          erow("u1", kind="unanswerable")]
    c = [row("q1", "use alpha"), row("q2", "nothing"), row("q3", "alpha"),
         row("q4", "no"), row("u1", "alpha")]
    a = [row("q1", "wrong"), row("q2", "try alpha"), row("q3", "alpha"),
         row("q4", "nope"), row("u1", "no")]
    return ev, {"c": c, "a": a}


def by_qid(items, keys):
    """[(qid, role, answer)] for the non-hidden entries, joined through the key."""
    view = {i["id"]: i for i in items}
    return sorted((e["qid"], e["role"], view[k["id"]]["answer"])
                  for k in keys for e in k["entries"])


def test_pack_selects_rows_whose_correctness_differs():
    with tempfile.TemporaryDirectory() as t:
        ev, runs = basic()
        code, out, err, items, keys = pack(t, ev, runs)
        check(code == 0, err)
        got = by_qid(items, keys)
        check(got == [("q1", "arm", "wrong"), ("q1", "control", "use alpha"),
                      ("q2", "arm", "try alpha"), ("q2", "control", "nothing")], got)
        check(out.strip() == "items 4 (controls 0) from 2 differing rows in 1 pair(s); "
                             "batches 1", out)
        check((Path(t) / "blind" / "rubric.md").read_text().count("CORRECT") >= 2, "rubric")
        r, _, e = call(jp.main, ["--eval", "x", "--pair", "c:a", "--out-dir", "o",
                                 "--key", "k"], t)
        check(r == 2 and "x" in e, "a missing input must return 2 naming the path")


def test_identical_answers_are_judged_once():
    with tempfile.TemporaryDirectory() as t:
        ev = [erow("q1"), erow("q2")]
        c = [row("q1", "same wrong"), row("q2", "alpha")]
        a1 = [row("q1", "alpha"), row("q2", "x")]
        a2 = [row("q1", "alpha"), row("q2", "y")]
        code, out, err, items, keys = pack(t, ev, {"c": c, "a1": a1, "a2": a2},
                                           pairs=("c:a1", "c:a2"))
        check(code == 0, err)
        # q1: control answer and arm answer are each shared by both pairs.
        check(len(items) == 5, [i["answer"] for i in items])
        shared = [k for k in keys if len(k["entries"]) == 2]
        check(len(shared) == 3, [k["entries"] for k in keys])
        shared_pairs = sorted(e["pair"] for e in
                              next(k for k in keys if k["entries"][0]["role"] == "control"
                                   and k["entries"][0]["qid"] == "q1")["entries"])
        check(shared_pairs == ["c:a1", "c:a2"], shared_pairs)


def test_refusals_are_not_items():
    with tempfile.TemporaryDirectory() as t:
        ev = [erow("q1"), erow("q2")]
        c = [row("q1", "alpha", abstained=True), row("q2", "alpha")]
        a = [row("q1", "alpha"), row("q2", "x", abstained=True)]
        code, out, err, items, keys = pack(t, ev, {"c": c, "a": a})
        check(code == 0, err)
        check(by_qid(items, keys) == [("q1", "arm", "alpha"), ("q2", "control", "alpha")],
              by_qid(items, keys))
        check(out.startswith("items 2 (controls 0) from 2 differing rows"), out)


def test_gold_sections_resolved_from_sec_ids():
    with tempfile.TemporaryDirectory() as t:
        _, _, md_sec = corpus(t)
        ev = [erow("q1", gold_sec_ids=["d.1#OPTIONS", "nope#MISSING", md_sec]),
              erow("q2", gold_sec_ids=["d.1#LONG"]),
              erow("q2.y1", variant_of="q2"),
              erow("q3", gold_sec_ids=["nope#MISSING"]),
              erow("q4", answer_contains=["alpha", "beta"], gold_sec_ids=["d.1#OTHER"])]
        c = [row(q, "x") for q in ("q1", "q2", "q2.y1", "q3", "q4")]
        a = [row(q, "alpha beta") for q in ("q1", "q2", "q2.y1", "q3", "q4")]
        code, out, err, items, keys = pack(t, ev, {"c": c, "a": a})
        check(code == 0, err)
        view = {i["id"]: i for i in items}
        by_q = {k["entries"][0]["qid"]: view[k["id"]] for k in keys
                if k["entries"][0]["role"] == "arm"}
        check([s["sec_id"] for s in by_q["q1"]["gold_sections"]] == ["d.1#OPTIONS", md_sec],
              by_q["q1"]["gold_sections"])
        check(by_q["q1"]["gold_sections"][0]["text"] == "alpha does the thing", "text")
        check("markdown gamma" in by_q["q1"]["gold_sections"][1]["text"], "markdown section")
        check(len(by_q["q2"]["gold_sections"][0]["text"]) == 6000, "cut at 6000")
        check([s["sec_id"] for s in by_q["q2.y1"]["gold_sections"]] == ["d.1#LONG"],
              "a variant uses its base row's gold_sec_ids")
        check(by_q["q3"]["gold_sections"] == [], "unresolvable: still an item, no sections")
        check(by_q["q4"]["reference_answer"] == ["alpha", "beta"], by_q["q4"])
        check(by_q["q4"]["question"] == "question q4?", by_q["q4"])
        check(set(by_q["q1"]) == {"id", "question", "answer", "reference_answer",
                                  "gold_sections"}, set(by_q["q1"]))


def test_view_files_never_contain_run_names_or_qids():
    with tempfile.TemporaryDirectory() as t:
        ev = [erow(f"zq{i}", question="how do I do it?") for i in range(6)]
        c = [row(f"zq{i}", "alpha" if i % 2 else "no") for i in range(6)]
        a = [row(f"zq{i}", "no" if i % 2 else "alpha") for i in range(6)]
        code, out, err, items, keys = pack(t, ev, {"ctlrun": c, "armrun": a},
                                           pairs=("ctlrun:armrun",), controls=0)
        check(code == 0, err)
        check(len(items) == 12, len(items))
        blind = Path(t) / "blind"
        files = sorted(blind.iterdir())
        check(any(f.name == "rubric.md" for f in files), "rubric.md written")
        check(not (blind / "key.jsonl").exists(), "key lives outside the blind dir")
        text = "".join(f.read_text() for f in files)
        for needle in ("ctlrun", "armrun", "answers.json", "zq0", "zq5", "ctlrun:armrun"):
            check(needle not in text, f"{needle!r} leaked into the blind files")
        key_text = (Path(t) / "key.jsonl").read_text()
        check("ctlrun" in key_text and "zq0" in key_text, "the key holds them")
        # a key inside the blind directory is refused
        man, md, _ = corpus(t)
        ev_path, al = Path(t) / "eval.jsonl", Path(t) / "aliases.json"
        code, _, err = call(jp.main, ["--eval", str(ev_path), "--aliases", str(al),
                                      "--man", str(man), "--md", f"{md}:emacs",
                                      "--pair", "ctlrun:armrun", "--out-dir", str(blind),
                                      "--key", str(blind / "key.jsonl")], t)
        check(code == 2, "key inside out-dir must be refused")


def test_key_maps_every_item():
    with tempfile.TemporaryDirectory() as t:
        ev, runs = basic()
        runs["c"].append(row("q5", "alpha"))
        runs["a"].append(row("q5", "alpha"))
        ev = ev + [erow("q5")]
        code, out, err, items, keys = pack(t, ev, runs, controls=1)
        check(code == 0, err)
        check([i["id"] for i in items] != [], "items")
        check(sorted(i["id"] for i in items) == sorted(k["id"] for k in keys),
              "every item has exactly one key line")
        check(sorted(i["id"] for i in items) == [f"J{n:03d}" for n in range(1, len(items) + 1)],
              [i["id"] for i in items])
        for k in keys:
            for e in k["entries"]:
                check(set(e) == {"qid", "run", "pair", "role"}, e)
                check(e["role"] in ("control", "arm", "hidden_control"), e)
                check(e["run"] in ("c-answers.json", "a-answers.json"), e)
                check(e["pair"] == "c:a", e)
        view = {i["id"]: i for i in items}
        for k in keys:
            e = k["entries"][0]
            src = runs[e["run"].split("-")[0]]
            want = next(r["answer"] for r in src if r["qid"] == e["qid"])
            check(view[k["id"]]["answer"] == want, (k, view[k["id"]]))
        check(sum(e["role"] == "hidden_control" for k in keys for e in k["entries"]) == 1,
              "one hidden control")


def test_hidden_controls_are_credited_in_both_runs():
    with tempfile.TemporaryDirectory() as t:
        names = [f"h{i}" for i in range(8)]
        ev = [erow(q) for q in names] + [erow(q) for q in ("n1", "n2", "n3", "n4", "d1")]
        c = [row(q, "alpha") for q in names]
        a = [row(q, "alpha") for q in names]
        c += [row("n1", "alpha", evidence=False), row("n2", "alpha"),
              row("n3", "wrong"), row("n4", "alpha", abstained=True), row("d1", "alpha")]
        a += [row("n1", "alpha"), row("n2", "alpha", evidence=False),
              row("n3", "alpha"), row("n4", "alpha"), row("d1", "wrong")]
        code, out, err, items, keys = pack(t, ev, {"c": c, "a": a}, controls=3)
        check(code == 0, err)
        hidden = [(e["qid"], e["run"]) for k in keys for e in k["entries"]
                  if e["role"] == "hidden_control"]
        check(len(hidden) == 3, hidden)
        check(all(q in names and r == "c-answers.json" for q, r in hidden), hidden)
        check("(controls 3)" in out, out)
        # asking for more than the pool holds takes the whole pool, never a bad row
        code, out, err, items, keys = pack(t, ev, {"c": c, "a": a}, controls=50)
        hidden = sorted(e["qid"] for k in keys for e in k["entries"]
                        if e["role"] == "hidden_control")
        check(hidden == sorted(names), hidden)


def test_items_shuffled_deterministically_by_seed():
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2, \
            tempfile.TemporaryDirectory() as t3:
        ev = [erow(f"q{i}") for i in range(10)]
        c = [row(f"q{i}", "no") for i in range(10)]
        a = [row(f"q{i}", "alpha") for i in range(10)]
        runs = {"c": c, "a": a}
        r1 = pack(t1, ev, runs, seed=3)
        r2 = pack(t2, ev, runs, seed=3)
        r3 = pack(t3, ev, runs, seed=4)
        check(r1[3] == r2[3] and r1[4] == r2[4], "same seed, same files")
        check(r1[3] != r3[3] or r1[4] != r3[4], "another seed shuffles differently")
        qids = [k["entries"][0]["qid"] for k in r1[4]]
        check(qids != sorted(qids), "items are not left in qid order")


def test_batches_respect_batch_size():
    with tempfile.TemporaryDirectory() as t:
        ev = [erow(f"q{i}") for i in range(7)]
        c = [row(f"q{i}", "no", abstained=True) for i in range(7)]   # refusals: one item per row
        a = [row(f"q{i}", "alpha") for i in range(7)]
        code, out, err, items, keys = pack(t, ev, {"c": c, "a": a}, extra=["--batch-size", "3"])
        check(code == 0, err)
        files = sorted(f.name for f in (Path(t) / "blind").glob("items-*.jsonl"))
        check(files == ["items-01.jsonl", "items-02.jsonl", "items-03.jsonl"], files)
        sizes = [len((Path(t) / "blind" / f).read_text().splitlines()) for f in files]
        check(sizes == [3, 3, 1], sizes)
        check(out.strip().endswith("batches 3"), out)
        check([i["id"] for i in items] == [f"J{n:03d}" for n in range(1, 8)],
              "ids run across batches")


def scoring_fixture(t, specs, controls=0, hidden_n=0):
    """specs: [(qid, control row, arm row)]; returns (key lines, eval file, aliases)."""
    ev = [erow(q) for q, _, _ in specs] + [erow(f"h{i}") for i in range(hidden_n)]
    c = [r for _, r, _ in specs] + [row(f"h{i}", "alpha") for i in range(hidden_n)]
    a = [r for _, _, r in specs] + [row(f"h{i}", "alpha") for i in range(hidden_n)]
    code, out, err, items, keys = pack(t, ev, {"c": c, "a": a}, controls=hidden_n)
    check(code == 0, err)
    return keys, Path(t) / "eval.jsonl", Path(t) / "aliases.json"


def write_judgments(tmp, keys, label_of, name="judgments.jsonl", nested=False):
    """label_of(qid, role) -> label or None (no judgment); hidden controls: label_of(qid, 'hidden_control')."""
    lines = []
    for k in keys:
        e = k["entries"][0]
        lab = label_of(e["qid"], e["role"])
        if lab is not None:
            lines.append({"id": k["id"], "label": lab})
    p = Path(tmp) / name
    if nested:
        p.write_text(json.dumps({"items": lines[: len(lines) // 2]}) + "\n"
                     + json.dumps({"items": lines[len(lines) // 2:]}) + "\n")
    else:
        p.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return p


def score(tmp, keys, ev, al, judgments):
    kp = Path(tmp) / "key.jsonl"
    argv = ["--key", str(kp), "--eval", str(ev), "--aliases", str(al)]
    for j in judgments:
        argv += ["--judgments", str(j)]
    return call(js.main, argv, tmp)


def test_score_judged_net_and_sign_test():
    with tempfile.TemporaryDirectory() as t:
        specs = [(f"g{i}", row(f"g{i}", "no"), row(f"g{i}", "alpha")) for i in range(5)]
        specs += [("l1", row("l1", "alpha"), row("l1", "no")),
                  ("l2", row("l2", "alpha"), row("l2", "no"))]
        keys, ev, al = scoring_fixture(t, specs)

        def lab(q, role):
            if q in ("g0", "g1", "g2", "g3"):
                return "CORRECT" if role == "arm" else "WRONG"
            if q == "g4":
                return "WRONG"             # judged wrong on both sides: no change
            if q == "l1":
                return "CORRECT" if role == "control" else "WRONG"
            return "CORRECT"               # l2: judged correct on both sides

        # one file of plain lines, one of {"items": [...]} objects: split the labels
        full = write_judgments(t, keys, lab)
        lines = full.read_text().splitlines()
        a_file, b_file = Path(t) / "j1.jsonl", Path(t) / "j2.jsonl"
        a_file.write_text("\n".join(lines[:5]) + "\n")
        b_file.write_text(json.dumps({"items": [json.loads(l) for l in lines[5:]]}) + "\n")
        code, out, err = score(t, keys, ev, al, [a_file, b_file])
        check(code == 0, err)
        want = ("c:a rows 7 labelled lost 2 gained 5 net +3 | judged lost 1 gained 4 "
                f"net +3 p {sr.sign_test_p(4, 1):.3g} unclear 0")
        check(out.splitlines()[0] == want, out)
        check(out.splitlines()[-1] == "hidden controls judged CORRECT 0/0", out)


def test_score_counts_refusal_as_wrong():
    with tempfile.TemporaryDirectory() as t:
        specs = [("r1", row("r1", "alpha", abstained=True), row("r1", "alpha")),
                 ("r2", row("r2", "alpha"), row("r2", "alpha", abstained=True)),
                 ("r3", row("r3", "alpha", abstained=True), row("r3", "alpha"))]
        keys, ev, al = scoring_fixture(t, specs)
        check(len(keys) == 3, "refusal sides are not items")
        j = write_judgments(t, keys, lambda q, role: {"r1": "CORRECT", "r2": "CORRECT",
                                                      "r3": "WRONG"}[q])
        code, out, err = score(t, keys, ev, al, [j])
        check(code == 0, err)
        want = ("c:a rows 3 labelled lost 1 gained 2 net +1 | judged lost 1 gained 1 "
                f"net +0 p {sr.sign_test_p(1, 1):.3g} unclear 0")
        check(out.splitlines()[0] == want, out)


def test_score_reports_unclear_and_control_reliability():
    with tempfile.TemporaryDirectory() as t:
        specs = [("u1", row("u1", "no"), row("u1", "alpha")),
                 ("u2", row("u2", "no"), row("u2", "alpha")),
                 ("u3", row("u3", "no"), row("u3", "alpha")),
                 ("u4", row("u4", "no"), row("u4", "alpha"))]
        keys, ev, al = scoring_fixture(t, specs, hidden_n=4)
        check(sum(k["entries"][0]["role"] == "hidden_control" for k in keys) == 4, "4 controls")
        hid = sorted(k["entries"][0]["qid"] for k in keys
                     if k["entries"][0]["role"] == "hidden_control")

        def lab(q, role):
            if role == "hidden_control":
                return "WRONG" if q == hid[0] else "CORRECT"
            if q == "u1":
                return "UNCLEAR" if role == "control" else "CORRECT"
            if q == "u2":
                return None if role == "arm" else "WRONG"     # no judgment: UNCLEAR
            return "CORRECT" if role == "arm" else "WRONG"

        j = write_judgments(t, keys, lab)
        code, out, err = score(t, keys, ev, al, [j])
        check(code == 0, err)
        want = ("c:a rows 4 labelled lost 0 gained 4 net +4 | judged lost 0 gained 2 "
                f"net +2 p {sr.sign_test_p(2, 0):.3g} unclear 2")
        check(out.splitlines()[0] == want, out)
        check(out.splitlines()[-1] == "hidden controls judged CORRECT 3/4", out)


def test_score_rejects_unknown_item_ids():
    with tempfile.TemporaryDirectory() as t:
        specs = [("g1", row("g1", "no"), row("g1", "alpha"))]
        keys, ev, al = scoring_fixture(t, specs)
        j = Path(t) / "bad.jsonl"
        j.write_text(json.dumps({"id": keys[0]["id"], "label": "CORRECT"}) + "\n"
                     + json.dumps({"id": "J999", "label": "WRONG"}) + "\n")
        code, out, err = score(t, keys, ev, al, [j])
        check(code == 2, (code, out, err))
        check("J999" in err and out == "", (out, err))
        ok = Path(t) / "ok.jsonl"
        ok.write_text(json.dumps({"id": keys[0]["id"], "label": "CORRECT"}) + "\n")
        check(score(t, keys, ev, al, [ok])[0] == 0, "known ids pass")


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"pass {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)
