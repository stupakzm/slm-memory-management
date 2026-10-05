"""Hermetic tests for scripts/retrieval_drift.py (stdlib only, tempdir fixtures)."""

import contextlib
import io
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import retrieval_drift as rd  # noqa: E402


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = rd.main(argv)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue()


def write_pool(d, name, qids):
    p = Path(d) / name
    p.write_text("".join(json.dumps({"qid": q, "question": "x"}) + "\n" for q in qids))
    return str(p)


def hit(cid, score=0.9, rerank=None):
    h = {"chunk_id": cid, "score": score}
    if rerank is not None:
        h["rerank_score"] = rerank
    return h


def write_json(d, name, obj):
    p = Path(d) / name
    p.write_text(json.dumps(obj))
    return str(p)


def test_sample_matches_phase13_rule():
    with tempfile.TemporaryDirectory() as d:
        a = write_pool(d, "a.jsonl", [f"a{i}" for i in range(30)])
        b = write_pool(d, "b.jsonl", [f"b{i}" for i in range(30)])
        out = str(Path(d) / "q.json")
        code, _ = run(["sample", "--pool", a, b, "--n", "10", "--seed", "13", "--out", out])
        assert code == 0
        pool = [f"a{i}" for i in range(30)] + [f"b{i}" for i in range(30)]
        assert json.loads(Path(out).read_text()) == random.Random(13).sample(pool, 10)
        c = write_pool(d, "c.jsonl", ["a1"])
        code, _ = run(["sample", "--pool", a, c, "--n", "3", "--out", out])
        assert code == 2


def test_sample_writes_qids_json():
    with tempfile.TemporaryDirectory() as d:
        a = write_pool(d, "a.jsonl", ["q1", "q2", "q3", "q4"])
        out = str(Path(d) / "q.json")
        assert run(["sample", "--pool", a, "--n", "2", "--out", out])[0] == 0
        got = json.loads(Path(out).read_text())
        assert isinstance(got, list) and len(got) == 2 and set(got) <= {"q1", "q2", "q3", "q4"}


def test_compare_counts_topk_and_gate():
    with tempfile.TemporaryDirectory() as d:
        base = [hit("c1", rerank=0.9), hit("c2"), hit("c3")]
        a = {
            "same": base,
            "top": base,
            "gate": [hit("c1", rerank=0.9)],
            "both": [hit("c1", rerank=0.9), hit("c2")],
            "beyond": base,
        }
        b = {
            "same": base,
            "top": [hit("c2", rerank=0.9), hit("c1"), hit("c3")],
            "gate": [hit("c1", rerank=0.1)],
            "both": [hit("c9", rerank=0.1)],
            "beyond": base + [hit("c4")],
        }
        qs = write_json(d, "q.json", ["same", "top", "gate", "both", "beyond"])
        code, out = run(["compare", "--a", write_json(d, "a.json", a),
                         "--b", write_json(d, "b.json", b), "--qids", qs, "--k", "3"])
        assert code == 0
        lines = out.strip().splitlines()
        assert lines[:3] == ["top: top-k", "gate: gate", "both: top-k,gate"], lines
        assert lines[-1] == "differ: 3 of 5 (top-k 2, gate 2)", lines
        # no-hits gate is -inf (< threshold); k=3 ignores the 4th hit
        a2, b2 = {"x": []}, {"x": [hit("c1", rerank=0.9)]}
        qs2 = write_json(d, "q2.json", ["x"])
        _, out = run(["compare", "--a", write_json(d, "a2.json", a2),
                      "--b", write_json(d, "b2.json", b2), "--qids", qs2])
        assert out.strip().splitlines()[-1] == "differ: 1 of 1 (top-k 1, gate 1)", out


def test_compare_reads_both_cache_shapes():
    with tempfile.TemporaryDirectory() as d:
        hits = [hit("c1", rerank=0.9), hit("c2")]
        a = {"q": hits}
        b = {"q": {"hits": hits, "gate_hits": hits}}
        c = {"q": {"hits": hits, "gate_hits": [hit("c1", rerank=0.1)]}}
        qs = write_json(d, "q.json", ["q"])
        pa, pb, pc = (write_json(d, n, o) for n, o in (("a", a), ("b", b), ("c", c)))
        _, out = run(["compare", "--a", pa, "--b", pb, "--qids", qs])
        assert out.strip() == "differ: 0 of 1 (top-k 0, gate 0)", out
        _, out = run(["compare", "--a", pa, "--b", pc, "--qids", qs])
        assert out.strip().splitlines() == ["q: gate", "differ: 1 of 1 (top-k 0, gate 1)"], out


def test_compare_missing_qid_exits_2():
    with tempfile.TemporaryDirectory() as d:
        full = {"q1": [hit("c1")], "q2": [hit("c1")]}
        part = {"q1": [hit("c1")]}
        qs = write_json(d, "q.json", ["q1", "q2"])
        pf, pp = write_json(d, "f.json", full), write_json(d, "p.json", part)
        assert run(["compare", "--a", pf, "--b", pp, "--qids", qs])[0] == 2
        assert run(["compare", "--a", pp, "--b", pf, "--qids", qs])[0] == 2


def test_summary_line_format():
    with tempfile.TemporaryDirectory() as d:
        c = {"q1": [hit("c1")], "q2": [hit("c2")]}
        p = write_json(d, "c.json", c)
        qs = write_json(d, "q.json", ["q1", "q2"])
        code, out = run(["compare", "--a", p, "--b", p, "--qids", qs])
        assert code == 0
        assert out == "differ: 0 of 2 (top-k 0, gate 0)\n", repr(out)


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"pass {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if failed else 0)
