"""Tests for Phase 15 R13's data side (tsk_20261004_ftdata): scripts/build_qvec.py's
--prompt linux / --sample / --seed, and the pure helpers of scripts/train_embedder.py
(pair loading, hash-chosen holdout, batching, text formats, leakage).

Hermetic (blk_test_env_constraints): /usr/bin/python3, no pytest, no torch, no GPU,
no data/ or models/. `sqlite_vec` is stubbed before build_qvec imports smm.store, as
tests/test_qvec.py does; the databases here are plain sqlite files in a tempdir.
The torch half of train_embedder (LoRA, pooling, loss, merge) is not covered here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sqlite3
import subprocess
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


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build_qvec = _load("build_qvec")
train_embedder = _load("train_embedder")

LINUX_PROMPT = (
    "You help people find things in Linux manual pages. Given one passage from a manual "
    "page, write three short questions a user might ask when they want what this passage "
    "describes, but do not know its terms. Use everyday words, the way a person would "
    "describe the task or the problem. You may name the program, but do not use any "
    "option, flag, variable or file name from the passage. One question per line, no "
    "numbering, nothing else."
)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def make_db(path: Path, rows) -> None:
    """rows: (chunk_id, domain, prefix, text); rowid follows the order given."""
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE chunks(rowid INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE NOT NULL, "
               "domain TEXT NOT NULL, prefix TEXT NOT NULL DEFAULT '', text TEXT NOT NULL)")
    db.executemany("INSERT INTO chunks(chunk_id, domain, prefix, text) VALUES(?,?,?,?)", rows)
    db.commit()
    db.close()


def test_emacs_prompt_unchanged():
    sha = hashlib.sha256(build_qvec.system_prompt(3).encode()).hexdigest()
    check(sha.startswith("b878e89e"), sha)
    check(build_qvec.system_prompt(3, "emacs") == build_qvec.system_prompt(3),
          "emacs is the default prompt")
    check(build_qvec.system_prompt(3).startswith("You help people find things in the GNU Emacs"),
          "emacs wording")


def test_linux_prompt_text():
    check(build_qvec.system_prompt(3, "linux") == LINUX_PROMPT, build_qvec.system_prompt(3, "linux"))
    check("two short questions" in build_qvec.system_prompt(2, "linux"), "n is spelled out")
    check(build_qvec.system_prompt(3, "linux") != build_qvec.system_prompt(3, "emacs"), "differ")


def test_sample_is_deterministic_and_sorted():
    rows = [(f"c{i:03d}", "emacs" if i % 4 else "man", "", f"text {i}") for i in range(80)]
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "i.db"
        make_db(db, rows)
        allrows = build_qvec.load_chunks(db, "emacs")
        a = build_qvec.load_chunks(db, "emacs", sample=12, seed=7)
        b = build_qvec.load_chunks(db, "emacs", sample=12, seed=7)
        c = build_qvec.load_chunks(db, "emacs", sample=12, seed=8)
        d = build_qvec.load_chunks(db, "emacs", sample=12)
        check(a == b and len(a) == 12, "same seed, same subset")
        check(a != c, "a different seed gives a different subset")
        check(all(r in allrows for r in a) and all(r[0] != "c000" for r in a),
              "only chunks of the domain")
        check([r[0] for r in a] == sorted(r[0] for r in a), "kept in rowid order")
        keep = sorted(random.Random(7).sample(range(len(allrows)), 12))
        check(a == [allrows[i] for i in keep], "random.Random(seed).sample over rowid order")
        keep = sorted(random.Random(20261004).sample(range(len(allrows)), 12))
        check(d == [allrows[i] for i in keep], "default seed is 20261004")
        check(build_qvec.load_chunks(db, "emacs", sample=10**6) == allrows, "N past the end: all")
        check(build_qvec.load_chunks(db, "emacs", limit=5) == allrows[:5], "--limit is unchanged")


def test_sample_and_limit_exclusive():
    old = sys.argv
    try:
        sys.argv = ["build_qvec.py", "--sample", "5", "--limit", "5"]
        rc = build_qvec.main()  # exits before any server is contacted
    finally:
        sys.argv = old
    check(rc == 2, rc)


def test_load_pairs_skips_missing_chunks():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        make_db(td / "i.db", [("a", "emacs", "", "ta"), ("b", "emacs", "", "tb")])
        (td / "q1.json").write_text(json.dumps({"a": ["qa1?", "qa2?"], "ghost": ["qg?"], "b": []}))
        (td / "q2.json").write_text(json.dumps({"b": ["qb?", "", "  "], "a": ["qa1?"], "gone": ["x"]}))
        pairs, skipped = train_embedder.load_pairs(td / "i.db", [td / "q1.json", td / "q2.json"])
    check(pairs == [("a", "qa1?"), ("a", "qa2?"), ("b", "qb?")], pairs)
    check(skipped == 2, skipped)


def test_holdout_split_stable_and_disjoint():
    pairs = [(f"chunk{i}", f"q{j}") for i in range(400) for j in range(3)]
    train, dev = train_embedder.holdout_split(pairs, 20261004, 0.1)
    check(len(train) + len(dev) == len(pairs) and set(train).isdisjoint(dev), "partition")
    check({c for c, _ in train}.isdisjoint({c for c, _ in dev}), "no chunk on both sides")
    check(all(len([p for p in dev if p[0] == c]) == 3 for c in {c for c, _ in dev}),
          "a chunk's questions all go the same way")
    check(0.05 < len({c for c, _ in dev}) / 400 < 0.15, len({c for c, _ in dev}))
    # stable: independent of order and of which other chunks exist
    again = train_embedder.holdout_split(list(reversed(pairs[:600])), 20261004, 0.1)[1]
    check({c for c, _ in again} == {c for c, _ in dev if c in {p[0] for p in pairs[:600]}},
          "per-chunk decision does not depend on the rest")
    # the rule itself: sha256 of "<seed>:<chunk_id>"
    for cid in ("chunk0", "chunk17", "chunk399"):
        h = hashlib.sha256(f"20261004:{cid}".encode()).digest()
        want = int.from_bytes(h[:8], "big") / 2**64 < 0.1
        check(train_embedder.is_dev(cid, 20261004, 0.1) == want, cid)
    other = train_embedder.holdout_split(pairs, 1, 0.1)[1]
    check({c for c, _ in other} != {c for c, _ in dev}, "the seed changes the split")


def test_query_and_doc_text_match_runtime():
    from smm.chunk import Chunk
    from smm.embed import query_text as runtime_query
    check(train_embedder.query_text("how do I undo?") == runtime_query("how do I undo?"),
          "query side is smm.embed.query_text")
    check(train_embedder.query_text("x").startswith("Instruct: ") and "\nQuery: x" in
          train_embedder.query_text("x"), "instruction wrapper present")
    for prefix in ("", "Emacs > Files > Visiting\n"):
        ch = Chunk(chunk_id="c", doc_id="d", ord=0, char_start=0, char_end=4, text="body",
                   prefix=prefix)
        check(train_embedder.doc_text(prefix, "body") == ch.embed_text, repr(prefix))


def test_batches_never_repeat_a_chunk():
    pairs = [(f"c{i}", f"q{i}-{j}") for i in range(37) for j in range(3)]
    out = train_embedder.batches(pairs, 8, random.Random(1))
    check(sorted(p for b in out for p in b) == sorted(pairs), "every pair exactly once")
    for b in out:
        check(len(b) <= 8, len(b))
        check(len({c for c, _ in b}) == len(b), f"a chunk twice in one batch: {b}")
    check(sum(1 for b in out if len(b) == 8) >= len(pairs) // 8 - 1, "batches are full")
    check(out == train_embedder.batches(pairs, 8, random.Random(1)), "deterministic")
    check(out != train_embedder.batches(pairs, 8, random.Random(2)), "the rng reorders")
    # worst case: three chunks only, so each batch holds at most 3 pairs
    few = [(f"c{i}", f"q{i}-{j}") for i in range(3) for j in range(9)]
    out = train_embedder.batches(few, 8, random.Random(0))
    check(sorted(p for b in out for p in b) == sorted(few), "every pair exactly once (few chunks)")
    check(all(len({c for c, _ in b}) == len(b) for b in out), "still no repeat")


def test_leakage_jaccard():
    f = train_embedder.leakage
    check(f(["How do I undo a change?"], ["how do i UNDO a change"]) == 1.0, "case and punctuation")
    check(f(["alpha beta"], ["gamma delta"]) == 0.0, "disjoint")
    check(abs(f(["a b c d"], ["a b c e"]) - 3 / 5) < 1e-12, "3 shared of 5 distinct")
    check(abs(f(["a b", "a b c d e", "x"], ["a b c"]) - 2 / 3) < 1e-12, "max over training questions")
    check(abs(f(["a b c"], ["a b c d", "q", "a b"]) - 3 / 4) < 1e-12, "max over eval questions")
    check(f([], ["a"]) == 0.0 and f(["a"], []) == 0.0 and f(["!!!"], ["???"]) == 0.0, "empties")
    # brute force agrees on a random sample
    rng = random.Random(3)
    words = [f"w{i}" for i in range(12)]
    tr = [" ".join(rng.sample(words, rng.randint(1, 6))) for _ in range(40)]
    ev = [" ".join(rng.sample(words, rng.randint(1, 6))) for _ in range(15)]
    want = max(len(set(a.split()) & set(b.split())) / len(set(a.split()) | set(b.split()))
               for a in ev for b in tr)
    check(abs(f(tr, ev) - want) < 1e-12, (f(tr, ev), want))


def test_module_imports_without_torch():
    code = (
        "import sys; sys.modules['torch'] = None; sys.modules['transformers'] = None\n"
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('te', r'{ROOT / 'scripts' / 'train_embedder.py'}')\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "print(m.query_text('q').startswith('Instruct:'))\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    check(r.returncode == 0 and r.stdout.strip() == "True", r.stderr[-500:])
    check("torch" not in sys.modules or sys.modules["torch"] is None, "torch leaked into this process")


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
