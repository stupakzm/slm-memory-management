"""Tests for scripts/reembed.py, scripts/embed_parity.py and servers.sh's
SMM_EMBED_MODEL (Phase 15 R13 deployment).

Hermetic (blk_test_env_constraints): stdlib only, sqlite_vec stubbed in
sys.modules, plain sqlite3 databases in a tempdir, a fake embedder and vector
writer, and a stub llama-server that only records its argv and exits. No torch,
no servers, no GPU, no data/ or models/.
"""

from __future__ import annotations

import importlib.util
import os
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


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


reembed = _load("reembed", "scripts/reembed.py")
parity = _load("embed_parity", "scripts/embed_parity.py")
SERVERS = ROOT / "scripts" / "servers.sh"
DEFAULT_MODEL = "Qwen3-Embedding-0.6B-Q8_0.gguf"


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def make_db(path, rows):
    """rows: [(prefix, text)], rowids 1..n."""
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE chunks(rowid INTEGER PRIMARY KEY, chunk_id TEXT, prefix TEXT, text TEXT)")
    db.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
    db.executemany("INSERT INTO chunks(chunk_id,prefix,text) VALUES(?,?,?)",
                   [(f"c{i}", p, t) for i, (p, t) in enumerate(rows)])
    db.commit()
    return db


ROWS = [("[a] ", "alpha"), ("", "beta"), ("[c] ", "gamma"), ("", "delta"), ("", "eps")]


class Fake:
    def __init__(self):
        self.texts, self.written = [], []

    def embed(self, texts):
        self.texts.extend(texts)
        return [[float(len(t))] for t in texts]

    def write(self, db, rid, vec):
        self.written.append((rid, vec))


def test_reembed_refuses_src_as_out():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.db"
        make_db(p, ROWS).close()
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "reembed.py"),
                            "--src", str(p), "--out", str(p)], capture_output=True, text=True)
        check(r.returncode == 2, f"expected exit 2, got {r.returncode}: {r.stderr!r}")
        try:
            reembed.prepare_out(p, p)
            raise AssertionError("prepare_out should refuse")
        except SystemExit:
            pass


def test_reembed_replaces_every_vector_in_rowid_order():
    with tempfile.TemporaryDirectory() as d:
        db = make_db(Path(d) / "a.db", ROWS)
        f = Fake()
        n = reembed.reembed(db, f.embed, f.write, batch=2, log=lambda *_: None)
        check(n == 5, n)
        check([r for r, _ in f.written] == [1, 2, 3, 4, 5], f.written)
        check(f.written[0][1] == [float(len("[a] alpha"))], f.written[0])


def test_reembed_resumes_from_progress():
    with tempfile.TemporaryDirectory() as d:
        db = make_db(Path(d) / "a.db", ROWS)
        f = Fake()
        calls = {"n": 0}

        def flaky(texts):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("server died")
            return f.embed(texts)

        try:
            reembed.reembed(db, flaky, f.write, batch=2, log=lambda *_: None)
            raise AssertionError("should have raised")
        except RuntimeError:
            pass
        done = db.execute("SELECT value FROM meta WHERE key='reembed_done_rowid'").fetchone()[0]
        check(done == "2", done)
        f2 = Fake()
        n = reembed.reembed(db, f2.embed, f2.write, batch=2, log=lambda *_: None)
        check(n == 3 and [r for r, _ in f2.written] == [3, 4, 5], f2.written)


def test_reembed_leaves_chunks_table_alone():
    with tempfile.TemporaryDirectory() as d:
        db = make_db(Path(d) / "a.db", ROWS)
        before = db.execute("SELECT rowid,chunk_id,prefix,text FROM chunks ORDER BY rowid").fetchall()
        changes0 = db.total_changes
        f = Fake()
        reembed.reembed(db, f.embed, f.write, batch=3, log=lambda *_: None)
        after = db.execute("SELECT rowid,chunk_id,prefix,text FROM chunks ORDER BY rowid").fetchall()
        check(before == after, "chunks rows changed")
        # only meta writes (one per batch) touched the db; the fake writer wrote nothing
        check(db.total_changes - changes0 == 2, db.total_changes - changes0)


def test_reembed_uses_runtime_doc_text():
    from smm.chunk import Chunk
    with tempfile.TemporaryDirectory() as d:
        db = make_db(Path(d) / "a.db", ROWS)
        f = Fake()
        reembed.reembed(db, f.embed, f.write, batch=10, log=lambda *_: None)
        want = [Chunk(chunk_id="x", doc_id="d", ord=0, char_start=0, char_end=0,
                      text=t, prefix=p).embed_text for p, t in ROWS]
        check(f.texts == want, f"{f.texts!r} != {want!r}")


def test_parity_reports_min_cosine_and_fails_below():
    hf = {"docs": [[1.0, 0.0], [0.0, 1.0]], "queries": [[1.0, 0.0]]}
    same = {"docs": [[1.0, 0.0], [0.0, 1.0]], "queries": [[1.0, 0.0]]}
    stats, ok = parity.parity_report(hf, same, 0.99)
    check(ok and abs(stats["docs"]["min"] - 1.0) < 1e-9, stats)
    off = {"docs": [[1.0, 0.0], [0.6, 0.8]], "queries": [[1.0, 0.0]]}
    stats, ok = parity.parity_report(hf, off, 0.99)
    check(not ok, "should fail below threshold")
    check(abs(stats["docs"]["min"] - 0.8) < 1e-9, stats["docs"])
    check(abs(stats["docs"]["mean"] - 0.9) < 1e-9, stats["docs"])
    check(stats["queries"]["min"] > 0.99, stats["queries"])
    _, ok = parity.parity_report(hf, off, 0.5)
    check(ok, "should pass at a lower threshold")


def _argv_for(profile, model):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        bindir = tmp / "bin"
        bindir.mkdir()
        exe = bindir / "llama-server"
        exe.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$SMM_TEST_ARGV"\nexit 1\n')
        exe.chmod(0o755)
        env = dict(os.environ)
        env.pop("SMM_EMBED_MODEL", None)
        env.update(SMM_RUN=str(tmp / "run"), LLAMA_CPP=str(bindir),
                   SMM_MODELS="/m", SMM_TEST_ARGV=str(tmp / "argv"))
        if model:
            env["SMM_EMBED_MODEL"] = model
        subprocess.run(["bash", str(SERVERS), "start", profile], env=env,
                       capture_output=True, text=True, timeout=60)
        lines = (tmp / "argv").read_text().splitlines()
        return lines[lines.index("-m") + 1]


def test_servers_embed_model_default_unchanged():
    for profile in ("embedder", "embedder-lean"):
        got = _argv_for(profile, None)
        check(got == f"/m/{DEFAULT_MODEL}", f"{profile}: {got}")


def test_servers_embed_model_override_both_profiles():
    for profile in ("embedder", "embedder-lean"):
        got = _argv_for(profile, "ft-q8_0.gguf")
        check(got == "/m/ft-q8_0.gguf", f"{profile}: {got}")


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)
