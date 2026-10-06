"""Tests for scripts/reembed_headings.py (heading-bearing re-embedding of an index).

Hermetic (blk_test_env_constraints): stdlib only, sqlite_vec stubbed in sys.modules,
synthetic documents, plain sqlite3 databases in a tempdir, a fake embedder and
vector writer. No torch, no servers, no data/ or models/.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
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

from smm.chunk import flatten  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rh = _load("reembed_headings", "scripts/reembed_headings.py")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def make_doc(doc_id="d.man", secs=(("NAME", "tar - archive"), ("OPTIONS", "-x extract\n-c create"),
                                   ("EMPTY", ""), ("SEE ALSO", "cp(1)"))):
    return {"doc_id": doc_id, "name": "d", "section": "1", "summary": "a doc",
            "sections": [{"sec_id": f"{doc_id}#{i}", "heading": h, "text": t}
                         for i, (h, t) in enumerate(secs)]}


def make_db(chunks):
    """chunks: [(doc_id, char_start, char_end, prefix, text)], rowids 1..n."""
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE chunks(rowid INTEGER PRIMARY KEY, chunk_id TEXT, doc_id TEXT, "
               "ord INTEGER, char_start INTEGER, char_end INTEGER, prefix TEXT, text TEXT)")
    db.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
    db.executemany("INSERT INTO chunks(chunk_id,doc_id,ord,char_start,char_end,prefix,text) "
                   "VALUES(?,?,?,?,?,?,?)",
                   [(f"{d}:{i}", d, i, a, b, p, t) for i, (d, a, b, p, t) in enumerate(chunks)])
    db.commit()
    return db


def chunks_of(doc, windows, prefix="P\n"):
    flat = flatten(doc)
    return [(doc["doc_id"], a, b, prefix, flat[a:b].strip()) for a, b in windows]


class Fake:
    def __init__(self):
        self.texts, self.written = [], []

    def embed(self, texts):
        self.texts.extend(texts)
        return [[float(len(t))] for t in texts]

    def write(self, db, rid, vec):
        self.written.append((rid, vec))


def test_heading_offsets_follow_flatten():
    doc = make_doc()
    flat = flatten(doc)
    offs = rh.heading_offsets(doc)
    check([h for _, h in offs] == ["NAME", "OPTIONS", "EMPTY", "SEE ALSO"], offs)
    check(offs[0][0] == 0, offs)
    for o, h in offs:
        check(flat[o:o + len(h)] == h, f"heading {h!r} not at {o}: {flat[o:o + len(h)]!r}")
        check(o == 0 or flat[o - 1] == "\n", f"offset {o} not at a line start")


def test_heading_for_picks_last_heading_at_or_before_start():
    offs = [(0, "A"), (10, "B"), (30, "C")]
    check(rh.heading_for(offs, 0) == "A", "start at first heading")
    check(rh.heading_for(offs, 9) == "A", "just before B")
    check(rh.heading_for(offs, 10) == "B", "exactly at B")
    check(rh.heading_for(offs, 29) == "B", "inside B")
    check(rh.heading_for(offs, 500) == "C", "past the last heading")
    check(rh.heading_for([(5, "X")], 4) == "", "before every heading")
    check(rh.heading_for([], 0) == "", "no headings")


def test_embed_text_adds_section_line_after_prefix():
    got = rh.embed_text("tar(1) - x\n", "OPTIONS", "-x extract")
    check(got == "tar(1) - x\nSection: OPTIONS\n-x extract", repr(got))


def test_embed_text_without_heading_is_unchanged():
    reembed = _load("reembed_plain", "scripts/reembed.py")
    for prefix in ("tar(1) - x\n", ""):
        got = rh.embed_text(prefix, "", "body")
        check(got == reembed.doc_text(prefix, "body") == f"{prefix}body", repr(got))


def test_verify_accepts_matching_window():
    doc = make_doc()
    n = len(flatten(doc))
    db = make_db(chunks_of(doc, [(0, 20), (14, n)]))
    check(rh.verify(db, {doc["doc_id"]: doc}) == (2, 0), rh.verify(db, {doc["doc_id"]: doc}))


def test_verify_counts_mismatches():
    doc = make_doc()
    good = chunks_of(doc, [(0, 20), (14, 40), (30, 50)])
    changed = [good[1][:4] + ("not the text",)]
    other = [("gone.man", 0, 5, "P\n", "x")]
    db = make_db([good[0]] + changed + other + [good[2]])
    docs = {doc["doc_id"]: doc}
    check(rh.verify(db, docs) == (4, 2), rh.verify(db, docs))
    # sample = the first rows by rowid only: rows 1-2 hold one mismatch
    check(rh.verify(db, docs, sample=2) == (2, 1), rh.verify(db, docs, sample=2))
    check(rh.verify(db, docs, sample=1) == (1, 0), rh.verify(db, docs, sample=1))


def _write_man(dirp, docs):
    p = Path(dirp) / "man.jsonl"
    p.write_text("".join(json.dumps(d) + "\n" for d in docs), encoding="utf-8")
    return p


def test_too_many_mismatches_exit_3_before_embedding():
    doc = make_doc()
    rows = chunks_of(doc, [(0, 20), (14, 40)]) + [("gone.man", 0, 5, "P\n", "x")] * 2
    with tempfile.TemporaryDirectory() as d:
        src, out = Path(d) / "src.db", Path(d) / "out.db"
        db = sqlite3.connect(src)
        db.execute("CREATE TABLE chunks(rowid INTEGER PRIMARY KEY, chunk_id TEXT, doc_id TEXT, "
                   "ord INTEGER, char_start INTEGER, char_end INTEGER, prefix TEXT, text TEXT)")
        db.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
        db.executemany("INSERT INTO chunks(chunk_id,doc_id,ord,char_start,char_end,prefix,text) "
                       "VALUES(?,?,?,?,?,?,?)",
                       [(f"c{i}", r[0], i, r[1], r[2], r[3], r[4]) for i, r in enumerate(rows)])
        db.commit()
        db.close()
        man = _write_man(d, [doc])
        empty_md = Path(d) / "md"
        empty_md.mkdir()
        args = ["--src", str(src), "--out", str(out), "--man", str(man),
                "--md", f"{empty_md}:emacs"]
        # 2 of 4 mismatched: over the default 1%
        check(rh.main(args) == 3, "expected exit 3")
        check(not out.exists(), "--out was created before the check passed")
        # the same index passes when the limit is loosened, and --check-only stops there
        check(rh.main(args + ["--max-mismatch-frac", "0.5", "--check-only"]) == 0,
              "expected exit 0 with --check-only under a loose limit")
        check(not out.exists(), "--check-only must not create --out")


def test_loop_resumes_after_recorded_rowid():
    doc = make_doc()
    rows = chunks_of(doc, [(0, 20), (14, 30), (20, 40), (30, 50), (40, 60)])
    db = make_db(rows)
    docs = {doc["doc_id"]: doc}
    f = Fake()
    db.execute("INSERT INTO meta VALUES(?,?)", (rh.DONE_KEY, "2"))
    n = rh.reembed_headings(db, docs, f.embed, f.write, batch=2, log=lambda *_: None)
    check(n == 3, f"expected 3 chunks after rowid 2, got {n}")
    check([r for r, _ in f.written] == [3, 4, 5], f.written)
    check(db.execute("SELECT value FROM meta WHERE key=?", (rh.DONE_KEY,)).fetchone()[0] == "5",
          "progress not recorded")
    f2 = Fake()
    check(rh.reembed_headings(db, docs, f2.embed, f2.write, log=lambda *_: None) == 0
          and not f2.written, "a finished run must do nothing")


def test_loop_never_writes_chunks_table():
    doc = make_doc()
    rows = chunks_of(doc, [(0, 20), (19, 30)]) + [("gone.man", 0, 5, "P\n", "x")]
    db = make_db(rows)
    before = db.execute("SELECT * FROM chunks ORDER BY rowid").fetchall()
    touched = []

    def auth(action, a1, a2, dbname, src):
        if a1 == "chunks" and action in (sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE,
                                         sqlite3.SQLITE_DELETE):
            touched.append(action)
        return sqlite3.SQLITE_OK

    db.set_authorizer(auth)
    f = Fake()
    rh.reembed_headings(db, {doc["doc_id"]: doc}, f.embed, f.write, log=lambda *_: None)
    db.set_authorizer(None)
    check(not touched, f"chunks table was written: {touched}")
    check(db.execute("SELECT * FROM chunks ORDER BY rowid").fetchall() == before, "rows changed")
    # located chunks get the heading of the section the window starts in; the
    # chunk whose document is missing is embedded as plain document text
    check(f.texts[0].startswith("P\nSection: NAME\n"), f.texts[0])
    check(f.texts[1].startswith("P\nSection: OPTIONS\n"), f.texts[1])
    check(f.texts[2] == "P\nx", f.texts[2])


def test_man_and_md_docs_are_loaded_by_doc_id():
    man_doc = make_doc("tar.man")
    with tempfile.TemporaryDirectory() as d:
        man = _write_man(d, [man_doc])
        md = Path(d) / "md"
        md.mkdir()
        (md / "emacs.md").write_text("# Emacs\n\nintro\n\n## Keys\n\nC-x C-f\n", encoding="utf-8")
        (md / "notes.md").write_text("# Notes\n\nhello\n", encoding="utf-8")
        docs = rh.load_docs(man, [(md, "emacs")])
        check(set(docs) == {"tar.man", "emacs.emacs", "notes.emacs"}, sorted(docs))
        check(docs["tar.man"]["sections"][1]["heading"] == "OPTIONS", "man doc not loaded as is")
        check([s["heading"] for s in docs["emacs.emacs"]["sections"]] == ["Emacs", "Keys"],
              docs["emacs.emacs"]["sections"])
        check(rh.load_docs(None, [(md, "emacs")]).keys() == {"emacs.emacs", "notes.emacs"},
              "man must be optional")


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"pass  {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {name}: {e}")
    sys.exit(1 if failed else 0)
