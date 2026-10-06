"""Tests for src/smm/info_index.py and scripts/build_ixvec.py (Emacs index entries as
question vectors).

Hermetic (blk_test_env_constraints): stdlib only, sqlite_vec stubbed in sys.modules,
synthetic info text in the real shape, a synthetic markdown document, plain sqlite3
databases in a tempdir, a fake embedder and a stand-in for the qvec store functions.
No torch, no servers, no data/ or models/.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
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

from smm import info_index as ii  # noqa: E402
from smm import store as real_store  # noqa: E402
from smm.chunk import flatten  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ix = _load("build_ixvec", "scripts/build_ixvec.py")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


HEADER = ("\nFile: emacs.info,  Node: {node},  Prev: Variable Index,  Up: Top\n\n{node}\n"
          "*************\n\n\x00\x08[index\x00\x08]\n* Menu:\n\n")

REAL_ENTRIES = (
    "* _emacs init file, MS-Windows:          Windows HOME.       (line   43)\n"
    "* ? in display:                          International Chars.\n"
    "                                                             (line   18)\n"
    '* "adb logcat":                          Android Startup.    (line   15)\n'
    "* .emacs file:                           Init File.          (line    6)\n"
)


def info(entries: str, node: str = "Concept Index", extra: str = "") -> str:
    return ("Preamble line\n\x1f\nFile: emacs.info,  Node: Top,  Next: Intro\n\nTop body\n"
            "\x1f" + HEADER.format(node=node) + entries + "\x1f" + extra +
            "\x1f\nTag Table:\nNode: Top\x7f100\n\x1f\nEnd Tag Table\n")


# -------------------------------------------------------------- info_index parsing

def test_parses_simple_entry():
    got = ii.parse_index(info(REAL_ENTRIES))
    check({"entry": ".emacs file", "node": "Init File", "line": 6} in got, got)
    check({"entry": "_emacs init file, MS-Windows", "node": "Windows HOME", "line": 43} in got,
          got)
    check([e["entry"] for e in got][0] == "_emacs init file, MS-Windows", "file order lost")
    check(all(isinstance(e["line"], int) for e in got), "line must be an int")


def test_parses_wrapped_entry_with_node_on_next_line():
    got = ii.parse_index(info(REAL_ENTRIES))
    check({"entry": "? in display", "node": "International Chars", "line": 18} in got, got)
    long = ("* /content/by-authority-named directory, android: Android Startup.\n"
            "                                                             (line   71)\n")
    got = ii.parse_index(info(long))
    check(got == [{"entry": "/content/by-authority-named directory, android",
                   "node": "Android Startup", "line": 71}], got)


def test_entry_may_contain_colon_and_punctuation():
    text = ("* a: b, c; d?:                           Two Colons.         (line    3)\n"
            '* "adb logcat":                          Android Startup.    (line   15)\n'
            "* ( in leftmost column:                  Left Margin Paren.  (line    6)\n")
    got = ii.parse_index(info(text))
    check([(e["entry"], e["node"], e["line"]) for e in got] == [
        ("a: b, c; d?", "Two Colons", 3),
        ('"adb logcat"', "Android Startup", 15),
        ("( in leftmost column", "Left Margin Paren", 6)], got)


def test_ignores_menu_header_and_non_entry_lines():
    text = ("\n* Not an entry because it has no line number: Somewhere.\n"
            "Some prose line in the index node.\n"
            "* .emacs file:                           Init File.          (line    6)\n"
            "\n  an indented stray line\n")
    got = ii.parse_index(info(text))
    check(got == [{"entry": ".emacs file", "node": "Init File", "line": 6}], got)
    check("Menu" not in " ".join(e["entry"] for e in got), got)


def test_only_the_named_index_node_is_read():
    other = HEADER.format(node="Key Index") + \
        "* C-x C-f:                              Visiting.           (line    9)\n"
    text = info(REAL_ENTRIES, extra=other)
    concept = ii.parse_index(text)
    check(all(e["entry"] != "C-x C-f" for e in concept), concept)
    keys = ii.parse_index(text, "Key Index")
    check(keys == [{"entry": "C-x C-f", "node": "Visiting", "line": 9}], keys)
    check(ii.parse_index(text, "No Such Index") == [], "unknown node must give nothing")


def test_duplicate_entries_are_dropped():
    pad = ".emacs file:                           Init File.          "
    text = f"* {pad}(line    6)\n* {pad}(line    6)\n* {pad}(line    7)\n"
    got = ii.parse_index(info(text))
    check([(e["entry"], e["line"]) for e in got] == [(".emacs file", 6), (".emacs file", 7)],
          got)


# ------------------------------------------------------------- chunk_for / find_section

def test_chunk_for_picks_window_containing_offset():
    chunks = [("a", 0, 50), ("b", 40, 100), ("c", 90, 150)]
    check(ii.chunk_for(chunks, 10)[0] == "a", "only a contains 10")
    check(ii.chunk_for(chunks, 45)[0] == "b", "overlap: the larger char_start wins")
    check(ii.chunk_for(chunks, 50)[0] == "b", "char_end is exclusive")
    check(ii.chunk_for(chunks, 95)[0] == "c", "overlap with c")
    check(ii.chunk_for(chunks, 149)[0] == "c", "last character of c")


def test_chunk_for_falls_back_to_next_window():
    chunks = [("a", 0, 10), ("b", 30, 40), ("c", 20, 25)]
    check(ii.chunk_for(chunks, 15)[0] == "c", "gap: smallest char_start after the offset")
    check(ii.chunk_for(chunks, 26)[0] == "b", "gap before b")
    check(ii.chunk_for(chunks, 40) is None, "past the last window")
    check(ii.chunk_for([], 0) is None, "no chunks")


def test_node_matched_ignoring_case_and_whitespace():
    offsets = [(0, "Intro"), (10, "Init  File"), (40, "Keys")]
    check(ii.find_section(offsets, "init file") == (10, "Init  File"), "case/whitespace")
    check(ii.find_section(offsets, "  INIT\n FILE ") == (10, "Init  File"), "runs of space")
    check(ii.find_section(offsets, "Keys") == (40, "Keys"), "exact")
    check(ii.find_section(offsets, "Init") is None, "a prefix is not a match")
    check(ii.find_section([], "Keys") is None, "no offsets")


# ------------------------------------------------------------------- build_ixvec

DOC_SECTIONS = [("Emacs", "Emacs is an editor."),
                ("Init File", "\n".join(f"init line {i}" for i in range(1, 9))),
                ("Keys", "key line a\nkey line b")]


def make_env(d: Path, qvec_in_src: bool = False):
    """(argv base, doc, chunk ids): a markdown doc, an index of 4 chunks over it."""
    md = d / "md"
    md.mkdir()
    (md / "emacs.md").write_text("".join(f"## {h}\n{t}\n\n" for h, t in DOC_SECTIONS))
    docs = ix.load_docs(None, [(md, "emacs")])
    doc = docs["emacs.emacs"]
    flat = flatten(doc)
    offs = ix.heading_offsets(doc)
    init, keys = offs[1][0], offs[2][0]
    mid = init + len("Init File") + 1 + len("init line 1\ninit line 2\ninit line 3\n")
    windows = [("c0", 0, init), ("c1", init, mid), ("c2", mid, keys), ("c3", keys, len(flat))]
    src = d / "src.db"
    db = sqlite3.connect(src)
    db.execute("CREATE TABLE chunks(chunk_id TEXT, doc_id TEXT, ord INT, char_start INT, "
               "char_end INT, prefix TEXT, text TEXT, domain TEXT)")
    db.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
    db.executemany("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)",
                   [(c, "emacs.emacs", i, a, b, "", flat[a:b], "emacs")
                    for i, (c, a, b) in enumerate(windows)])
    db.execute("CREATE TABLE other(x)")
    db.execute("INSERT INTO chunks VALUES('o0','other.emacs',0,0,10000,'','x','emacs')")
    if qvec_in_src:
        db.execute("CREATE TABLE chunk_questions(rowid INTEGER PRIMARY KEY, chunk_id, question)")
        db.execute("CREATE TABLE qvec(embedding)")
    db.commit()
    db.close()
    info_path = d / "emacs.info"
    return src, md, info_path


class StubStore:
    """The qvec store functions without sqlite-vec: plain tables, same call shapes."""
    has_qvec = staticmethod(real_store.has_qvec)
    set_meta = staticmethod(real_store.set_meta)

    @staticmethod
    def connect(path):
        return sqlite3.connect(path)

    @staticmethod
    def create_qvec(db, dim):
        db.execute("CREATE TABLE IF NOT EXISTS chunk_questions("
                   "rowid INTEGER PRIMARY KEY, chunk_id TEXT NOT NULL, question TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS qvec(rowid INTEGER PRIMARY KEY, embedding TEXT)")
        db.commit()

    @staticmethod
    def add_questions(db, rows):
        for cid, q, vec in rows:
            cur = db.execute("INSERT INTO chunk_questions(chunk_id,question) VALUES(?,?)",
                             (cid, q))
            db.execute("INSERT INTO qvec(rowid,embedding) VALUES(?,?)",
                       (cur.lastrowid, repr(list(vec))))
        db.commit()


class Fake:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0] for t in texts]


def run(args, fake):
    """main() with the store stubbed; (exit code, stdout, stderr)."""
    real = ix.store
    ix.store = StubStore
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = ix.main(args, embed_fn=fake.embed)
    finally:
        ix.store = real
    return code, out.getvalue(), err.getvalue()


ENTRIES = (
    "* init things:                           Init File.          (line    2)\n"
    "* late init things:                      Init File.\n"
    "                                                             (line    6)\n"
    "* the keys:                              Keys.               (line    2)\n"
    "* nowhere to go:                         Missing Node.       (line    2)\n"
)


def args_for(d, src, md, info_path, *more):
    return ["--src", str(src), "--out", str(d / "out.db"), "--info", str(info_path),
            "--md", f"{md}:emacs", *more]


def test_dry_run_reports_coverage_and_writes_nothing():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        src, md, info_path = make_env(d)
        info_path.write_text(info(ENTRIES))
        fake = Fake()
        code, out, _ = run(args_for(d, src, md, info_path, "--dry-run", "--min-coverage", "0.5"),
                           fake)
        check(code == 0, f"exit {code}")
        check("entries 4, mapped 3, unmapped 1, coverage 0.750" in out, out)
        check(not (d / "out.db").exists(), "--dry-run created --out")
        check(not fake.calls, "--dry-run called the embedder")


def test_low_coverage_exits_3():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        src, md, info_path = make_env(d)
        info_path.write_text(info(ENTRIES))
        fake = Fake()
        code, out, _ = run(args_for(d, src, md, info_path), fake)  # 0.75 < default 0.9
        check(code == 3, f"exit {code}")
        check("coverage 0.750" in out, out)
        check(not (d / "out.db").exists(), "--out was created before the coverage check")
        check(not fake.calls, "the embedder was called before the coverage check")


def test_refuses_src_that_already_has_question_vectors():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        src, md, info_path = make_env(d, qvec_in_src=True)
        info_path.write_text(info(ENTRIES))
        fake = Fake()
        code, _, err = run(args_for(d, src, md, info_path, "--min-coverage", "0"), fake)
        check(code == 2, f"exit {code}")
        check("question vectors" in err, err)
        check(not (d / "out.db").exists() and not fake.calls, "wrote or embedded anyway")


def test_embeds_entry_text_not_node_name():
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        src, md, info_path = make_env(d)
        info_path.write_text(info(ENTRIES))
        fake = Fake()
        code, out, _ = run(args_for(d, src, md, info_path, "--min-coverage", "0.5",
                                    "--batch", "2"), fake)
        check(code == 0, f"exit {code}: {out}")
        embedded = [t for call in fake.calls for t in call]
        check(embedded == ["init things", "late init things", "the keys"], embedded)
        check(len(fake.calls) == 2, "batches of 2")
        db = sqlite3.connect(d / "out.db")
        rows = db.execute("SELECT chunk_id, question FROM chunk_questions ORDER BY rowid")\
            .fetchall()
        # line 2 -> start of Init File's text (c1); line 6 -> past the 3rd line (c2)
        check(rows == [("c1", "init things"), ("c2", "late init things"), ("c3", "the keys")],
              rows)
        check(db.execute("SELECT count(*) FROM qvec").fetchone()[0] == 3, "one vector per row")
        meta = dict(db.execute("SELECT key, value FROM meta"))
        check(meta.get("qvec_domain") == "emacs" and meta.get("qvec_prompt") == "info-index"
              and meta.get("qvec_source") == "emacs.info" and meta.get("qvec_per_chunk") == "0",
              meta)
        db.close()
        check(not ix.store.has_qvec(sqlite3.connect(src)), "the source index was written")
        # a re-run resumes: nothing left to embed
        fake2 = Fake()
        code, _, _ = run(args_for(d, src, md, info_path, "--min-coverage", "0.5"), fake2)
        check(code == 0 and not fake2.calls, f"re-run embedded {fake2.calls}")


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
