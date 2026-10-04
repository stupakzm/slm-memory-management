#!/usr/bin/env python3
"""Phase 15 R13: re-embed every chunk vector of an index with the running embedder.

Copies --src to --out (sqlite backup API; the source is only ever opened
read-only), then replaces each vec_chunks vector, in rowid order, with the
embedding of the chunk's document text (prefix + text, as Chunk.embed_text).
The chunks table is never written. Progress is committed per batch under the
meta key reembed_done_rowid, so a re-run resumes after the last finished batch.

  .venv/bin/python scripts/reembed.py --src data/index/phase11.db --out data/index/phase11-ft.db \\
      --model-name qwen3-embed-ft-q8_0

Needs the embedder server (scripts/servers.sh start embedder, SMM_EMBED_MODEL
choosing the weights) on --url.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DONE_KEY = "reembed_done_rowid"


def doc_text(prefix: str, text: str) -> str:
    """Same string smm.chunk.Chunk.embed_text builds."""
    return f"{prefix}{text}" if prefix else text


def prepare_out(src: Path, out: Path) -> None:
    """Copy src to out with the backup API (src read-only) unless out exists."""
    if src.resolve() == out.resolve():
        raise SystemExit("--out must not be --src: the source index is never written")
    if out.exists():
        return
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    d = sqlite3.connect(out)
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()


def _set_meta(db, **kw) -> None:
    db.executemany("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
                   [(k, str(v)) for k, v in kw.items()])


def reembed(db, embed_fn, write_vec_fn, batch: int = 64, log=print) -> int:
    """Re-embed chunks after the recorded progress; returns how many were done.

    embed_fn(list[str]) -> list[vector]; write_vec_fn(db, rowid, vector) replaces
    one vector. Commits (vectors + progress) after every batch.
    """
    row = db.execute("SELECT value FROM meta WHERE key=?", (DONE_KEY,)).fetchone()
    done_rowid = int(row[0]) if row else 0
    total = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    todo = db.execute("SELECT count(*) FROM chunks WHERE rowid>?", (done_rowid,)).fetchone()[0]
    log(f"{todo} of {total} chunks to embed (resuming after rowid {done_rowid})")
    n, t0 = 0, time.time()
    while True:
        rows = db.execute("SELECT rowid, prefix, text FROM chunks WHERE rowid>? "
                          "ORDER BY rowid LIMIT ?", (done_rowid, batch)).fetchall()
        if not rows:
            break
        vecs = embed_fn([doc_text(p, t) for _, p, t in rows])
        if len(vecs) != len(rows):
            raise RuntimeError(f"embedder returned {len(vecs)} vectors for {len(rows)} texts")
        for (rid, _, _), v in zip(rows, vecs):
            write_vec_fn(db, rid, v)
        done_rowid = rows[-1][0]
        _set_meta(db, **{DONE_KEY: done_rowid})
        db.commit()
        n += len(rows)
        log(f"  {n}/{todo}  {n / max(time.time() - t0, 1e-9):.1f} chunks/s")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--model-name", default="unknown")
    a = ap.parse_args()
    if a.src.resolve() == a.out.resolve():
        print("--out must not be --src: the source index is never written", file=sys.stderr)
        return 2

    from smm import store
    from smm.embed import Embedder

    prepare_out(a.src, a.out)
    emb = Embedder(a.url)
    db = store.connect(a.out)

    def write_vec(d, rid, vec):
        # vec0 rejects INSERT OR REPLACE on an existing rowid: delete first.
        d.execute("DELETE FROM vec_chunks WHERE rowid=?", (rid,))
        d.execute("INSERT INTO vec_chunks(rowid,embedding) VALUES(?,?)", (rid, store.pack(vec)))

    try:
        reembed(db, emb.embed_documents, write_vec, a.batch)
        _set_meta(db, reembed_complete=1, embedder=a.model_name)
        db.commit()
    finally:
        db.close()
    print(f"done -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
