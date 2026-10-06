#!/usr/bin/env python3
"""Re-embed every chunk of an index with its section heading in the embedded text.

The current index is flat: chunk rows have no heading. This copies --src to --out
(sqlite backup API; the source is only ever opened read-only) and replaces each
vec_chunks vector with the embedding of

    <document prefix>Section: <heading>\\n<chunk text>

where <heading> is the heading of the section the chunk's window starts in. Chunk
ids, boundaries and text are untouched (the chunks table is never written), and
queries are embedded as before - the embedded chunk text is the only variable.

Before anything is embedded each chunk's text is located in its source document
(flatten(doc)[char_start:char_end].strip() == text); if more than
--max-mismatch-frac of the chunks cannot be located the run stops with exit 3
before touching --out or the embedder. A chunk that cannot be located is embedded
as the plain document text, as scripts/reembed.py would.

  .venv/bin/python scripts/reembed_headings.py --src data/index/phase11.db \\
      --out data/index/phase11-headings.db --model-name qwen3-embed-q8_0

  --check-only  stops after the verification (exit 0 when it passes).

Needs the embedder server on --url. Progress is committed per batch under the meta
key reembed_headings_done_rowid, so a re-run resumes after the last finished batch.
"""

from __future__ import annotations

import argparse
import bisect
import importlib.util
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DONE_KEY = "reembed_headings_done_rowid"
COMPLETE_KEY = "reembed_headings_complete"


def _load_reembed():
    spec = importlib.util.spec_from_file_location("reembed", ROOT / "scripts" / "reembed.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_reembed = _load_reembed()
doc_text = _reembed.doc_text
prepare_out = _reembed.prepare_out
_set_meta = _reembed._set_meta


def heading_offsets(doc: dict) -> list[tuple[int, str]]:
    """(offset in flatten(doc), heading) for each section's heading line."""
    out, pos = [], 0
    for sec in doc["sections"]:
        out.append((pos, sec["heading"]))
        pos += len(sec["heading"]) + 1
        if sec["text"]:
            pos += len(sec["text"]) + 1
    return out


def heading_for(offsets: list[tuple[int, str]], char_start: int) -> str:
    """Heading of the last entry whose offset <= char_start; '' if none."""
    i = bisect.bisect_right([o for o, _ in offsets], char_start)
    return offsets[i - 1][1] if i else ""


def embed_text(prefix: str, heading: str, text: str) -> str:
    if not heading:
        return doc_text(prefix, text)
    return f"{prefix}Section: {heading}\n{text}"


def load_docs(man_path: Path | None, md_dirs: list[tuple[Path, str]]) -> dict[str, dict]:
    """doc_id -> doc from the man jsonl and from each (dir, domain) markdown dir."""
    from smm import ingest

    docs: dict[str, dict] = {}
    if man_path is not None:
        with Path(man_path).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    docs[d["doc_id"]] = d
    for d_dir, domain in md_dirs:
        batch = [ingest.from_file(p, domain) for p in sorted(Path(d_dir).glob("*.md"))]
        for d in ingest.dedupe_ids(batch):
            docs[d["doc_id"]] = d
    return docs


class _Flat:
    """flatten() of the document last asked for (chunks arrive grouped by doc)."""

    def __init__(self, docs: dict[str, dict]):
        self.docs, self._id, self._flat = docs, None, ""

    def get(self, doc_id: str) -> str | None:
        if doc_id not in self.docs:
            return None
        if doc_id != self._id:
            from smm.chunk import flatten
            self._id, self._flat = doc_id, flatten(self.docs[doc_id])
        return self._flat

    def located(self, doc_id: str, a: int, b: int, text: str) -> bool:
        flat = self.get(doc_id)
        return flat is not None and flat[a:b].strip() == text


def verify(db, docs: dict[str, dict], sample: int | None = None) -> tuple[int, int]:
    """(checked, mismatched) over all chunks, or the first `sample` by rowid."""
    sql = "SELECT doc_id, char_start, char_end, text FROM chunks ORDER BY rowid"
    rows = db.execute(sql + (" LIMIT ?" if sample is not None else ""),
                      (sample,) if sample is not None else ()).fetchall()
    flat = _Flat(docs)
    bad = sum(0 if flat.located(*r) else 1 for r in rows)
    return len(rows), bad


def reembed_headings(db, docs, embed_fn, write_vec_fn, batch: int = 64, log=print) -> int:
    """Re-embed chunks after the recorded progress with heading-bearing text."""
    row = db.execute("SELECT value FROM meta WHERE key=?", (DONE_KEY,)).fetchone()
    done_rowid = int(row[0]) if row else 0
    total = db.execute("SELECT count(*) FROM chunks").fetchone()[0]
    todo = db.execute("SELECT count(*) FROM chunks WHERE rowid>?", (done_rowid,)).fetchone()[0]
    log(f"{todo} of {total} chunks to embed (resuming after rowid {done_rowid})")
    flat, offsets = _Flat(docs), {}
    n, t0 = 0, time.time()
    while True:
        rows = db.execute("SELECT rowid, doc_id, char_start, char_end, prefix, text FROM chunks "
                          "WHERE rowid>? ORDER BY rowid LIMIT ?", (done_rowid, batch)).fetchall()
        if not rows:
            break
        texts = []
        for _, doc_id, a, b, prefix, text in rows:
            if flat.located(doc_id, a, b, text):
                if doc_id not in offsets:
                    offsets = {doc_id: heading_offsets(docs[doc_id])}
                texts.append(embed_text(prefix, heading_for(offsets[doc_id], a), text))
            else:
                texts.append(doc_text(prefix, text))
        vecs = embed_fn(texts)
        if len(vecs) != len(rows):
            raise RuntimeError(f"embedder returned {len(vecs)} vectors for {len(rows)} texts")
        for r, v in zip(rows, vecs):
            write_vec_fn(db, r[0], v)
        done_rowid = rows[-1][0]
        _set_meta(db, **{DONE_KEY: done_rowid})
        db.commit()
        n += len(rows)
        log(f"  {n}/{todo}  {n / max(time.time() - t0, 1e-9):.1f} chunks/s")
    return n


def _md_arg(s: str) -> tuple[Path, str]:
    d, sep, dom = s.rpartition(":")
    if not sep or not d or not dom:
        raise argparse.ArgumentTypeError(f"expected DIR:DOMAIN, got {s!r}")
    return Path(d), dom


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--man", type=Path, default=Path("data/corpus/man.jsonl"))
    ap.add_argument("--md", type=_md_arg, action="append", metavar="DIR:DOMAIN",
                    help="markdown directory and its domain (repeatable; "
                         "default data/corpus/domains/emacs:emacs)")
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--model-name", default="unknown")
    ap.add_argument("--max-mismatch-frac", type=float, default=0.01)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args(argv)
    if a.src.resolve() == a.out.resolve():
        print("--out must not be --src: the source index is never written", file=sys.stderr)
        return 2
    md = a.md if a.md is not None else [(Path("data/corpus/domains/emacs"), "emacs")]

    docs = load_docs(a.man if a.man and a.man.exists() else None, md)
    src = sqlite3.connect(f"file:{a.src}?mode=ro", uri=True)
    try:
        n, m = verify(src, docs)
    finally:
        src.close()
    print(f"verified {n}, mismatched {m}")
    if n == 0 or m / n > a.max_mismatch_frac:
        print(f"too many chunks cannot be located in their documents "
              f"(limit {a.max_mismatch_frac:.2%})", file=sys.stderr)
        return 3
    if a.check_only:
        return 0

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
        reembed_headings(db, docs, emb.embed_documents, write_vec, a.batch)
        _set_meta(db, **{COMPLETE_KEY: 1, "embedder": a.model_name})
        db.commit()
    finally:
        db.close()
    print(f"done -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
