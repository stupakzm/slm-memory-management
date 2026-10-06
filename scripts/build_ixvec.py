#!/usr/bin/env python3
"""Question vectors from the Emacs manual's own index entries.

scripts/extract_info.py drops every node whose name ends in "Index", which throws away
thousands of human-written entries that map a wording to a section
(`* .emacs file:   Init File.   (line 6)`). This parses those entries
(src/smm/info_index.py), points each at the chunk its `(line N)` falls in, and embeds
the ENTRY TEXT ALONE (never the node name or the heading) into a copy of --src, in the
same qvec tables scripts/build_qvec.py writes, so Retriever(question_vectors=M) reads
them unchanged.

  .venv/bin/python scripts/build_ixvec.py --out data/index/phase11-ix.db --dry-run
  .venv/bin/python scripts/build_ixvec.py --out data/index/phase11-ix.db

`(line N)` counts from the info node header line; the markdown section has a heading
line and a blank line instead, so the target offset is an approximation:
heading offset + len(heading) + 1 + the characters of the first max(N-2, 0) lines of
the section text, clamped to the section.

--dry-run prints `entries n, mapped m, unmapped u, coverage c` and stops (exit 0). A
coverage below --min-coverage exits 3 before --out or the embedder is touched. A --src
that already has question vectors is refused (exit 2). Re-running resumes: (chunk_id,
entry) pairs already in --out are skipped. Needs the embedder server on --url.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import info_index, store  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_rh = _load("reembed_headings", "scripts/reembed_headings.py")
_bq = _load("build_qvec", "scripts/build_qvec.py")
heading_offsets = _rh.heading_offsets
load_docs = _rh.load_docs
prepare_out = _bq.prepare_out


def target_offset(doc: dict, offsets: list[tuple[int, str]], section: tuple[int, str],
                  line: int) -> int:
    """Offset in flatten(doc) approximating `(line N)` of the entry's node."""
    sec = doc["sections"][offsets.index(section)]
    text = sec["text"]
    skipped = len("".join(text.splitlines(keepends=True)[:max(line - 2, 0)]))
    return section[0] + len(section[1]) + 1 + min(skipped, len(text))


def map_entries(entries: list[dict], doc: dict, chunks: list[tuple]) -> tuple[list, int]:
    """([(chunk_id, entry text)] for the mapped entries, number unmapped)."""
    offsets = heading_offsets(doc)
    pairs, unmapped = [], 0
    for e in entries:
        sec = info_index.find_section(offsets, e["node"])
        chunk = (info_index.chunk_for(chunks, target_offset(doc, offsets, sec, e["line"]))
                 if sec is not None else None)
        if chunk is None:
            unmapped += 1
        else:
            pairs.append((chunk[0], e["entry"]))
    return pairs, unmapped


def read_info(path: Path) -> str:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    return Path(path).read_text(encoding="utf-8", errors="replace")


def embed_pairs(db, pairs: list[tuple[str, str]], embed_fn, batch: int = 64, log=print) -> int:
    """Embed each (chunk_id, entry) not yet in `db`; the entry text alone is embedded."""
    have = set()
    if store.has_qvec(db):
        have = set(db.execute("SELECT chunk_id, question FROM chunk_questions"))
    seen, todo = set(have), []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            todo.append(p)
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        vecs = embed_fn([q for _, q in part])
        if len(vecs) != len(part):
            raise RuntimeError(f"embedder returned {len(vecs)} vectors for {len(part)} texts")
        store.create_qvec(db, len(vecs[0]))
        store.add_questions(db, [(c, q, v) for (c, q), v in zip(part, vecs)])
        log(f"  embedded {min(i + batch, len(todo))}/{len(todo)}")
    return len(todo)


def main(argv: list[str] | None = None, embed_fn=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=Path("data/index/phase11.db"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--info", type=Path, default=Path("/usr/share/info/emacs.info.gz"))
    ap.add_argument("--index-node", action="append", default=None,
                    help='index node to read (repeatable; default "Concept Index")')
    ap.add_argument("--md", type=_rh._md_arg, default=(Path("data/corpus/domains/emacs"), "emacs"),
                    metavar="DIR:DOMAIN")
    ap.add_argument("--doc-id", default="emacs.emacs")
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--min-coverage", type=float, default=0.9)
    ap.add_argument("--limit", type=int, default=0, help="only the first N entries")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.src.resolve() == a.out.resolve():
        print("--out must not be --src: the source index is never written", file=sys.stderr)
        return 2

    src = sqlite3.connect(f"file:{a.src}?mode=ro", uri=True)
    try:
        if store.has_qvec(src):
            print(f"{a.src} already has question vectors; refusing to extend it",
                  file=sys.stderr)
            return 2
        chunks = src.execute("SELECT chunk_id, char_start, char_end FROM chunks "
                             "WHERE doc_id=? ORDER BY char_start", (a.doc_id,)).fetchall()
    finally:
        src.close()

    text = read_info(a.info)
    entries = []
    for node in a.index_node or ["Concept Index"]:
        entries += info_index.parse_index(text, node)
    if a.limit:
        entries = entries[:a.limit]
    docs = load_docs(None, [a.md])
    if a.doc_id not in docs:
        print(f"document {a.doc_id} not found under {a.md[0]}", file=sys.stderr)
        return 2
    pairs, unmapped = map_entries(entries, docs[a.doc_id], chunks)
    n = len(entries)
    coverage = len(pairs) / n if n else 0.0
    print(f"entries {n}, mapped {len(pairs)}, unmapped {unmapped}, coverage {coverage:.3f}")
    if coverage < a.min_coverage:
        print(f"coverage below --min-coverage {a.min_coverage}", file=sys.stderr)
        return 3
    if a.dry_run:
        return 0

    if embed_fn is None:
        from smm.embed import Embedder
        emb = Embedder(a.url)
        if not emb.health():
            print("embedder not running: ./scripts/servers.sh start embedder", file=sys.stderr)
            return 2
        embed_fn = emb.embed_documents
    prepare_out(a.src, a.out)
    db = store.connect(a.out)
    try:
        embed_pairs(db, pairs, embed_fn, a.batch)
        store.set_meta(db, qvec_domain=a.md[1], qvec_per_chunk=0, qvec_prompt="info-index",
                       qvec_source=a.info.name)
    finally:
        db.close()
    print(f"done -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
