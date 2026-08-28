#!/usr/bin/env python3
"""Add user material to an index under a domain, reusing the phase 1-2 pipeline.

  .venv/bin/python scripts/ingest.py --db data/index/phase5.db --domain cooking notes.md
  .venv/bin/python scripts/ingest.py --db data/index/phase5.db --domain literature \\
      data/corpus/domains/literature/
  .venv/bin/python scripts/ingest.py --db data/index/phase5.db --domain refs --url https://...

A domain is a namespace, not a separate index. One index keeps a cross-domain
question answerable and keeps the comparison honest: with two indexes the dilution
phase 5 is meant to measure could not happen, so it could not be measured either.
Retrieval then scopes to a domain, and `scripts/eval_domains.py` reports what
scoping is worth.

Resumable: chunks already stored are skipped, so an interrupted ingest can be re-run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import ingest, store  # noqa: E402
from smm.chunk import CHUNK_CHARS, OVERLAP_CHARS, chunk_doc, to_dict  # noqa: E402
from smm.embed import Embedder  # noqa: E402

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".text", ""}


def collect(paths: list[str]) -> list[Path]:
    out = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out += sorted(f for f in path.rglob("*")
                          if f.is_file() and f.suffix.lower() in TEXT_SUFFIXES)
        elif path.is_file():
            out.append(path)
        else:
            print(f"  no such path: {p}", file=sys.stderr)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="files or directories of text/markdown")
    ap.add_argument("--db", default="data/index/phase5.db")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--url", action="append", default=[])
    ap.add_argument("--url-mode", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--chunk-chars", type=int, default=CHUNK_CHARS)
    ap.add_argument("--overlap", type=int, default=OVERLAP_CHARS)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--embed-url", default="http://127.0.0.1:8081")
    ap.add_argument("--dry-run", action="store_true", help="chunk but do not embed")
    args = ap.parse_args()

    if args.domain == "linux":
        print("refusing to ingest into the 'linux' domain: it is the man page corpus, "
              "rebuilt by scripts/build_index.py", file=sys.stderr)
        return 2

    docs = [ingest.from_file(f, args.domain) for f in collect(args.paths)]
    for url in args.url:
        docs.append(ingest.from_url(url, args.domain))
    docs = ingest.dedupe_ids(docs)
    if not docs:
        print("nothing to ingest", file=sys.stderr)
        return 2

    chunks = []
    for d in docs:
        cs = chunk_doc(d, size=args.chunk_chars, overlap=args.overlap)
        for c in cs:
            c_d = to_dict(c)
            c_d["domain"] = args.domain
            chunks.append(c_d)
    print(f"{len(docs)} documents -> {len(chunks)} chunks "
          f"({sum(len(c['text']) for c in chunks)/1e6:.1f} MB) in domain {args.domain!r}")
    if args.dry_run:
        return 0

    emb = Embedder(args.embed_url)
    if not emb.health():
        print(f"no embedder at {args.embed_url}", file=sys.stderr)
        return 2

    out = ROOT / args.db
    if not out.exists():
        print(f"no index at {out}; copy one from data/index/ first", file=sys.stderr)
        return 2
    # store.create() migrates a pre-phase-5 index by adding the domain column,
    # defaulting every existing chunk to 'linux'.
    db = store.connect(out, dim=emb.dim)
    have = {r[0] for r in db.execute("SELECT chunk_id FROM chunks WHERE domain=?",
                                     (args.domain,))}
    todo = [c for c in chunks if c["chunk_id"] not in have]
    if have:
        print(f"resuming: {len(have)} chunks already present, {len(todo)} to add")

    t0, done = time.time(), 0
    for i in range(0, len(todo), args.batch):
        batch = todo[i: i + args.batch]
        vecs = emb.embed_documents([f"{c['prefix']}{c['text']}" for c in batch])
        store.add(db, list(zip(batch, vecs)))
        done += len(batch)
        if sys.stdout.isatty():
            print(f"\r  {done}/{len(todo)}  {done/max(time.time()-t0,1e-9):5.1f}/s",
                  end="", flush=True)
    print(f"\ningested {done} chunks in {time.time()-t0:.0f}s")
    print("domains now:", store.domains(db))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
