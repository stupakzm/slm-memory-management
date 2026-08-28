#!/usr/bin/env python3
"""Build a dense index: flat or structure-aware chunks, sqlite-vec.

  --chunker flat        phase 1/2: a fixed window slid over the flattened page
  --chunker structured  phase 3: one tagged entry per unit, packed to --chunk-chars
  --section-path        phase 3 only: put the heading trail in the embedded prefix

Needs a llama-server running the embedder:
  llama-server -m models/Qwen3-Embedding-0.6B-Q8_0.gguf --embedding \
      --pooling last -c 2048 -ngl 99 --port 8081 -b 8192 -ub 8192

Resumable: re-running skips chunks already stored.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import store  # noqa: E402
from smm.chunk import (CHUNK_CHARS, MAX_ENTRY_CHARS, OVERLAP_CHARS,  # noqa: E402
                       TAGGED_SECTION_RATIO, chunk_corpus, chunk_corpus_structured,
                       to_dict)
from smm.embed import Embedder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus/man.jsonl")
    ap.add_argument("--out", default="data/index/phase1.db")
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--chunk-chars", type=int, default=CHUNK_CHARS)
    ap.add_argument("--overlap", type=int, default=OVERLAP_CHARS)
    ap.add_argument("--no-prefix", action="store_true", help="omit the doc identity line")
    ap.add_argument("--chunker", choices=("flat", "structured"), default="flat")
    ap.add_argument("--max-entry", type=int, default=MAX_ENTRY_CHARS,
                    help="structured: window an entry longer than this")
    ap.add_argument("--tagged-ratio", type=float, default=TAGGED_SECTION_RATIO,
                    help="structured: entry-pack a section only when this fraction of "
                         "its text sits in tagged paragraphs; 0 entry-packs every "
                         "section, which reproduces the phase3-entry/path indexes")
    ap.add_argument("--section-path", action="store_true",
                    help="structured: add the heading trail to the embedded prefix")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="stop after N chunks (smoke test)")
    args = ap.parse_args()

    emb = Embedder(args.url)
    if not emb.health():
        print(f"no llama-server at {args.url} - start the embedder first", file=sys.stderr)
        return 2
    dim = emb.dim
    print(f"embedder up, dim={dim}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    db = store.connect(out, dim=dim)
    already = {r[0] for r in db.execute("SELECT chunk_id FROM chunks").fetchall()}
    if already:
        print(f"resuming: {len(already)} chunks already indexed")

    store.set_meta(
        db, dim=dim, chunk_chars=args.chunk_chars, overlap=args.overlap,
        prefix=not args.no_prefix, corpus=args.corpus,
        chunker=args.chunker, section_path=args.section_path,
        max_entry=args.max_entry if args.chunker == "structured" else "",
        tagged_ratio=args.tagged_ratio if args.chunker == "structured" else "",
        phase="3-structured" if args.chunker == "structured" else "1-flat-dense",
    )

    if args.chunker == "structured":
        chunks = chunk_corpus_structured(
            ROOT / args.corpus, size=args.chunk_chars, overlap=args.overlap,
            max_entry=args.max_entry, with_prefix=not args.no_prefix,
            with_path=args.section_path, tagged_ratio=args.tagged_ratio)
    else:
        chunks = chunk_corpus(ROOT / args.corpus, size=args.chunk_chars,
                              overlap=args.overlap, with_prefix=not args.no_prefix)
    batch, n, t0 = [], len(already), time.time()
    total_new = 0
    for c in chunks:
        if c.chunk_id in already:
            continue
        batch.append(c)
        if len(batch) >= args.batch:
            vecs = emb.embed_documents([b.embed_text for b in batch])
            store.add(db, list(zip((to_dict(b) for b in batch), vecs)))
            total_new += len(batch)
            n += len(batch)
            rate = total_new / max(time.time() - t0, 1e-9)
            print(f"\r  {n} chunks  {rate:6.1f}/s", end="", flush=True)
            batch = []
            if args.limit and total_new >= args.limit:
                break
    if batch and not (args.limit and total_new >= args.limit):
        vecs = emb.embed_documents([b.embed_text for b in batch])
        store.add(db, list(zip((to_dict(b) for b in batch), vecs)))
        n += len(batch)
    print(f"\nindexed {store.count(db)} chunks in {time.time()-t0:.0f}s -> {out}")
    print(f"db size: {out.stat().st_size/1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
