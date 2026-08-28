#!/usr/bin/env python3
"""Chunking cost and chunking ceiling, for any chunker config, before indexing.

The ceiling is the number that keeps every later result honest: the fraction of
answerable questions whose complete answer lands inside at least one chunk. If it is
below 100%, some of the recall gap is the chunker's fault rather than the
retriever's, and no amount of reranking can reach those questions.

Phase 1 reported 100% for flat windows in prose. Phase 3 moves every boundary in the
corpus, so it has to be re-earned rather than inherited - a finer cut is exactly the
kind of change that severs an answer spanning two entries.

  .venv/bin/python scripts/chunk_stats.py
  .venv/bin/python scripts/chunk_stats.py --chunk-chars 600 --max-entry 900
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm.chunk import (CHUNK_CHARS, MAX_ENTRY_CHARS, OVERLAP_CHARS,  # noqa: E402
                       TAGGED_SECTION_RATIO, chunk_doc, chunk_doc_structured)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus/man.jsonl")
    ap.add_argument("--eval", default="data/eval/questions.jsonl")
    ap.add_argument("--chunk-chars", type=int, default=CHUNK_CHARS)
    ap.add_argument("--overlap", type=int, default=OVERLAP_CHARS)
    ap.add_argument("--max-entry", type=int, default=MAX_ENTRY_CHARS)
    ap.add_argument("--tagged-ratio", type=float, default=TAGGED_SECTION_RATIO)
    ap.add_argument("--verbose", action="store_true", help="name the questions lost")
    args = ap.parse_args()

    docs = [json.loads(l) for l in (ROOT / args.corpus).open(encoding="utf-8")]
    by_id = {d["doc_id"]: d for d in docs}
    rows = [json.loads(l) for l in (ROOT / args.eval).open(encoding="utf-8")]
    ans = [r for r in rows if r["kind"] == "answerable"]

    chunkers = {
        f"flat {args.chunk_chars}":
            lambda d: chunk_doc(d, size=args.chunk_chars, overlap=args.overlap),
        f"structured {args.chunk_chars}":
            lambda d: chunk_doc_structured(d, size=args.chunk_chars, overlap=args.overlap,
                                           max_entry=args.max_entry,
                                           tagged_ratio=args.tagged_ratio),
        f"structured, every section":
            lambda d: chunk_doc_structured(d, size=args.chunk_chars, overlap=args.overlap,
                                           max_entry=args.max_entry, tagged_ratio=0.0),
    }

    print(f"{len(docs)} documents, {len(ans)} answerable questions\n")
    print(f"{'chunker':22} {'chunks':>8} {'median':>7} {'p90':>6} {'ceiling':>8}  lost")
    for name, f in chunkers.items():
        sizes, n = [], 0
        for d in docs:
            cs = f(d)
            n += len(cs)
            sizes.extend(len(c.text) for c in cs)
        lost = [r["qid"] for r in ans
                if not any(all(t in c.text for t in r["answer_contains"])
                           for c in f(by_id[r["doc"]]))]
        print(f"{name:22} {n:8} {statistics.median(sizes):7.0f} "
              f"{sorted(sizes)[int(len(sizes) * .9)]:6} {1 - len(lost)/len(ans):8.1%}  {len(lost)}")
        if lost and args.verbose:
            for qid in lost:
                r = next(x for x in ans if x["qid"] == qid)
                print(f"{'':22}   {qid:6} {r['doc']:18} {r['answer_contains']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
