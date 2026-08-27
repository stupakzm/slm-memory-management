#!/usr/bin/env python3
"""Ask the corpus a question. Phase 1: flat dense retrieval, no gate, no reranker.

  ./scripts/ask.py "how do I exclude files listed in a text file from a tar archive"
  ./scripts/ask.py --retrieve-only -k 10 "watch a log file as it grows"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import store  # noqa: E402
from smm.embed import Embedder  # noqa: E402
from smm.generate import Generator  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--db", default="data/index/phase1.db")
    ap.add_argument("--retrieve-only", action="store_true")
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args()
    question = " ".join(args.question)

    emb = Embedder()
    if not emb.health():
        print("embedder not running: ./scripts/servers.sh start embedder", file=sys.stderr)
        return 2
    db = store.connect(ROOT / args.db)
    hits = store.search(db, emb.embed_query(question), k=args.k)

    if args.retrieve_only or args.show_context:
        for i, h in enumerate(hits, 1):
            head = h["text"].splitlines()[0][:70] if h["text"] else ""
            print(f"[{i}] {h['score']:.4f}  {h['chunk_id']:28} {head}")
        if args.retrieve_only:
            return 0
        print()

    gen = Generator()
    if not gen.health():
        print("generator not running: ./scripts/servers.sh start generator", file=sys.stderr)
        return 2
    print(gen.answer(question, hits))
    print("\nsources: " + ", ".join(f"[{i}] {h['doc_id']}" for i, h in enumerate(hits, 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
