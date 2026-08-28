#!/usr/bin/env python3
"""Add the BM25 half to an index that already has the dense half.

The embeddings cost 42 minutes to compute; the lexical index costs seconds and
needs no model at all, so phase 2 copies the phase 1 database rather than
rebuilding it. Same chunks, same vectors, same doc-identity prefix - the only
difference is an FTS5 table alongside them, which keeps the phase 1 and phase 2
retrieval numbers comparable on exactly the ground they should be.

  .venv/bin/python scripts/build_lexical.py --from data/index/phase1-prefix.db --out data/index/phase2.db
  .venv/bin/python scripts/build_lexical.py --db data/index/phase2.db          # rebuild in place
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import lexical, store  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", help="dense index to copy first")
    ap.add_argument("--db", default="data/index/phase2.db")
    ap.add_argument("--out", dest="out", help="alias for --db when copying")
    ap.add_argument("--tokenizer", default=lexical.TOKENIZER)
    args = ap.parse_args()

    out = ROOT / (args.out or args.db)
    if args.src:
        src = ROOT / args.src
        if not src.exists():
            print(f"no such index: {src}", file=sys.stderr)
            return 2
        if out.exists():
            print(f"{out} exists; refusing to overwrite", file=sys.stderr)
            return 2
        out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        shutil.copy2(src, out)
        print(f"copied {src.name} -> {out.name} "
              f"({out.stat().st_size/1e6:.0f} MB, {time.time()-t0:.0f}s)")

    before = out.stat().st_size
    db = store.connect(out)
    lexical.create(db, args.tokenizer)
    t0 = time.time()
    n = lexical.populate(db)
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.commit()
    store.set_meta(db, phase="2-hybrid", fts_tokenizer=args.tokenizer)
    after = out.stat().st_size
    print(f"indexed {n} chunks into chunks_fts in {time.time()-t0:.0f}s "
          f"(+{(after-before)/1e6:.0f} MB, tokenizer={args.tokenizer!r})")

    q = "how do I exclude files listed in a text file from a tar archive"
    print(f"\nsmoke: {q!r}\n  terms: {lexical.terms(q)}")
    for i, h in enumerate(lexical.search(db, q, k=5), 1):
        head = h["text"].splitlines()[0][:60]
        print(f"  [{i}] {h['score']:8.3f}  {h['chunk_id']:24} {head}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
