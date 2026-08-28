#!/usr/bin/env python3
"""Record or check the identity of the extracted corpus.

  .venv/bin/python scripts/corpus_fingerprint.py            # print it
  .venv/bin/python scripts/corpus_fingerprint.py --write    # record it
  .venv/bin/python scripts/corpus_fingerprint.py --check    # has the machine moved?

`--check` exits non-zero when the corpus on disk no longer matches the recorded
fingerprint, and names the documents that were added, removed or changed. That is
the signal phase 4 did not have: 33 cmake pages appeared between two extractions and
nothing said so.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import fingerprint as fp  # noqa: E402

EXTRACTOR = ROOT / "src" / "smm" / "corpus" / "manpages.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus/man.jsonl")
    ap.add_argument("--out", default="data/corpus/fingerprint.json")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--limit", type=int, default=15, help="documents to name per class")
    args = ap.parse_args()

    corpus = ROOT / args.corpus
    if not corpus.exists():
        print(f"no corpus at {corpus}", file=sys.stderr)
        return 2
    cur = fp.compute(corpus, EXTRACTOR)
    out = ROOT / args.out
    print(f"corpus     {fp.summary(cur)}")

    old = fp.load(out)
    if old:
        print(f"recorded   {fp.summary(old)}")

    if args.check:
        if not old:
            print(f"\nno recorded fingerprint at {out}; run --write", file=sys.stderr)
            return 2
        if old["digest"] == cur["digest"]:
            print("\nmatch - the corpus is the one the recorded numbers were measured on")
            return 0
        d = fp.diff(old, cur)
        print("\nCORPUS HAS MOVED")
        if d["extractor_changed"]:
            print("  the extractor itself changed, so every document may differ")
        for label in ("added", "removed", "changed"):
            names = d[label]
            if not names:
                continue
            shown = ", ".join(names[: args.limit])
            more = f" (+{len(names) - args.limit} more)" if len(names) > args.limit else ""
            print(f"  {label:8} {len(names):5}  {shown}{more}")
        print("\nIndexes and eval results recorded against the old digest are no longer\n"
              "strictly comparable. Rebuild, or say so when reporting.")
        return 1

    if args.write:
        fp.save(cur, out)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
