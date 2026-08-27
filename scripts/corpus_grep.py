#!/usr/bin/env python3
"""Search the extracted corpus. Used to locate gold sections when writing eval
questions, and generally to check a claim against what is actually installed.

Usage:
  corpus_grep.py PATTERN [--doc tar.1] [--heading OPTIONS] [--sec 1,8]
                         [--context 1] [--limit 20] [--ids-only]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus" / "man.jsonl"


def load(path: Path = CORPUS):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("--doc", help="restrict to a doc_id or name prefix")
    ap.add_argument("--heading", help="restrict to sections whose heading matches")
    ap.add_argument("--sec", help="restrict to man sections, e.g. 1,8")
    ap.add_argument("--context", type=int, default=1, help="lines of context")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--ids-only", action="store_true")
    ap.add_argument("--fixed", action="store_true", help="literal string, not regex")
    args = ap.parse_args()

    pat = re.compile(re.escape(args.pattern) if args.fixed else args.pattern)
    heading_pat = re.compile(args.heading, re.I) if args.heading else None
    keep_secs = tuple(s.strip() for s in args.sec.split(",")) if args.sec else None

    hits = 0
    for doc in load():
        if args.doc and not (doc["doc_id"] == args.doc or doc["name"].startswith(args.doc)):
            continue
        if keep_secs and not doc["section"].startswith(keep_secs):
            continue
        for sec in doc["sections"]:
            if heading_pat and not heading_pat.search(sec["heading"]):
                continue
            if not pat.search(sec["text"]):
                continue
            hits += 1
            if args.ids_only:
                print(sec["sec_id"])
            else:
                print(f"\n\033[1m{sec['sec_id']}\033[0m  ({len(sec['text'])} ch)")
                lines = sec["text"].splitlines()
                shown = set()
                for i, ln in enumerate(lines):
                    if pat.search(ln):
                        for j in range(max(0, i - args.context), min(len(lines), i + args.context + 1)):
                            if j not in shown:
                                shown.add(j)
                                print(f"    {lines[j]}")
                        if len(shown) > 24:
                            print("    ...")
                            break
            if hits >= args.limit:
                print(f"\n[stopped at --limit {args.limit}]")
                return 0
    print(f"\n{hits} section(s) matched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
