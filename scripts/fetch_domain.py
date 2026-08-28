#!/usr/bin/env python3
"""Fetch a second, unrelated domain: public-domain books from Project Gutenberg.

Phase 5's exit gate is that adding an unrelated domain does not regress the Linux
numbers. That needs a domain which is genuinely unrelated and genuinely large -
a handful of pasted notes would not compete for a single retrieval slot, and a test
nothing can fail proves nothing.

Books are used because they are public domain, unambiguously not Linux
documentation, and big enough to matter. The header and licence footer Project
Gutenberg wraps each text in are stripped: they are identical across every book and
would otherwise be the most duplicated passage in the index.

  .venv/bin/python scripts/fetch_domain.py --domain literature
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Curated, all long out of copyright. Anything that 404s is skipped rather than
# guessed at, and what actually arrived is reported.
BOOKS = [
    1342, 11, 84, 2701, 1661, 98, 1400, 174, 345, 46, 5200, 76, 219, 1080,
    2542, 1232, 6130, 3207, 2814, 120, 160, 205, 215, 236, 271, 514, 730,
    768, 829, 1260, 1497, 2000, 2148, 2554, 3600, 4517, 5827, 16328, 25344, 35,
]
URL = "https://www.gutenberg.org/cache/epub/{0}/pg{0}.txt"

_START = re.compile(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
                    re.I | re.S)
_END = re.compile(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*", re.I | re.S)
_TITLE = re.compile(r"^Title:\s*(.+)$", re.M)


def strip_boilerplate(raw: str) -> tuple[str, str]:
    title = (_TITLE.search(raw).group(1).strip() if _TITLE.search(raw) else "")
    m = _START.search(raw)
    body = raw[m.end():] if m else raw
    body = _END.split(body)[0]
    return title, body.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="literature")
    ap.add_argument("--out", default="data/corpus/domains")
    ap.add_argument("--limit", type=int, default=len(BOOKS))
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()

    dest = ROOT / args.out / args.domain
    dest.mkdir(parents=True, exist_ok=True)
    got = skipped = 0
    total = 0
    for i, book in enumerate(BOOKS[: args.limit], 1):
        out = dest / f"{book}.txt"
        if out.exists():
            total += out.stat().st_size
            got += 1
            continue
        try:
            req = urllib.request.Request(
                URL.format(book), headers={"User-Agent": "slm-memory-management/0.1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read().decode("utf-8", "replace")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            print(f"  skip {book}: {e}")
            skipped += 1
            time.sleep(args.delay)
            continue
        title, body = strip_boilerplate(raw)
        if len(body) < 20000:
            print(f"  skip {book}: only {len(body)} chars after stripping")
            skipped += 1
            continue
        out.write_text(f"{title}\n\n{body}", encoding="utf-8")
        total += out.stat().st_size
        got += 1
        print(f"  {i:3}/{args.limit}  {book:6} {len(body)/1e6:5.2f} MB  {title[:52]}")
        time.sleep(args.delay)

    print(f"\n{got} texts, {skipped} skipped, {total/1e6:.1f} MB -> {dest}")
    return 0 if got else 1


if __name__ == "__main__":
    raise SystemExit(main())
