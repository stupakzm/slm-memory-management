#!/usr/bin/env python3
"""Extract every English man page in the given sections to JSONL.

Usage: extract_man.py [--sections 1,8] [--out data/corpus/man.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smm.corpus.manpages import ManFile, discover, extract, redirect_target  # noqa: E402


def _work(mf: ManFile):
    target = redirect_target(mf.path)
    if target:
        return ("alias", mf.doc_id, target)
    doc = extract(mf)
    if doc is None:
        return ("fail", mf.doc_id, None)
    return ("ok", mf.doc_id, doc.to_dict())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default="1,8")
    ap.add_argument("--out", default="data/corpus/man.jsonl")
    ap.add_argument("--jobs", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    sections = tuple(s.strip() for s in args.sections.split(","))
    files = discover(sections)
    print(f"discovered {len(files)} pages in sections {sections}", file=sys.stderr)

    t0 = time.time()
    jobs = args.jobs or None
    with Pool(jobs) as pool:
        results = pool.map(_work, files, chunksize=16)

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    docs, aliases, failures = [], {}, []
    for kind, doc_id, payload in results:
        if kind == "ok":
            docs.append(payload)
        elif kind == "alias":
            aliases[doc_id] = payload
        else:
            failures.append(doc_id)

    with out_path.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    n_sections = sum(len(d["sections"]) for d in docs)
    n_edges = sum(len(d["see_also"]) for d in docs)
    heading_counts = Counter(s["heading"] for d in docs for s in d["sections"] if s["level"] == 1)
    chars = sum(len(s["text"]) for d in docs for s in d["sections"])

    stats = {
        "pages_discovered": len(files),
        "pages_extracted": len(docs),
        "so_aliases": len(aliases),
        "failures": len(failures),
        "failure_ids": failures[:50],
        "sections_total": n_sections,
        "sections_per_page": round(n_sections / max(len(docs), 1), 1),
        "see_also_edges": n_edges,
        "pages_with_edges": sum(1 for d in docs if d["see_also"]),
        "name_aliases": sum(len(d["aliases"]) for d in docs),
        "body_chars": chars,
        "top_headings": heading_counts.most_common(25),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_path.parent / "man_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "top_headings"}, indent=2))
    print("\ntop L1 headings:")
    for h, c in stats["top_headings"]:
        print(f"  {c:5}  {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
