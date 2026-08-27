#!/usr/bin/env python3
"""Compare two retrieval eval runs question by question.

Aggregate numbers hide the interesting part: which questions a change fixed and
which it broke. A change that nets +2% while breaking eleven questions and fixing
thirteen is a different thing from one that fixes two and breaks none.

Usage: compare_runs.py phase1-prefix phase1-noprefix
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "eval" / "results"
KS = (1, 3, 5, 10, 20)


def load(name: str) -> dict:
    p = RESULTS / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"no such run: {p}")
    d = json.loads(p.read_text())
    d["by_qid"] = {q["qid"]: q for q in d["per_question"]}
    return d


def recall_at(pool, k):
    pool = [p for p in pool if p["kind"] == "answerable"]
    if not pool:
        return 0.0
    return sum(1 for p in pool if p["first_hit_rank"] and p["first_hit_rank"] <= k) / len(pool)


def mrr(pool, k=10):
    pool = [p for p in pool if p["kind"] == "answerable"]
    if not pool:
        return 0.0
    return sum(1 / p["first_hit_rank"] if p["first_hit_rank"] and p["first_hit_rank"] <= k else 0
               for p in pool) / len(pool)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("-k", type=int, default=5, help="k at which fixed/broken is judged")
    args = ap.parse_args()

    a, b = load(args.baseline), load(args.candidate)
    qids = [q for q in a["by_qid"] if q in b["by_qid"]]
    av = [a["by_qid"][q] for q in qids]
    bv = [b["by_qid"][q] for q in qids]

    print(f"baseline  {args.baseline:24} {a.get('meta',{})}")
    print(f"candidate {args.candidate:24} {b.get('meta',{})}\n")

    print("metric            baseline  candidate   delta")
    for k in KS:
        x, y = recall_at(av, k), recall_at(bv, k)
        print(f"  recall@{k:<2}        {x:7.1%}  {y:8.1%}  {y-x:+7.1%}")
    x, y = mrr(av), mrr(bv)
    print(f"  MRR@10          {x:7.3f}  {y:8.3f}  {y-x:+7.3f}")

    k = args.k
    def ok(p):
        return bool(p["first_hit_rank"] and p["first_hit_rank"] <= k)
    fixed  = [q for q in qids if a["by_qid"][q]["kind"] == "answerable"
              and not ok(a["by_qid"][q]) and ok(b["by_qid"][q])]
    broken = [q for q in qids if a["by_qid"][q]["kind"] == "answerable"
              and ok(a["by_qid"][q]) and not ok(b["by_qid"][q])]
    print(f"\nat k={k}: fixed {len(fixed)}, broken {len(broken)}, net {len(fixed)-len(broken):+d}")
    if fixed:
        print("  fixed  : " + ", ".join(sorted(fixed)))
    if broken:
        print("  broken : " + ", ".join(sorted(broken)))

    print("\nby tag                    n   baseline  candidate   delta")
    groups = defaultdict(list)
    for q in qids:
        if a["by_qid"][q]["kind"] != "answerable":
            continue
        for t in a["by_qid"][q]["tags"]:
            groups[t].append(q)
    for t in sorted(groups, key=lambda x: -len(groups[x])):
        g = groups[t]
        x = recall_at([a["by_qid"][q] for q in g], k)
        y = recall_at([b["by_qid"][q] for q in g], k)
        flag = "  <-" if abs(y - x) >= 0.10 else ""
        print(f"  {t:22} {len(g):3}   {x:7.1%}  {y:8.1%}  {y-x:+7.1%}{flag}")

    # Abstention headroom: separation between answerable and unanswerable top scores.
    for label, run in ((args.baseline, a), (args.candidate, b)):
        per = run["per_question"]
        ansc = sorted(p["top_score"] for p in per if p["kind"] == "answerable")
        unac = sorted(p["top_score"] for p in per if p["kind"] == "unanswerable")
        best = (0, 0)
        for t in [x / 400 for x in range(160, 320)]:
            kept = sum(1 for s in ansc if s >= t) / len(ansc)
            rec = sum(1 for s in unac if s < t) / len(unac)
            if kept >= 0.90 and rec > best[1]:
                best = (t, rec)
        print(f"\n{label:24} score gate @>=90% coverage: "
              f"threshold {best[0]:.3f} -> abstention recall {best[1]:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
