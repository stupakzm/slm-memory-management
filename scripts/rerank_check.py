#!/usr/bin/env python3
"""Phase 14: does a reranker score depend on the other documents it is batched with?

For a seeded sample of dev questions this scores one question's candidates in two
contexts and reports the largest score difference for any chunk present in both:

  (i)  the open top --candidates, as the run-time retriever would score them;
  (ii) the deduped union of that list and every domain's top --domain-k, shuffled
       (so each chunk gets different batch-mates, and a different position).

If scores are independent of batch composition the drift is exactly 0 (or float
noise); the pre-registration in docs/phase14-results.md sets the threshold. The
client batch size (--batch) is applied to both contexts; the server's --parallel is set
with `servers.sh start reranker` (see that script).

  .venv/bin/python scripts/rerank_check.py --split dev --eval data/eval/questions.jsonl \\
      --batch 16 --out data/eval/results/p14-rbatch-16.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from smm import store  # noqa: E402
from smm.embed import Embedder  # noqa: E402
from smm.rerank import Reranker  # noqa: E402

from quota_sweep import split_rows  # noqa: E402


def check_question(question: str, emb, rr, db, domains: list, candidates: int,
                   domain_k: int, context_seed: int) -> dict:
    """Score `question`'s candidates in contexts (i) and (ii); see the module docstring."""
    vec = emb.embed_query(question)
    open_hits = store.search(db, vec, k=candidates, domain=None)
    union: dict[str, dict] = {}
    for h in open_hits:
        union.setdefault(h["chunk_id"], h)
    for d in domains:
        for h in store.search(db, vec, k=domain_k, domain=d):
            union.setdefault(h["chunk_id"], h)
    t0 = time.time()
    first = {h["chunk_id"]: h["rerank_score"] for h in rr.rerank(question, open_hits)}
    seconds = time.time() - t0
    shuffled = list(union.values())
    random.Random(context_seed).shuffle(shuffled)
    second = {h["chunk_id"]: h["rerank_score"] for h in rr.rerank(question, shuffled)}
    shared = sorted(set(first) & set(second))
    drift = max((abs(first[c] - second[c]) for c in shared), default=0.0)
    return {"drift": drift, "rerank_seconds": seconds, "n_open": len(first),
            "n_union": len(second), "n_shared": len(shared)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/index/phase2.db")
    ap.add_argument("--eval", action="append", default=None,
                    help="eval jsonl (repeatable; rows are merged by qid)")
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--split-seed", type=int, default=14)
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--sample-seed", type=int, default=18)
    ap.add_argument("--batch", type=int, default=16, help="documents per reranker request")
    ap.add_argument("--candidates", type=int, default=50)
    ap.add_argument("--domain-k", type=int, default=25)
    ap.add_argument("--context-seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write per-question details here (JSON)")
    args = ap.parse_args()

    rows: list[dict] = []
    seen_qids: set[str] = set()
    for path in args.eval or ["data/eval/questions.jsonl"]:
        for line in (ROOT / path).open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["qid"] in seen_qids:
                print(f"duplicate qid {row['qid']!r} (in {path})", file=sys.stderr)
                return 2
            seen_qids.add(row["qid"])
            rows.append(row)
    dev_rows, test_rows = split_rows(rows, args.split_seed)
    pool = dev_rows if args.split == "dev" else test_rows
    by_qid = {r["qid"]: r for r in pool}
    qids = random.Random(args.sample_seed).sample(sorted(by_qid), min(args.sample, len(by_qid)))

    emb, rr = Embedder(), Reranker(batch=args.batch)
    if not emb.health():
        print("embedder not running: ./scripts/servers.sh start embedder", file=sys.stderr)
        return 2
    if not rr.health():
        print("reranker not running: ./scripts/servers.sh start reranker", file=sys.stderr)
        return 2
    db = store.connect(ROOT / args.db)
    domains = sorted(store.domains(db))

    details = []
    for qid in qids:
        res = check_question(by_qid[qid]["question"], emb, rr, db, domains,
                             args.candidates, args.domain_k, args.context_seed)
        details.append(dict(res, qid=qid))
    max_drift = max((d["drift"] for d in details), default=0.0)
    mean_s = sum(d["rerank_seconds"] for d in details) / len(details) if details else 0.0
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"batch": args.batch, "max_drift": max_drift,
                                   "mean_rerank_s": mean_s, "n": len(details),
                                   "args": vars(args), "questions": details}, indent=2))
    print(f"batch {args.batch}: max_drift {max_drift:.3e} mean_rerank_s {mean_s:.3f} "
          f"n {len(details)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
