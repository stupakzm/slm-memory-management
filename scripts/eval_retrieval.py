#!/usr/bin/env python3
"""Score retrieval against the phase 0 eval set.

A retrieved chunk counts as a hit when it contains every one of the question's
answer tokens. That rule is deliberately independent of how the corpus was cut up,
so a flat baseline, a section index and a tree can all be scored on the same
footing - which is the only way phase 3 can be judged against phase 2.

Phase 2 adds the pipeline switches. Each layer can be turned off independently, so
every claim below is an ablation rather than a bundle:

  --mode dense                      phase 1, reproduced
  --mode bm25                       the lexical half alone
  --mode hybrid                     RRF over both
  --mode hybrid --rerank            the full phase 2 pipeline

Usage: eval_retrieval.py --db data/index/phase2.db --mode hybrid --rerank --name phase2-hybrid-rerank
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import lexical, store  # noqa: E402
from smm.embed import Embedder  # noqa: E402
from smm.rerank import Reranker  # noqa: E402
from smm.retrieve import Retriever  # noqa: E402

KS = (1, 3, 5, 10, 20)


def hit(chunk_text: str, tokens: list[str]) -> bool:
    return all(t in chunk_text for t in tokens)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/index/phase2.db")
    ap.add_argument("--eval", default="data/eval/questions.jsonl")
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--rerank-url", default="http://127.0.0.1:8082")
    ap.add_argument("--name", default="run")
    ap.add_argument("--mode", choices=("dense", "bm25", "hybrid"), default="dense")
    ap.add_argument("--rerank", action="store_true", help="cross-encoder over the candidate pool")
    ap.add_argument("--candidates", type=int, default=50, help="pool size handed to the reranker")
    ap.add_argument("--dense-weight", type=float, default=1.0)
    ap.add_argument("--bm25-weight", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=max(KS), help="depth the metrics are scored to")
    args = ap.parse_args()

    emb = None
    if args.mode in ("dense", "hybrid"):
        emb = Embedder(args.url)
        if not emb.health():
            print(f"no embedder at {args.url}", file=sys.stderr)
            return 2
    rr = None
    if args.rerank:
        rr = Reranker(args.rerank_url)
        if not rr.health():
            print(f"no reranker at {args.rerank_url}: "
                  f"./scripts/servers.sh start reranker", file=sys.stderr)
            return 2

    db = store.connect(ROOT / args.db)
    if args.mode in ("bm25", "hybrid") and not lexical.has_index(db):
        print(f"{args.db} has no chunks_fts: .venv/bin/python scripts/build_lexical.py --db {args.db}",
              file=sys.stderr)
        return 2
    meta = store.get_meta(db)
    n_chunks = store.count(db)
    # Without a reranker the candidate pool *is* the ranking, so it need only be as
    # deep as the metrics are scored; with one, the pool is what gets rescored.
    pool = max(args.candidates, args.topk) if args.rerank else args.topk
    r = Retriever(db, embedder=emb, reranker=rr, mode=args.mode, candidates=pool,
                  dense_weight=args.dense_weight, bm25_weight=args.bm25_weight)
    config = {"mode": args.mode, "rerank": args.rerank, "candidates": pool,
              "dense_weight": args.dense_weight, "bm25_weight": args.bm25_weight}
    print(f"index: {n_chunks} chunks  meta={meta}")
    print(f"pipeline: {config}\n")

    rows = [json.loads(l) for l in (ROOT / args.eval).open(encoding="utf-8")]
    per_q, t0 = [], time.time()
    for i, row in enumerate(rows, 1):
        res = r.retrieve(row["question"], k=args.topk)
        top = res[0] if res else None
        rec = {
            "qid": row["qid"], "kind": row["kind"], "tags": row["tags"],
            "variant_kind": row.get("variant_kind"), "variant_of": row.get("variant_of"),
            "paraphrase_of": row.get("paraphrase_of"),
            # `top_score` is whatever the *gate* would threshold: the reranker score
            # when there is one, the dense/RRF score otherwise.
            "top_score": (top.get("rerank_score", top.get("score", 0.0)) if top else 0.0),
            "top_dense": (top.get("score", 0.0) if top else 0.0),
            "top_doc": top["doc_id"] if top else None,
        }
        if row["kind"] == "answerable":
            toks = row["answer_contains"]
            ranks = [j + 1 for j, c in enumerate(res) if hit(c["text"], toks)]
            rec["first_hit_rank"] = ranks[0] if ranks else None
            rec["n_hits"] = len(ranks)
            rec["doc_rank"] = next((j + 1 for j, c in enumerate(res) if c["doc_id"] == row["doc"]), None)
        per_q.append(rec)
        if sys.stdout.isatty():
            print(f"\r  {i}/{len(rows)}  {(time.time()-t0)/i:.2f}s/q", end="", flush=True)
    elapsed = time.time() - t0

    ans = [p for p in per_q if p["kind"] == "answerable"]
    una = [p for p in per_q if p["kind"] == "unanswerable"]

    def recall_at(pool_, k):
        if not pool_:
            return 0.0
        return sum(1 for p in pool_ if p["first_hit_rank"] and p["first_hit_rank"] <= k) / len(pool_)

    def mrr(pool_, k=10):
        if not pool_:
            return 0.0
        return sum(1 / p["first_hit_rank"] if p["first_hit_rank"] and p["first_hit_rank"] <= k else 0
                   for p in pool_) / len(pool_)

    print(f"\n=== {args.name} ===  {len(rows)} questions in {elapsed:.0f}s "
          f"({elapsed/len(rows)*1000:.0f} ms/q)\n")
    print("answer recall@k  " + "  ".join(f"@{k}={recall_at(ans,k):6.1%}" for k in KS))
    print(f"MRR@10           {mrr(ans):.3f}")
    doc_at5 = sum(1 for p in ans if p["doc_rank"] and p["doc_rank"] <= 5) / len(ans)
    print(f"gold doc in top5 {doc_at5:.1%}")

    print("\nby tag                    n   r@5    r@20   MRR")
    groups = defaultdict(list)
    for p in ans:
        for t in p["tags"]:
            groups[t].append(p)
    for t in sorted(groups, key=lambda x: -len(groups[x])):
        g = groups[t]
        print(f"  {t:22} {len(g):3}  {recall_at(g,5):5.1%}  {recall_at(g,20):5.1%}  {mrr(g):.3f}")

    # Query-noise robustness: variants scored against their own base.
    base = {p["qid"]: p for p in ans if not p.get("variant_kind")}
    print("\nquery-noise robustness    n   base r@5 -> variant r@5")
    vk = defaultdict(list)
    for p in ans:
        if p.get("variant_kind"):
            vk[p["variant_kind"]].append(p)
    for kind, g in sorted(vk.items()):
        bases = [base[p["variant_of"]] for p in g if p["variant_of"] in base]
        print(f"  {kind:22} {len(g):3}  {recall_at(bases,5):5.1%} -> {recall_at(g,5):5.1%}")

    # The abstention signal: can a score threshold separate the two populations?
    a_top = sorted(p["top_score"] for p in ans)
    u_top = sorted(p["top_score"] for p in una)
    def q(v, f):
        return v[min(int(len(v) * f), len(v) - 1)] if v else 0.0
    print("\ntop-1 score distribution   p10      p50      p90")
    print(f"  answerable   n={len(ans):3}  {q(a_top,.1):8.4f} {q(a_top,.5):8.4f} {q(a_top,.9):8.4f}")
    print(f"  unanswerable n={len(una):3}  {q(u_top,.1):8.4f} {q(u_top,.5):8.4f} {q(u_top,.9):8.4f}")
    overlap = sum(1 for u in u_top if u >= q(a_top, .1)) / max(len(u_top), 1)
    print(f"  unanswerable scoring above answerable p10: {overlap:.1%}"
          f"   (low = a threshold gate can work)")

    outdir = ROOT / "data" / "eval" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{args.name}.json"
    outfile.write_text(json.dumps({
        "name": args.name, "db": args.db, "meta": meta, "config": config,
        "n_chunks": n_chunks, "elapsed_s": round(elapsed, 1),
        "recall": {f"@{k}": recall_at(ans, k) for k in KS},
        "mrr@10": mrr(ans), "gold_doc_top5": doc_at5,
        "per_question": per_q,
    }, indent=2))
    print(f"\nwrote {outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
