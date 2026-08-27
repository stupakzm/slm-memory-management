#!/usr/bin/env python3
"""Score retrieval against the phase 0 eval set.

A retrieved chunk counts as a hit when it contains every one of the question's
answer tokens. That rule is deliberately independent of how the corpus was cut up,
so a flat baseline, a section index and a tree can all be scored on the same
footing - which is the only way phase 3 can be judged against phase 2.

Usage: eval_retrieval.py --db data/index/phase1.db --name phase1-prefix
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import store  # noqa: E402
from smm.embed import Embedder  # noqa: E402

KS = (1, 3, 5, 10, 20)


def hit(chunk_text: str, tokens: list[str]) -> bool:
    return all(t in chunk_text for t in tokens)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/index/phase1.db")
    ap.add_argument("--eval", default="data/eval/questions.jsonl")
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--name", default="run")
    ap.add_argument("--topk", type=int, default=max(KS))
    args = ap.parse_args()

    emb = Embedder(args.url)
    if not emb.health():
        print(f"no llama-server at {args.url}", file=sys.stderr)
        return 2
    db = store.connect(ROOT / args.db)
    meta = store.get_meta(db)
    n_chunks = store.count(db)
    print(f"index: {n_chunks} chunks  meta={meta}\n")

    rows = [json.loads(l) for l in (ROOT / args.eval).open(encoding="utf-8")]
    per_q, t0 = [], time.time()
    for r in rows:
        qv = emb.embed_query(r["question"])
        res = store.search(db, qv, k=args.topk)
        rec = {
            "qid": r["qid"], "kind": r["kind"], "tags": r["tags"],
            "variant_kind": r.get("variant_kind"), "variant_of": r.get("variant_of"),
            "paraphrase_of": r.get("paraphrase_of"),
            "top_score": res[0]["score"] if res else 0.0,
            "top_doc": res[0]["doc_id"] if res else None,
        }
        if r["kind"] == "answerable":
            toks = r["answer_contains"]
            ranks = [i + 1 for i, c in enumerate(res) if hit(c["text"], toks)]
            rec["first_hit_rank"] = ranks[0] if ranks else None
            rec["n_hits"] = len(ranks)
            rec["doc_rank"] = next((i + 1 for i, c in enumerate(res) if c["doc_id"] == r["doc"]), None)
        per_q.append(rec)
    elapsed = time.time() - t0

    ans = [p for p in per_q if p["kind"] == "answerable"]
    una = [p for p in per_q if p["kind"] == "unanswerable"]

    def recall_at(pool, k):
        if not pool:
            return 0.0
        return sum(1 for p in pool if p["first_hit_rank"] and p["first_hit_rank"] <= k) / len(pool)

    def mrr(pool, k=10):
        if not pool:
            return 0.0
        return sum(1 / p["first_hit_rank"] if p["first_hit_rank"] and p["first_hit_rank"] <= k else 0
                   for p in pool) / len(pool)

    print(f"=== {args.name} ===  {len(rows)} questions in {elapsed:.0f}s "
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
    print("\ntop-1 score distribution   p10     p50     p90")
    print(f"  answerable   n={len(ans):3}    {q(a_top,.1):.4f}  {q(a_top,.5):.4f}  {q(a_top,.9):.4f}")
    print(f"  unanswerable n={len(una):3}    {q(u_top,.1):.4f}  {q(u_top,.5):.4f}  {q(u_top,.9):.4f}")
    overlap = sum(1 for u in u_top if u >= q(a_top, .1)) / max(len(u_top), 1)
    print(f"  unanswerable scoring above answerable p10: {overlap:.1%}"
          f"   (low = a threshold gate can work)")

    outdir = ROOT / "data" / "eval" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{args.name}.json"
    outfile.write_text(json.dumps({
        "name": args.name, "db": args.db, "meta": meta, "n_chunks": n_chunks,
        "elapsed_s": round(elapsed, 1),
        "recall": {f"@{k}": recall_at(ans, k) for k in KS},
        "mrr@10": mrr(ans), "gold_doc_top5": doc_at5,
        "per_question": per_q,
    }, indent=2))
    print(f"\nwrote {outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
