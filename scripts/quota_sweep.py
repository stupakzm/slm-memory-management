#!/usr/bin/env python3
"""Phase 14 R11: simulate every floor of the floor quota from one rerank pass.

A cross-encoder scores each question-passage pair independently of the other
candidates in its pool, so a floor's reranked top k can be read off cached scores.
For each question this embeds once, runs the open search (top --candidates) and each
domain's own search (top max floor), reranks the UNION of those once, and then, for
every floor F, builds smm.retrieve.floor_pool and cuts its scored chunks to the top
k. One retrieval pass instead of one per floor. docs/phase14-results.md says how the
premise is validated before any result is read.

The eval rows are split into a dev and a test half at the BASE-question level: every
variant or paraphrase follows its base (found by following `variant_of` /
`paraphrase_of` to the root), so no wording of a test question is seen in dev.

Output, per floor: data/eval/results/NAME-f{F}-retrieved.json, the cache that
`eval_answers.py --stage generate --cache NAME-f{F}` reads. With --split, also
NAME-qids.json, the processed qids, for `eval_answers.py --qids`.

  .venv/bin/python scripts/quota_sweep.py --print-split --eval data/eval/questions.jsonl ...
  .venv/bin/python scripts/quota_sweep.py --split dev --name p14-dev --eval ... --eval ...
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
from smm.retrieve import floor_pool  # noqa: E402

from eval_answers import cache_entry  # noqa: E402


def base_of(qid: str, by_qid: dict) -> str:
    """Root of a row's variant/paraphrase chain; a pointer to a qid that is not in the
    eval set ends the chain at that qid, so the variants of a missing base still group."""
    seen: set[str] = set()
    while qid in by_qid and qid not in seen:
        seen.add(qid)
        parent = by_qid[qid].get("variant_of") or by_qid[qid].get("paraphrase_of")
        if not parent:
            break
        qid = parent
    return qid


def split_rows(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """(dev rows, test rows): the sorted base qids are shuffled with Random(seed), the
    first half are dev. Rows keep the eval set's own order."""
    by_qid = {r["qid"]: r for r in rows}
    bases = sorted({base_of(r["qid"], by_qid) for r in rows})
    random.Random(seed).shuffle(bases)
    dev = set(bases[: len(bases) // 2])
    return ([r for r in rows if base_of(r["qid"], by_qid) in dev],
            [r for r in rows if base_of(r["qid"], by_qid) not in dev])


def count_line(label: str, rows: list[dict], by_qid: dict) -> str:
    return f"{label}: bases {len({base_of(r['qid'], by_qid) for r in rows})} rows {len(rows)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="append", default=None,
                    help="eval jsonl (repeatable; rows are merged by qid)")
    ap.add_argument("--db", default="data/index/phase2.db")
    ap.add_argument("--name", default="quota-sweep")
    ap.add_argument("--floors", default="0,5,10,15,20,25")
    ap.add_argument("--candidates", type=int, default=50)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--split", choices=("dev", "test"), default=None)
    ap.add_argument("--split-seed", type=int, default=14)
    ap.add_argument("--qids", default=None,
                    help="path to a JSON list of qids; restricts rows to those qids")
    ap.add_argument("--print-split", action="store_true",
                    help="print the dev and test base/row counts and exit")
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
    by_qid = {r["qid"]: r for r in rows}
    dev_rows, test_rows = split_rows(rows, args.split_seed)

    if args.print_split:
        print(count_line("dev", dev_rows, by_qid))
        print(count_line("test", test_rows, by_qid))
        return 0

    try:
        floors = sorted({int(f) for f in args.floors.split(",")})
    except ValueError:
        print(f"--floors must be comma-separated integers: {args.floors!r}", file=sys.stderr)
        return 2
    if not floors or floors[0] < 0:
        print("--floors must be non-negative", file=sys.stderr)
        return 2

    if args.split:
        rows = dev_rows if args.split == "dev" else test_rows
    if args.qids:
        qid_set = set(json.loads(Path(args.qids).read_text()))
        rows = [r for r in rows if r["qid"] in qid_set]

    emb, rr = Embedder(), Reranker()
    if not emb.health():
        print("embedder not running: ./scripts/servers.sh start embedder", file=sys.stderr)
        return 2
    if not rr.health():
        print("reranker not running: ./scripts/servers.sh start reranker", file=sys.stderr)
        return 2
    db = store.connect(ROOT / args.db)
    domains = sorted(store.domains(db))
    if len(domains) * floors[-1] > args.candidates:
        print(f"{len(domains)} domains x floor {floors[-1]} exceeds --candidates "
              f"{args.candidates}", file=sys.stderr)
        return 2
    max_floor = floors[-1]

    retrieved: dict[int, dict] = {f: {} for f in floors}
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        q = row["question"]
        vec = emb.embed_query(q)
        open_hits = store.search(db, vec, k=args.candidates, domain=None)
        per_domain = {d: (store.search(db, vec, k=max_floor, domain=d) if max_floor else [])
                      for d in domains}
        union: dict[str, dict] = {}
        for lst in [open_hits, *(per_domain[d] for d in domains)]:
            for h in lst:
                union.setdefault(h["chunk_id"], h)
        scored = {h["chunk_id"]: h for h in rr.rerank(q, list(union.values()))}
        for f in floors:
            pool = floor_pool(open_hits, {d: lst[:f] for d, lst in per_domain.items()},
                              args.candidates, f)
            hits = sorted((scored[h["chunk_id"]] for h in pool),
                          key=lambda h: (-h["rerank_score"], h["chunk_id"]))[: args.k]
            retrieved[f][row["qid"]] = cache_entry(hits, hits, 0)
        if sys.stdout.isatty():
            print(f"\r  sweep {i}/{len(rows)}  {(time.time()-t0)/i:.2f}s/q", end="", flush=True)

    outdir = ROOT / "data" / "eval" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    for f in floors:
        (outdir / f"{args.name}-f{f}-retrieved.json").write_text(json.dumps(retrieved[f]))
    if args.split:
        (outdir / f"{args.name}-qids.json").write_text(json.dumps([r["qid"] for r in rows]))
    print(f"\n  swept floors {floors} over {len(rows)} questions in {time.time()-t0:.0f}s "
          f"-> {outdir}/{args.name}-f*-retrieved.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
