"""Retrieval drift check (phase 13's control drift check, tracked).

sample:  random.Random(seed).sample(pool qids in file order, n) -> JSON qid list
         (the format eval_answers.py --qids reads).
compare: for those qids, which rows differ between two retrieval caches in
         ordered top-k chunk ids or in the gate decision.

Stdlib only; unpack_entry and gate_score are copied (eval_answers needs
sqlite_vec, smm.retrieve needs the full stack).
"""

import argparse
import json
import random
import sys
from pathlib import Path


def unpack_entry(entry):
    # copy of scripts/eval_answers.py unpack_entry
    if isinstance(entry, dict):
        return entry["hits"], entry["gate_hits"]
    return entry, entry


def gate_score(hits):
    # copy of smm.retrieve.gate_score
    if not hits:
        return float("-inf")
    return hits[0].get("rerank_score", hits[0].get("score", 0.0))


def pool_qids(files):
    qids, seen = [], set()
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                q = json.loads(line)["qid"]
                if q in seen:
                    print(f"duplicate qid in pool: {q}", file=sys.stderr)
                    sys.exit(2)
                seen.add(q)
                qids.append(q)
    return qids


def cmd_sample(args):
    qids = pool_qids(args.pool)
    picked = random.Random(args.seed).sample(qids, args.n)
    Path(args.out).write_text(json.dumps(picked), encoding="utf-8")
    return 0


def cmd_compare(args):
    qids = json.loads(Path(args.qids).read_text(encoding="utf-8"))
    a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    for name, cache in (("a", a), ("b", b)):
        missing = [q for q in qids if q not in cache]
        if missing:
            print(f"qid missing from cache {name}: {', '.join(missing)}", file=sys.stderr)
            return 2
    n_diff = n_top = n_gate = 0
    for q in qids:
        ha, ga = unpack_entry(a[q])
        hb, gb = unpack_entry(b[q])
        top = [h["chunk_id"] for h in ha[: args.k]] != [h["chunk_id"] for h in hb[: args.k]]
        gate = (gate_score(ga) < args.gate) != (gate_score(gb) < args.gate)
        if top or gate:
            n_diff += 1
            n_top += top
            n_gate += gate
            print(f"{q}: " + ",".join(x for x, f in (("top-k", top), ("gate", gate)) if f))
    print(f"differ: {n_diff} of {len(qids)} (top-k {n_top}, gate {n_gate})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--pool", nargs="+", required=True)
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--seed", type=int, default=13)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_sample)
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--qids", required=True)
    c.add_argument("--k", type=int, default=5)
    c.add_argument("--gate", type=float, default=0.65)
    c.set_defaults(fn=cmd_compare)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
