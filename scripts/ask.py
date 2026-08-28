#!/usr/bin/env python3
"""Ask the corpus a question. Phase 2: rerank, gate, cited answer.

  .venv/bin/python scripts/ask.py "how do I exclude files listed in a text file from a tar archive"
  .venv/bin/python scripts/ask.py --retrieve-only -k 10 "watch a log file as it grows"
  .venv/bin/python scripts/ask.py --no-gate "how do I install python packages with pip"   # see it speak

The gate is on by default and the threshold is not a taste decision - it is the
operating point `scripts/sweep_gate.py` picked off the labelled set. Below it the
generator is never invoked, so there is nothing to speculate with.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import grammar, store  # noqa: E402
from smm.embed import Embedder  # noqa: E402
from smm.generate import Generator  # noqa: E402
from smm.rerank import Reranker  # noqa: E402
from smm.retrieve import Retriever, gate_score  # noqa: E402

# sweep_gate.py --answers, swept on end-to-end outcomes rather than on a retrieval
# proxy for them: 92.5% abstention recall for 1.6 points of answer accuracy.
GATE = 0.65


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--db", default="data/index/phase2.db")
    ap.add_argument("--candidates", type=int, default=50)
    ap.add_argument("--gate", type=float, default=GATE)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-grammar", action="store_true")
    ap.add_argument("--retrieve-only", action="store_true")
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args()
    question = " ".join(args.question)

    emb = Embedder()
    if not emb.health():
        print("embedder not running: ./scripts/servers.sh start embedder", file=sys.stderr)
        return 2
    rr = None
    if not args.no_rerank:
        rr = Reranker()
        if not rr.health():
            print("reranker not running: ./scripts/servers.sh start reranker", file=sys.stderr)
            return 2

    db = store.connect(ROOT / args.db)
    r = Retriever(db, embedder=emb, reranker=rr, mode="dense", candidates=args.candidates)
    hits = r.retrieve(question, k=args.k)
    score = gate_score(hits)

    if args.retrieve_only or args.show_context:
        for i, h in enumerate(hits, 1):
            head = h["text"].splitlines()[0][:66] if h["text"] else ""
            print(f"[{i}] {h.get('rerank_score', h['score']):.4f}  {h['chunk_id']:24} {head}")
        if args.retrieve_only:
            return 0
        print()

    if not args.no_gate and score < args.gate:
        print(f"{grammar.REFUSAL}\n\n(gate: best evidence scored {score:.3f}, "
              f"below {args.gate:.3f} - the model was not asked)")
        return 0

    gen = Generator()
    if not gen.health():
        print("generator not running: ./scripts/servers.sh start generator", file=sys.stderr)
        return 2
    print(gen.answer(question, hits, cite_grammar=not args.no_grammar))
    print("\nsources: " + ", ".join(f"[{i}] {h['doc_id']}" for i, h in enumerate(hits, 1)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
