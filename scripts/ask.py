#!/usr/bin/env python3
"""Ask the corpus a question: rerank, gate, cited answer.

The default index is the phase 2 flat one. Phase 3's structure-aware chunker is
still here behind `--db data/index/phase3-mixed.db --expand 1`, and it is not the
default because it did not beat this - see docs/phase3-results.md.

  .venv/bin/python scripts/ask.py "how do I exclude files listed in a text file from a tar archive"
  .venv/bin/python scripts/ask.py --retrieve-only -k 10 "watch a log file as it grows"
  .venv/bin/python scripts/ask.py --no-gate "how do I install python packages with pip"   # see it speak
  .venv/bin/python scripts/ask.py --act "delete the build directory and everything in it"

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

from smm import grammar, store, tools  # noqa: E402
from smm.embed import Embedder  # noqa: E402
from smm.generate import Generator  # noqa: E402
from smm.rerank import Reranker  # noqa: E402
from smm.retrieve import Retriever, expand, gate_score  # noqa: E402

# sweep_gate.py --answers, swept on end-to-end outcomes rather than on a retrieval
# proxy for them: 92.5% abstention recall for 1.6 points of answer accuracy.
GATE = 0.65

# Tool mode gets its own threshold, and the gap is the phase 4 finding: a request
# ("delete the build directory") retrieves lower than the question it corresponds to
# ("how do I delete a directory"), while a request naming a tool this machine does
# not document retrieves near zero. The populations separate an order of magnitude
# further down. Inheriting 0.65 here refused a quarter of the answerable requests.
GATE_ACT = 0.30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--db", default="data/index/phase2.db")
    ap.add_argument("--candidates", type=int, default=50)
    ap.add_argument("--domain", default=None,
                    help="restrict retrieval to one ingested namespace")
    ap.add_argument("--expand", type=int, default=0,
                    help="structured index only: widen each hit by N neighbouring "
                         "chunks in its section (a no-op on the flat default index)")
    ap.add_argument("--gate", type=float, default=GATE)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-grammar", action="store_true")
    ap.add_argument("--act", action="store_true",
                    help="tool mode: answer, propose a command, open a page, or refuse")
    ap.add_argument("--execute", action="store_true",
                    help="allow read-only tools to actually run; a proposed command "
                         "is never run by this program under any flag")
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
    r = Retriever(db, embedder=emb, reranker=rr, mode="dense",
                  candidates=args.candidates, domain=args.domain)
    hits = r.retrieve(question, k=args.k)
    score = gate_score(hits)
    if args.act and args.gate == GATE:
        args.gate = GATE_ACT
    if args.expand:
        hits = expand(db, hits, span=args.expand)

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

    if args.act:
        return act(gen, db, question, hits, args)

    print(gen.answer(question, hits, cite_grammar=not args.no_grammar))
    print("\nsources: " + ", ".join(f"[{i}] {h['doc_id']}" for i, h in enumerate(hits, 1)))
    return 0


def act(gen, db, question, hits, args) -> int:
    """Tool mode. The command is printed for a person to read, never run."""
    pages = {r[0] for r in db.execute("SELECT DISTINCT doc_id FROM chunks")}
    raw = gen.chat(tools.build_messages(question, hits, tools.TOOLS),
                   grammar=tools.grammar(tools.TOOLS, len(hits)), max_tokens=600)
    call = tools.parse_call(raw, tools.TOOLS, len(hits), known_pages=pages)
    if not call.valid:
        print(f"{grammar.REFUSAL}\n\n(invalid tool call: {'; '.join(call.problems)})")
        return 1

    src = f"  [{call.cite}] {hits[call.cite-1]['doc_id']}" if call.cite else ""
    if call.tool == "refuse":
        print(grammar.REFUSAL)
    elif call.tool == "answer":
        print(call.args["text"] + (f"\n\nsource:{src}" if src else ""))
    elif call.tool == "propose_command":
        print(f"$ {call.args['command']}\n\n{call.args['explanation']}")
        print(f"\nrisk: {call.risk} - not run. Review it and run it yourself.")
        if src:
            print(f"source: {src.strip()}")
    elif call.tool == "show_manpage":
        out = tools.execute(call, confirm=lambda _c: False,
                            allow_execution=args.execute)
        if out["ran"]:
            print(out["output"])
        else:
            print(f"man {call.args['section']} {call.args['page']}"
                  f"\n\n({out['reason']}; pass --execute to open it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
