#!/usr/bin/env python3
"""Ask the corpus a question: rerank, gate, cited answer.

The default index is the phase 2 flat one. Phase 3's structure-aware chunker is
still here behind `--db data/index/phase3-mixed.db --expand 1`, and it is not the
default because it did not beat this - see docs/phase3-results.md.

  .venv/bin/python scripts/ask.py "how do I exclude files listed in a text file from a tar archive"
  .venv/bin/python scripts/ask.py --retrieve-only -k 10 "watch a log file as it grows"
  .venv/bin/python scripts/ask.py --no-gate "how do I install python packages with pip"   # see it speak
  .venv/bin/python scripts/ask.py --act "delete the build directory and everything in it"
  .venv/bin/python scripts/ask.py --rewrites 2 "how to list files via size"   # phrasing robustness

The gate is on by default and the threshold is not a taste decision - it is the
operating point `scripts/sweep_gate.py` picked off the labelled set. Below it the
generator is never invoked, so there is nothing to speculate with.

`--rewrites N` (default 0, in every mode) fuses the question with N
model-generated rewrites when N > 0: each variant is retrieved and reranked
separately and the reranked lists are combined by rank fusion (measured; see
src/smm/retrieve.py:fuse_variants - reranking a single pre-fused pool was tried
and barely moved gold). The default single-query path never starts the generator
for rewrites; phase 11 R4b measured that one default rewrite lost 47/600 answers
against plain retrieval and cost ~1 s.

The gate itself, with rewrites on, still reads the ORIGINAL question's own
reranked top-1 - never the fused list's. rrf() orders by rank, so the fused
top-1's rerank_score is whatever the winning variant happened to score, not a
number comparable across queries; gating on it dropped answerable p10 from 0.80
to 0.30 and no threshold recovered the baseline trade (measured,
tsk_20260828_a47f0496). `GATE`/`GATE_ACT` were swept against the original
question's own top-1, so that is what stays gated on - the generator answers
from the fused `hits`, the gate decides from `gate_hits`.

Known, accepted tradeoff (applies only when `--rewrites` > 0 is passed
explicitly): the generator has to produce the
rewrites *before* retrieval can run, so it now starts before the embedder/reranker
and before the gate decision - on every such ask, including one the gate goes on
to refuse. That both reorders the embedder -> reranker -> generator lazy-start
sequence and pays the generator's ~3.7GB VRAM load on a query-only outcome, which
is exactly what the project's VRAM-budget and lazy-start notes say never happens.
This was a deliberate choice for this feature, not an oversight.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import cascade, grammar, store, tools  # noqa: E402
from smm import normalize as qnorm  # noqa: E402
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
    ap.add_argument("--db", default="data/index/main.db")
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
    ap.add_argument("--rewrites", type=int, default=0,
                    help="fuse the question with N model-generated rewrites, each "
                         "retrieved and reranked separately (default 0: no "
                         "rewrites, generator not started for them). --rewrites 0 "
                         "is the single-query path.")
    ap.add_argument("--normalize", choices=("off", "spell"), default="off",
                    help="phase 11 R4a: 'spell' corrects the question against the "
                         "index's own vocabulary (smm.normalize) before retrieval AND "
                         "generation; 'off' (default) reproduces today's behaviour "
                         "exactly")
    ap.add_argument("--vocab-cache", default=None,
                    help="override path for --normalize spell's vocab cache (default: "
                         "<db>.vocab.json next to --db)")
    ap.add_argument("--rewrite-style", choices=("man", "docs"), default="man",
                    help="phase 11 R4b: 'docs' asks the rewriter for documentation's "
                         "own terminology (manual pages and the GNU Emacs manuals) "
                         "instead of everyday words; only meaningful with --rewrites > "
                         "0. 'man' (default) reproduces today's behaviour exactly.")
    ap.add_argument("--llm-correct", action="store_true",
                    help="phase 11 R4a': ask the 4B generator itself to fix spelling "
                         "in the question (Generator.correct) before retrieval, "
                         "including any rewrites/fusion, and generation; composes with "
                         "--normalize (normalize runs first). Default off reproduces "
                         "today's behaviour exactly.")
    ap.add_argument("--cascade", action="store_true",
                    help="phase 11 R4d: rewrite only after the plain search refuses "
                         "(smm.cascade). Tier 0 is today's plain dense+rerank+gate "
                         "path; a refusal (gated or the reader abstaining) escalates "
                         "to a widened retrieval against 1, then 2, docs-style "
                         "rewrites - the reader always sees the original question. "
                         "Replaces --rewrites' fusion path entirely (--rewrites is "
                         "ignored under --cascade). Default off reproduces today's "
                         "behaviour exactly; a no-op under --retrieve-only, which "
                         "stays generator-free.")
    ap.add_argument("--question-vectors", type=int, default=0,
                    help="phase 11 R8: add the chunks of the M nearest generated-"
                         "question vectors to the candidate pool (needs an index built "
                         "by scripts/build_qvec.py). 0 (default) reproduces today's "
                         "behaviour exactly.")
    ap.add_argument("--route", choices=("dense-vote",), default=None,
                    help="phase 13 R9: infer the domain from the question (majority "
                         "domain of the open search's top 5 chunks) and retrieve within "
                         "it; exclusive with --domain. Default off reproduces today's "
                         "behaviour exactly.")
    args = ap.parse_args()
    question = " ".join(args.question)
    if args.route and args.domain:
        print("--route and --domain are exclusive: a route chooses the domain",
              file=sys.stderr)
        return 2

    if args.normalize == "spell":
        vocab = qnorm.build_vocab(ROOT / args.db, cache_path=args.vocab_cache)
        normalized_question, _normalize_edits = qnorm.normalize_query(
            question, "spell", vocab)
        if normalized_question != question:
            print(f'(read as: "{normalized_question}")')
        question = normalized_question

    rewrite_texts: list[str] = []
    interpreted_idx = 0
    gen = None
    # --cascade replaces the --rewrites fusion path entirely (ignored below),
    # but --llm-correct still runs here exactly as it does today - the cascade
    # only ever sees the (possibly corrected) `question`, the same as any
    # other retrieval path; it gets no special wiring of its own.
    if args.llm_correct or (args.rewrites > 0 and not args.cascade):
        # Deliberate reorder (see module docstring): the generator has to run
        # before retrieval to produce the rewrites and/or the corrected text,
        # so it starts here - before the embedder/reranker and before the
        # gate - on every such ask.
        gen = Generator()
        if not gen.health():
            print("generator not running: ./scripts/servers.sh start generator", file=sys.stderr)
            return 2
    if args.llm_correct:
        # Runs on the already-normalized text (--normalize runs first, above) -
        # the corrected text then feeds retrieval, including rewrites/fusion,
        # and generation (phase 11 R4a'; blk_normalize_spell_api's wiring
        # pattern, extended).
        corrected_question = gen.correct(question)
        if corrected_question != question:
            print(f'(read as: "{corrected_question}")')
        question = corrected_question
    if args.rewrites > 0 and not args.cascade:
        rewrite_texts = gen.rewrites(question, n=args.rewrites, style=args.rewrite_style)

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
    if args.question_vectors > 0 and not store.has_qvec(db):
        print(f"{args.db} has no question vectors: build them with scripts/build_qvec.py",
              file=sys.stderr)
        return 2
    # Only passed when set, so a default ask constructs Retriever exactly as before.
    route_kw = {"route": args.route} if args.route else {}
    r = Retriever(db, embedder=emb, reranker=rr, mode="dense",
                  candidates=args.candidates, domain=args.domain,
                  question_vectors=args.question_vectors, **route_kw)

    if args.cascade and not args.retrieve_only:
        # --retrieve-only stays generator-free (module docstring); --cascade
        # is a no-op there and falls through to the plain path below.
        if args.act and args.gate == GATE:
            args.gate = GATE_ACT
        if gen is None:
            gen = Generator()
            if not gen.health():
                print("generator not running: ./scripts/servers.sh start generator",
                      file=sys.stderr)
                return 2

        def answer_fn(q: str, hits: list) -> str:
            return gen.answer(q, hits, cite_grammar=not args.no_grammar)

        cres = cascade.run_cascade(r, gen, question, answer_fn,
                                   0.0 if args.no_gate else args.gate, k=args.k)
        hits = expand(db, cres["hits"], span=args.expand) if args.expand else cres["hits"]

        if cres["tier"] > 0:
            shown = ", ".join(f'"{t}"' for t in cres["rewrites"])
            print(f"(found on retry {cres['tier']}: {shown})")

        if args.show_context:
            for i, h in enumerate(hits, 1):
                head = h["text"].splitlines()[0][:66] if h["text"] else ""
                print(f"[{i}] {h.get('rerank_score', h['score']):.4f}  {h['chunk_id']:24} {head}")
            print()

        if cres["gated"]:
            print(f"{grammar.REFUSAL}\n\n(gate: best evidence scored {cres['score']:.3f}, "
                  f"below {args.gate:.3f} - the model was not asked)")
            return 0

        # --act's tool-calling path is deliberately not wired into the
        # cascade (design note): --cascade always answers in plain reader
        # mode, whatever --act says.
        print(cres["answer"])
        print("\nsources: " + ", ".join(f"[{i}] {h['doc_id']}" for i, h in enumerate(hits, 1)))
        return 0

    if args.rewrites > 0:
        hits, interpreted_idx, _variants, gate_hits = r.retrieve_fused(
            question, rewrites=rewrite_texts, k=args.k)
    else:
        hits = r.retrieve(question, k=args.k)
        gate_hits = hits  # --rewrites 0: identical list, gate on exactly what it always has
    # Gate on the ORIGINAL question's own reranked top-1, never the fused one:
    # after rrf() the fused list is ordered by rank, so its top-1 rerank_score is
    # whichever variant happened to win, not the distribution GATE/GATE_ACT were
    # swept against (measured, tsk_20260828_a47f0496 - see retrieve_fused).
    score = gate_score(gate_hits)
    if args.act and args.gate == GATE:
        args.gate = GATE_ACT
    if args.expand:
        hits = expand(db, hits, span=args.expand)

    if interpreted_idx != 0 and rewrite_texts:
        print(f'interpreted as: "{rewrite_texts[interpreted_idx - 1]}"')

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

    if gen is None:
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
