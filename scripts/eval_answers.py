#!/usr/bin/env python3
"""Score generated answers: correctness, and the abstention behaviour that is the
whole point of the project.

Phase 1 had no gate and no grammar - abstention was requested in the prompt and
enforced nowhere - and the result was the project's worst number: when retrieval
found nothing useful, the model invented an answer 56.5% of the time, sometimes
with a fabricated citation attached. Phase 2 adds two mechanisms and this script
turns each on independently, because a bundle that improves the headline tells you
nothing about which half did it:

  --gate T     refuse before generation when the top-1 score is below T; the model
               is never invoked, so it cannot speculate
  --grammar    constrained decoding: every claim carries an in-range citation

Two stages, because 6 GB of VRAM does not hold the embedder, the reranker and the
4B generator at once.

  .venv/bin/python scripts/eval_answers.py --stage retrieve --mode dense --rerank --name phase2-full
  ./scripts/servers.sh stop && ./scripts/servers.sh start generator
  .venv/bin/python scripts/eval_answers.py --stage generate --name phase2-full --gate 0.65 --grammar
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import grammar, lexical, store  # noqa: E402
from smm.embed import Embedder  # noqa: E402
from smm.generate import Generator  # noqa: E402
from smm.rerank import Reranker  # noqa: E402
from smm.retrieve import Retriever, gate_score  # noqa: E402

ABSTAIN_RE = re.compile(
    r"\bi\s*(?:do\s*n[o']?t|don'?t)\s+know\b"
    r"|\bnot\s+(?:in|contained\s+in|found\s+in|covered\s+by)\s+the\s+extracts?\b"
    r"|\bthe\s+extracts?\s+do\s*(?:es)?\s*n[o']?t\s+contain\b"
    r"|\bno\s+(?:relevant\s+)?information\b",
    re.I,
)


def abstained(text: str) -> bool:
    return bool(ABSTAIN_RE.search(text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/index/phase2.db")
    ap.add_argument("--eval", default="data/eval/questions.jsonl")
    ap.add_argument("--name", default="answers")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode", choices=("dense", "bm25", "hybrid"), default="dense")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--candidates", type=int, default=50)
    ap.add_argument("--gate", type=float, default=0.0,
                    help="refuse before generation below this top-1 score (0 = no gate)")
    ap.add_argument("--grammar", action="store_true", help="GBNF-enforced citations")
    ap.add_argument("--stage", choices=("retrieve", "generate", "both"), default="both")
    args = ap.parse_args()

    cache = ROOT / "data" / "eval" / "results" / f"{args.name}-retrieved.json"
    rows = [json.loads(l) for l in (ROOT / args.eval).open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    # --- stage 1: retrieval only (embedder + reranker resident) ---
    if args.stage in ("retrieve", "both"):
        emb = None
        if args.mode in ("dense", "hybrid"):
            emb = Embedder()
            if not emb.health():
                print("embedder not running: ./scripts/servers.sh start embedder", file=sys.stderr)
                return 2
        rr = None
        if args.rerank:
            rr = Reranker()
            if not rr.health():
                print("reranker not running: ./scripts/servers.sh start reranker", file=sys.stderr)
                return 2
        db = store.connect(ROOT / args.db)
        if args.mode in ("bm25", "hybrid") and not lexical.has_index(db):
            print(f"{args.db} has no chunks_fts", file=sys.stderr)
            return 2
        r = Retriever(db, embedder=emb, reranker=rr, mode=args.mode, candidates=args.candidates)
        retrieved, t0 = {}, time.time()
        for i, row in enumerate(rows, 1):
            retrieved[row["qid"]] = r.retrieve(row["question"], k=args.k)
            if sys.stdout.isatty():
                print(f"\r  retrieve {i}/{len(rows)}  {(time.time()-t0)/i:.2f}s/q", end="", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(retrieved))
        print(f"\n  cached retrieval for {len(retrieved)} questions in {time.time()-t0:.0f}s "
              f"(mode={args.mode}, rerank={args.rerank})")
        if args.stage == "retrieve":
            print(f"  -> {cache}\n  now: ./scripts/servers.sh stop && "
                  f"./scripts/servers.sh start generator")
            return 0

    # --- stage 2: generation only (generator resident) ---
    gen = Generator()
    if not gen.health():
        print("generator not running: ./scripts/servers.sh start generator", file=sys.stderr)
        return 2
    if not cache.exists():
        print(f"no cached retrieval at {cache}; run --stage retrieve first", file=sys.stderr)
        return 2
    retrieved = json.loads(cache.read_text())

    results, t0 = [], time.time()
    for i, row in enumerate(rows, 1):
        hits = retrieved[row["qid"]]
        score = gate_score(hits)
        # The gate is architectural: below threshold the model is never invoked, so
        # there is no opportunity to speculate. That is the whole of Finding 04.
        gated = bool(args.gate) and score < args.gate
        text = grammar.REFUSAL if gated else gen.answer(row["question"], hits,
                                                        cite_grammar=args.grammar)
        rec = {
            "qid": row["qid"], "kind": row["kind"], "tags": row["tags"],
            "variant_kind": row.get("variant_kind"),
            "question": row["question"], "answer": text,
            "abstained": gated or abstained(text),
            "gated": gated,
            "retrieved_docs": [h["doc_id"] for h in hits],
            "top_score": score,
        }
        if row["kind"] == "answerable":
            toks = row["answer_contains"]
            rec["correct"] = all(t in text for t in toks)
            rec["evidence_retrieved"] = any(all(t in h["text"] for t in toks) for h in hits)
            if not gated:
                rec.update(grammar.verify_citations(text, hits, toks))
        results.append(rec)
        if sys.stdout.isatty():
            print(f"\r  {i}/{len(rows)}  {(time.time()-t0)/i:.1f}s/q", end="", flush=True)
    print()

    ans = [r for r in results if r["kind"] == "answerable"]
    una = [r for r in results if r["kind"] == "unanswerable"]

    correct = sum(1 for r in ans if r["correct"])
    with_ev = [r for r in ans if r["evidence_retrieved"]]
    no_ev = [r for r in ans if not r["evidence_retrieved"]]
    correct_given_ev = sum(1 for r in with_ev if r["correct"])
    wrong_abstain = sum(1 for r in with_ev if r["abstained"])
    right_abstain = sum(1 for r in una if r["abstained"])
    spoke_blind = sum(1 for r in no_ev if not r["abstained"])
    unsupported = (spoke_blind + (len(una) - right_abstain)) / max(len(results), 1)

    print(f"\n=== {args.name} ===  {len(results)} questions, {time.time()-t0:.0f}s"
          f"  (gate={args.gate or 'off'}, grammar={args.grammar})\n")
    print(f"answerable   n={len(ans)}")
    print(f"  answer contains gold token   {correct/len(ans):6.1%}  ({correct}/{len(ans)})")
    print(f"  evidence was retrieved       {len(with_ev)/len(ans):6.1%}")
    print(f"  correct WHEN evidence there  {correct_given_ev/max(len(with_ev),1):6.1%}"
          f"   <- generation quality, retrieval factored out")
    print(f"  false abstention (had ev.)   {wrong_abstain/max(len(with_ev),1):6.1%}  <- coverage lost")
    print(f"\nunanswerable n={len(una)}")
    print(f"  correctly abstained          {right_abstain/max(len(una),1):6.1%}"
          f"  ({right_abstain}/{len(una)})   <- THE core requirement")

    print(f"\nspeaking without evidence")
    print(f"  answered with no evidence    {spoke_blind/max(len(no_ev),1):6.1%}"
          f"  ({spoke_blind}/{len(no_ev)})   <- phase 1's worst number")
    print(f"  unsupported, all questions   {unsupported:6.1%}"
          f"  ({spoke_blind + len(una) - right_abstain}/{len(results)})")
    n_gated = sum(1 for r in results if r["gated"])
    if args.gate:
        print(f"  gate fired                   {n_gated/len(results):6.1%}  ({n_gated}/{len(results)})")

    # Only real answers can carry a citation: a refusal is uncited by construction,
    # and counting refusals as uncited would flatter or damn the grammar at random.
    answered = [r for r in ans if not r["gated"] and not r["abstained"]]
    if answered and "n_citations" in answered[0]:
        uncited = sum(1 for r in answered if r.get("uncited"))
        oor = sum(1 for r in answered if r.get("out_of_range", 0))
        supported = sum(1 for r in answered if r.get("cite_supported"))
        print(f"\ncitations    n={len(answered)} answers the model actually wrote")
        print(f"  no citation at all           {uncited/len(answered):6.1%}")
        print(f"  citation out of range        {oor/len(answered):6.1%}")
        print(f"  cited extract supports it    {supported/max(len(answered),1):6.1%}"
              f"   <- the check the grammar exists to enable")

    by_reason = defaultdict(list)
    for r in una:
        for t in r["tags"]:
            if t in ("tool-not-installed", "out-of-corpus", "requires-execution"):
                by_reason[t].append(r)
    print("\n  abstention by reason         n    rate")
    for t, g in sorted(by_reason.items()):
        print(f"    {t:26} {len(g):3}  {sum(1 for x in g if x['abstained'])/len(g):6.1%}")

    outdir = ROOT / "data" / "eval" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{args.name}-answers.json"
    out.write_text(json.dumps({
        "name": args.name, "k": args.k, "n": len(results),
        "config": {"mode": args.mode, "rerank": args.rerank, "gate": args.gate,
                   "grammar": args.grammar, "candidates": args.candidates},
        "answer_accuracy": correct / max(len(ans), 1),
        "accuracy_given_evidence": correct_given_ev / max(len(with_ev), 1),
        "false_abstention_with_evidence": wrong_abstain / max(len(with_ev), 1),
        "abstention_recall": right_abstain / max(len(una), 1),
        "spoke_without_evidence": spoke_blind / max(len(no_ev), 1),
        "unsupported_rate": unsupported,
        "results": results,
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
