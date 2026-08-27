#!/usr/bin/env python3
"""Score generated answers: correctness, and the abstention behaviour that is the
whole point of the project.

Phase 1 has no gate - abstention is requested in the prompt only. The research
predicts that works badly at 4B. This measures how badly, which is the number the
phase 2 architectural gate must beat.

Usage: eval_answers.py --name phase1-prefix [--limit 40]
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

from smm import store  # noqa: E402
from smm.embed import Embedder  # noqa: E402
from smm.generate import Generator  # noqa: E402

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
    ap.add_argument("--db", default="data/index/phase1.db")
    ap.add_argument("--eval", default="data/eval/questions.jsonl")
    ap.add_argument("--name", default="answers")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stage", choices=("retrieve", "generate", "both"), default="both",
                    help="6 GB of VRAM does not comfortably hold the embedder and the "
                         "4B generator at once; run the stages separately to keep only "
                         "one model resident")
    args = ap.parse_args()

    cache = ROOT / "data" / "eval" / "results" / f"{args.name}-retrieved.json"

    rows = [json.loads(l) for l in (ROOT / args.eval).open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    # --- stage 1: retrieval only (embedder resident) ---
    if args.stage in ("retrieve", "both"):
        emb = Embedder()
        if not emb.health():
            print("embedder not running: ./scripts/servers.sh start embedder", file=sys.stderr)
            return 2
        db = store.connect(ROOT / args.db)
        retrieved = {}
        t0 = time.time()
        for i, r in enumerate(rows, 1):
            hits = store.search(db, emb.embed_query(r["question"]), k=args.k)
            retrieved[r["qid"]] = hits
            print(f"\r  retrieve {i}/{len(rows)}", end="", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(retrieved))
        print(f"\n  cached retrieval for {len(retrieved)} questions in {time.time()-t0:.0f}s")
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
    for i, r in enumerate(rows, 1):
        hits = retrieved[r["qid"]]
        text = gen.answer(r["question"], hits)
        rec = {
            "qid": r["qid"], "kind": r["kind"], "tags": r["tags"],
            "variant_kind": r.get("variant_kind"),
            "question": r["question"], "answer": text,
            "abstained": abstained(text),
            "retrieved_docs": [h["doc_id"] for h in hits],
            "top_score": hits[0]["score"] if hits else 0.0,
        }
        if r["kind"] == "answerable":
            toks = r["answer_contains"]
            rec["correct"] = all(t in text for t in toks)
            rec["evidence_retrieved"] = any(all(t in h["text"] for t in toks) for h in hits)
        results.append(rec)
        print(f"\r  {i}/{len(rows)}  {(time.time()-t0)/i:.1f}s/q", end="", flush=True)
    print()

    ans = [r for r in results if r["kind"] == "answerable"]
    una = [r for r in results if r["kind"] == "unanswerable"]

    correct = sum(1 for r in ans if r["correct"])
    with_ev = [r for r in ans if r["evidence_retrieved"]]
    correct_given_ev = sum(1 for r in with_ev if r["correct"])
    wrong_abstain = sum(1 for r in ans if r["abstained"])
    right_abstain = sum(1 for r in una if r["abstained"])

    print(f"\n=== {args.name} ===  {len(results)} questions, {time.time()-t0:.0f}s\n")
    print(f"answerable   n={len(ans)}")
    print(f"  answer contains gold token   {correct/len(ans):6.1%}  ({correct}/{len(ans)})")
    print(f"  evidence was retrieved       {len(with_ev)/len(ans):6.1%}")
    print(f"  correct WHEN evidence there  {correct_given_ev/max(len(with_ev),1):6.1%}"
          f"   <- generation quality, retrieval factored out")
    print(f"  wrongly abstained            {wrong_abstain/len(ans):6.1%}  <- coverage lost")
    print(f"\nunanswerable n={len(una)}")
    print(f"  correctly abstained          {right_abstain/max(len(una),1):6.1%}"
          f"  ({right_abstain}/{len(una)})   <- THE core requirement")
    print(f"  hallucinated an answer       {(len(una)-right_abstain)/max(len(una),1):6.1%}")

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
        "answer_accuracy": correct / max(len(ans), 1),
        "accuracy_given_evidence": correct_given_ev / max(len(with_ev), 1),
        "false_abstention": wrong_abstain / max(len(ans), 1),
        "abstention_recall": right_abstain / max(len(una), 1),
        "results": results,
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
