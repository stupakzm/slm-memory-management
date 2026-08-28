#!/usr/bin/env python3
"""Choose the abstention threshold, on the labelled set, the way Finding 04 says to.

The gate is the phase 2 answer to phase 1's worst number: when retrieval quietly
returned nothing useful, the model invented an answer 56.5% of the time. A gate can
only fix that if it fires on exactly those questions, so this sweeps a threshold on
the pipeline's top-1 score and reports four things, not one:

  abstention recall     unanswerable questions correctly refused - the requirement
  coverage              answerable questions that still reach the model
  coverage w/ evidence  answerable questions *whose answer was actually retrieved*
                        that still reach the model
  silent failures       answerable questions whose answer was NOT retrieved that
                        reach the model anyway - phase 1's 56.5% case

The third is the one to hold fixed while maximising the first. Plain coverage
punishes the gate for refusing a question whose retrieval had already failed, but
refusing that question is the correct action: the model was never going to see the
answer, so letting it through buys a fabrication, not a hit.

There is a second mode. `--answers RUN` sweeps an *answer* run instead, and it is
the one that should decide the threshold. The gate only chooses whether to invoke
the model; it never changes what the model writes. So every threshold can be
replayed exactly from a single ungated generation run - gated questions become
refusals, the rest keep the answer they already produced - and the operating point
is chosen on the numbers that matter (accuracy, abstention, unsupported answers)
rather than on a retrieval proxy for them. Zero extra GPU time.

Usage: sweep_gate.py phase2-dense-rerank [--target 0.90]
       sweep_gate.py --answers phase2-rerank-grammar
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "eval" / "results"


def load(name: str) -> list[dict]:
    p = RESULTS / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"no such run: {p}")
    return json.loads(p.read_text())["per_question"]


def operating_points(per_q: list[dict], k: int) -> list[dict]:
    ans = [p for p in per_q if p["kind"] == "answerable"]
    una = [p for p in per_q if p["kind"] == "unanswerable"]
    # An answerable question "has evidence" when the answer chunk is inside the k the
    # model would actually be handed - not merely somewhere in the ranking.
    withev = [p for p in ans if p.get("first_hit_rank") and p["first_hit_rank"] <= k]
    noev = [p for p in ans if p not in withev]

    # Candidate thresholds come from the data: every observed score, plus a point
    # just below it, so no achievable operating point is missed by a fixed grid.
    scores = sorted({p["top_score"] for p in per_q})
    cands = [scores[0] - 1e-6] + [s + 1e-9 for s in scores]

    out = []
    for t in cands:
        out.append({
            "threshold": t,
            "abstention_recall": sum(1 for p in una if p["top_score"] < t) / max(len(una), 1),
            "coverage": sum(1 for p in ans if p["top_score"] >= t) / max(len(ans), 1),
            "coverage_with_evidence": sum(1 for p in withev if p["top_score"] >= t) / max(len(withev), 1),
            "silent_failures": sum(1 for p in noev if p["top_score"] >= t) / max(len(ans), 1),
        })
    return out


def answer_points(results: list[dict]) -> list[dict]:
    """Replay every threshold against a recorded ungated generation run."""
    ans = [r for r in results if r["kind"] == "answerable"]
    una = [r for r in results if r["kind"] == "unanswerable"]
    withev = [r for r in ans if r["evidence_retrieved"]]
    noev = [r for r in ans if not r["evidence_retrieved"]]
    scores = sorted({r["top_score"] for r in results})
    out = []
    for t in [scores[0] - 1e-6] + [s + 1e-9 for s in scores]:
        gated = lambda r: r["top_score"] < t          # noqa: E731
        spoke_blind = sum(1 for r in noev if not gated(r) and not r["abstained"])
        una_spoke = sum(1 for r in una if not gated(r) and not r["abstained"])
        out.append({
            "threshold": t,
            "accuracy": sum(1 for r in ans if not gated(r) and r["correct"]) / max(len(ans), 1),
            "abstention_recall": (len(una) - una_spoke) / max(len(una), 1),
            "false_abstention": sum(1 for r in withev if gated(r) or r["abstained"]) / max(len(withev), 1),
            "spoke_blind": spoke_blind / max(len(noev), 1),
            "unsupported": (spoke_blind + una_spoke) / max(len(results), 1),
            "gate_fired": sum(1 for r in results if gated(r)) / max(len(results), 1),
        })
    return out


def sweep_answers(name: str, max_accuracy_loss: float) -> None:
    d = json.loads((RESULTS / f"{name}-answers.json").read_text())
    if d.get("config", {}).get("gate"):
        raise SystemExit(f"{name} already ran with a gate; sweep an ungated run")
    pts = answer_points(d["results"])
    ungated = pts[0]
    print(f"\n=== {name} ===  end-to-end, replayed over {len(pts)} thresholds")
    print(f"  ungated: accuracy {ungated['accuracy']:.1%}  abstention "
          f"{ungated['abstention_recall']:.1%}  unsupported {ungated['unsupported']:.1%}")
    print("\n  threshold  accuracy  abstain  false-abst  spoke-blind  unsupported  fired")
    step = max(len(pts) // 14, 1)
    for i, p in enumerate(pts):
        if i % step and i != len(pts) - 1:
            continue
        print(f"  {p['threshold']:9.4f}  {p['accuracy']:8.1%}  {p['abstention_recall']:7.1%}"
              f"  {p['false_abstention']:10.1%}  {p['spoke_blind']:11.1%}"
              f"  {p['unsupported']:11.1%}  {p['gate_fired']:5.1%}")

    # The gate exists to cut unsupported answers. Buying that with unlimited accuracy
    # is not a trade this project wants, so accuracy is the constraint and unsupported
    # is the objective - not the other way round.
    floor = ungated["accuracy"] - max_accuracy_loss
    best = min((p for p in pts if p["accuracy"] >= floor),
               key=lambda p: (p["unsupported"], -p["abstention_recall"]))
    print(f"\n  chosen (accuracy >= {floor:.1%}, i.e. at most {max_accuracy_loss:.0%} below ungated)")
    print(f"    threshold {best['threshold']:.4f}  accuracy {best['accuracy']:.1%}"
          f"  abstention {best['abstention_recall']:.1%}"
          f"  unsupported {best['unsupported']:.1%}  gate fires {best['gate_fired']:.1%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--answers", nargs="+", default=[],
                    help="sweep end-to-end outcomes of an ungated generation run")
    ap.add_argument("--max-accuracy-loss", type=float, default=0.02)
    ap.add_argument("-k", type=int, default=5, help="chunks the model would be handed")
    ap.add_argument("--target", type=float, default=0.90,
                    help="coverage-with-evidence to hold while maximising abstention")
    args = ap.parse_args()

    for name in args.answers:
        sweep_answers(name, args.max_accuracy_loss)

    for name in args.runs:
        per_q = load(name)
        pts = operating_points(per_q, args.k)
        una = sum(1 for p in per_q if p["kind"] == "unanswerable")
        ans = sum(1 for p in per_q if p["kind"] == "answerable")
        print(f"\n=== {name} ===  {ans} answerable, {una} unanswerable, gate on top-1 @k={args.k}")

        print("\n  threshold   abstain  coverage  cov w/ev  silent-fail")
        # A readable slice of the curve rather than all 166 rows.
        shown, step = set(), max(len(pts) // 12, 1)
        for i, p in enumerate(pts):
            if i % step and i != len(pts) - 1:
                continue
            shown.add(i)
            print(f"  {p['threshold']:9.4f}  {p['abstention_recall']:7.1%}  {p['coverage']:8.1%}"
                  f"  {p['coverage_with_evidence']:8.1%}  {p['silent_failures']:10.1%}")

        best = max((p for p in pts if p["coverage_with_evidence"] >= args.target),
                   key=lambda p: (p["abstention_recall"], -p["silent_failures"]), default=None)
        naive = max((p for p in pts if p["coverage"] >= args.target),
                    key=lambda p: p["abstention_recall"], default=None)
        if best:
            print(f"\n  chosen  threshold {best['threshold']:.4f}  "
                  f"abstention {best['abstention_recall']:.1%}  "
                  f"coverage {best['coverage']:.1%}  "
                  f"cov w/ev {best['coverage_with_evidence']:.1%}  "
                  f"silent {best['silent_failures']:.1%}")
        if naive:
            print(f"  (phase 1's metric - plain coverage >= {args.target:.0%}: "
                  f"threshold {naive['threshold']:.4f}, abstention {naive['abstention_recall']:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
