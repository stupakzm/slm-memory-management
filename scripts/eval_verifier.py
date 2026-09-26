#!/usr/bin/env python3
"""Score and replay the verifier (smm.verify): a second, independent check that
reads each cited claim against the extract it names.

Two stages, split for the same reason eval_answers.py splits retrieve/generate:
the reranker and the judge (the 4B generator) cannot both be resident, and the
lexical mechanism needs neither.

  --stage score --run NAME --mech lexical|xenc|judge
      Reads NAME-answers.json and NAME-retrieved.json, scores every answered
      (not gated, not abstained) question with the chosen mechanism, and
      merges {qid: {mech: score}} into NAME-verify.json. Each mechanism is a
      separate pass over the same file - a later --mech ADDS a key, it never
      clobbers the ones already there.

  --stage replay --run NAME [--mech ...]
      Offline, no servers. Like sweep_gate.py's `--answers` mode, replays the
      policy "an answered question whose score < t becomes a refusal" against
      every observed threshold, entirely from files already on disk - the
      verifier decides keep-or-refuse and never rewrites what the model wrote,
      so this can be done for every t without spending a token of GPU time.

Usage:
  .venv/bin/python scripts/eval_verifier.py --stage score --run phase2-full --mech lexical
  ./scripts/servers.sh stop && ./scripts/servers.sh start reranker
  .venv/bin/python scripts/eval_verifier.py --stage score --run phase2-full --mech xenc
  .venv/bin/python scripts/eval_verifier.py --stage replay --run phase2-full
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import verify  # noqa: E402

RESULTS = ROOT / "data" / "eval" / "results"


def _load_eval_answers():
    """Import scripts/eval_answers.py by path, the way tests/test_eval_answers.py
    does, so its `unpack_entry` (the *-retrieved.json cache format, see
    blk_eval_cache_format) is reused rather than re-specified here. Kept out of
    the module's top-level imports: eval_answers.py imports smm.store, which
    needs the third-party sqlite_vec extension, and --stage replay must run
    with neither a server nor a real index."""
    spec = importlib.util.spec_from_file_location(
        "eval_answers", ROOT / "scripts" / "eval_answers.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fold_of(qid: str) -> int:
    """Deterministic 2-way fold for the qid's *group* (see group_of): every
    variant of a question - a11, a11.t, a11.n, a11.z - must land in the same
    fold, so zlib.crc32 (not the randomized `hash()`) keys off the base id."""
    base = qid.split(".", 1)[0]
    return zlib.crc32(base.encode()) % 2


def group_of(qid: str) -> str:
    return qid.split(".", 1)[0]


def _is_good(record: dict) -> bool:
    """good = answerable and correct; everything else that was answered (a
    wrong answerable, or an unanswerable question the model spoke on) is bad."""
    return record["kind"] == "answerable" and bool(record.get("correct"))


def replay(records: list[dict], scores: dict, t) -> dict:
    """Apply the policy "an answered question whose score < t becomes a
    refusal; None means keep" over `records` restricted to those present in
    `scores` (the answered, scored population for this mechanism).

    `t` is either a float threshold or the literal string "off" (never
    removes anything - the baseline with no verifier applied at all).

    Returns COUNTS: kept_good, removed_good, kept_bad, removed_bad.
    """
    counts = {"kept_good": 0, "removed_good": 0, "kept_bad": 0, "removed_bad": 0}
    for r in records:
        qid = r["qid"]
        if qid not in scores:
            continue
        s = scores[qid]
        removed = t != "off" and s is not None and s < t
        good = _is_good(r)
        if good:
            counts["removed_good" if removed else "kept_good"] += 1
        else:
            counts["removed_bad" if removed else "kept_bad"] += 1
    return counts


def choose_threshold(records: list[dict], scores: dict, qids) -> tuple:
    """The t (from the distinct observed scores restricted to `qids`, plus
    "off") that maximises (removed_bad - removed_good) over that same subset.
    Ties favour the lower t - "off" behaves like the lowest possible threshold
    (it removes nothing, same as any t at or below the minimum observed
    score), so it is tried first and only beaten by a strictly better t.

    Returns (t, counts_at_t).
    """
    sub_scores = {q: s for q, s in scores.items() if q in qids}
    sub_records = [r for r in records if r["qid"] in qids]
    candidates = ["off"] + sorted({s for s in sub_scores.values() if s is not None})
    best_t, best_counts, best_obj = candidates[0], None, None
    for t in candidates:
        c = replay(sub_records, sub_scores, t)
        obj = c["removed_bad"] - c["removed_good"]
        if best_obj is None or obj > best_obj:
            best_t, best_counts, best_obj = t, c, obj
    return best_t, best_counts


def _sum_counts(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in
            ("kept_good", "removed_good", "kept_bad", "removed_bad")}


def stage_score(args) -> int:
    if not args.mech:
        print("--mech is required for --stage score", file=sys.stderr)
        return 2
    answers_path = RESULTS / f"{args.run}-answers.json"
    retrieved_path = RESULTS / f"{args.run}-retrieved.json"
    if not answers_path.exists() or not retrieved_path.exists():
        print(f"missing {answers_path} or {retrieved_path}", file=sys.stderr)
        return 2

    eval_answers = _load_eval_answers()
    ans = json.loads(answers_path.read_text())
    retrieved = json.loads(retrieved_path.read_text())
    records = ans["results"]
    if args.limit:
        records = records[: args.limit]

    if args.mech == "lexical":
        scorer = verify.lexical
    elif args.mech == "xenc":
        from smm.rerank import Reranker
        rr = Reranker(args.rerank_url) if args.rerank_url else Reranker()
        if not rr.health():
            print(f"reranker not running at {rr.url}: "
                  f"./scripts/servers.sh start reranker", file=sys.stderr)
            return 2
        scorer = verify.CrossEncoderScorer(rr)
    else:
        from smm.generate import Generator
        gen = Generator(args.gen_url) if args.gen_url else Generator()
        if not gen.health():
            print(f"generator not running at {gen.url}: "
                  f"./scripts/servers.sh start generator", file=sys.stderr)
            return 2
        scorer = verify.JudgeScorer(gen)

    scores = {}
    for r in records:
        if r.get("gated") or r.get("abstained"):
            continue
        hits, _gate_hits = eval_answers.unpack_entry(retrieved[r["qid"]])
        scores[r["qid"]] = verify.score_answer(r["answer"], hits, scorer)

    verify_path = RESULTS / f"{args.run}-verify.json"
    verify_path.parent.mkdir(parents=True, exist_ok=True)
    merged = json.loads(verify_path.read_text()) if verify_path.exists() else {}
    for qid, s in scores.items():
        merged.setdefault(qid, {})[args.mech] = s
    verify_path.write_text(json.dumps(merged, indent=2))
    n_scored = sum(1 for s in scores.values() if s is not None)
    print(f"scored {len(scores)} answered questions with --mech {args.mech} "
          f"({n_scored} with an opinion, {len(scores) - n_scored} None) -> {verify_path}")
    return 0


def stage_replay(args) -> int:
    answers_path = RESULTS / f"{args.run}-answers.json"
    verify_path = RESULTS / f"{args.run}-verify.json"
    if not answers_path.exists() or not verify_path.exists():
        print(f"missing {answers_path} or {verify_path}", file=sys.stderr)
        return 2

    ans = json.loads(answers_path.read_text())
    records = ans["results"]
    verify_scores = json.loads(verify_path.read_text())

    mechs = [args.mech] if args.mech else sorted(
        {m for v in verify_scores.values() for m in v})

    groups: dict[str, list[str]] = defaultdict(list)
    for r in records:
        groups[group_of(r["qid"])].append(r["qid"])
    fold_qids = {0: set(), 1: set()}
    for base, qids in groups.items():
        fold_qids[fold_of(base)].update(qids)

    report = {"run": args.run, "mechanisms": {}}

    for mech in mechs:
        scores = {qid: v[mech] for qid, v in verify_scores.items() if mech in v}
        all_qids = set(scores.keys())

        in_sample_t, in_sample_counts = choose_threshold(records, scores, all_qids)

        held_out_sum = {"kept_good": 0, "removed_good": 0, "kept_bad": 0, "removed_bad": 0}
        folds = {}
        removed_qids: set[str] = set()
        for train_fold in (0, 1):
            test_fold = 1 - train_fold
            train_qids = fold_qids[train_fold] & all_qids
            test_qids = fold_qids[test_fold] & all_qids
            t, _train_counts = choose_threshold(records, scores, train_qids)
            test_records = [r for r in records if r["qid"] in test_qids]
            test_counts = replay(test_records, scores, t)
            held_out_sum = _sum_counts(held_out_sum, test_counts)
            folds[train_fold] = {"trained_on_fold": train_fold, "applied_to_fold": test_fold,
                                  "threshold": t, "held_out_counts": test_counts}
            if t != "off":
                for qid in test_qids:
                    s = scores.get(qid)
                    if s is not None and s < t:
                        removed_qids.add(qid)

        by_qid = {r["qid"]: r for r in records}

        def still_answered(qid: str) -> bool:
            r = by_qid[qid]
            return not r["abstained"] and qid not in removed_qids

        ans_recs = [r for r in records if r["kind"] == "answerable"]
        una_recs = [r for r in records if r["kind"] == "unanswerable"]
        correct_final = sum(1 for r in ans_recs
                             if r["correct"] and r["qid"] not in removed_qids)
        abstained_final = sum(1 for r in una_recs
                               if r["abstained"] or r["qid"] in removed_qids)
        no_ev_answered = sum(1 for r in ans_recs
                              if not r["evidence_retrieved"] and still_answered(r["qid"]))
        none_coverage = sum(1 for s in scores.values() if s is None)

        end_to_end = {
            "correct_over_answerable": [correct_final, len(ans_recs)],
            "unanswerable_correctly_abstained": [abstained_final, len(una_recs)],
            "answered_with_no_evidence": [no_ev_answered, len(ans_recs)],
            "none_coverage": [none_coverage, len(all_qids)],
        }

        report["mechanisms"][mech] = {
            "in_sample": {"threshold": in_sample_t, "counts": in_sample_counts},
            "folds": folds,
            "held_out_sum": held_out_sum,
            "end_to_end": end_to_end,
        }

        print(f"\n=== {args.run} :: {mech} ===  {len(all_qids)} scored questions")
        print(f"  in-sample     t={in_sample_t!r:>8}  {in_sample_counts}")
        for tf in (0, 1):
            f = folds[tf]
            print(f"  fold {tf}->fold{f['applied_to_fold']}  t={f['threshold']!r:>8}  "
                  f"{f['held_out_counts']}")
        print(f"  held-out sum              {held_out_sum}")
        print(f"  correct/answerable            {correct_final}/{len(ans_recs)}")
        print(f"  unanswerable correctly abstained  {abstained_final}/{len(una_recs)}")
        print(f"  answered with no evidence      {no_ev_answered}/{len(ans_recs)}")
        print(f"  no-opinion (None) coverage     {none_coverage}/{len(all_qids)}")

    out = RESULTS / f"{args.run}-verify-replay.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("score", "replay"), required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--mech", choices=("lexical", "xenc", "judge"), default=None,
                    help="--stage score: required, which mechanism to add. "
                         "--stage replay: optional filter; default is every "
                         "mechanism present in NAME-verify.json")
    ap.add_argument("--rerank-url", default=None, help="--mech xenc only")
    ap.add_argument("--gen-url", default=None, help="--mech judge only")
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on the first N records")
    args = ap.parse_args()

    if args.stage == "score":
        return stage_score(args)
    return stage_replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
