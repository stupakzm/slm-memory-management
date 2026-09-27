#!/usr/bin/env python3
"""Offline rescoring (tsk_20260926_a51d0707): for every existing
data/eval/results/*-answers.json, recompute 'correct' both strict and
aliased from the stored answer text alone - no server, no re-generation,
and the input files are never modified.

Reports, per run, the qids that flip from strict-wrong to aliased-right (the
noise gold aliases exist to fix) and the reverse (which must never happen:
aliasing only widens what counts as correct, so a strict-correct answer can
never become aliased-wrong). The reverse is asserted empty as well as
reported, so a bug in gold.is_correct fails this script loudly rather than
silently shipping a worse label.

R1 (tsk_20260927_keyalias): tokens come from every --eval passed (repeatable),
defaulting to both data/eval/questions.jsonl (man pages) and
data/eval/emacs_questions.jsonl (Emacs) - so the p11 Emacs runs are scored
too, not silently skipped for having no tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import gold  # noqa: E402

RESULTS = ROOT / "data" / "eval" / "results"
DEFAULT_EVALS = [
    "data/eval/questions.jsonl",
    "data/eval/emacs_questions.jsonl",
    "data/eval/variations_typo.jsonl",
]


def _tokens_by_qid(eval_paths: list[str]) -> tuple[dict, dict]:
    """(tokens_by_qid, base_by_qid): tokens_by_qid holds answerable rows'
    answer_contains; base_by_qid holds every row's variant_of/paraphrase_of
    (tsk_20260927_typos), used for the gold.aliases_for fallback below."""
    tokens, base = {}, {}
    for rel in eval_paths:
        path = ROOT / rel
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            base[row["qid"]] = row.get("variant_of") or row.get("paraphrase_of")
            if row.get("kind") == "answerable":
                tokens[row["qid"]] = row["answer_contains"]
    return tokens, base


def rescore_run(data: dict, tokens_by_qid: dict, aliases: dict, base_by_qid: dict) -> dict:
    ans = [r for r in data.get("results", []) if r["kind"] == "answerable"]
    strict_correct = aliased_correct = n = 0
    flipped_wrong_to_right: list[str] = []
    flipped_right_to_wrong: list[str] = []
    for r in ans:
        toks = tokens_by_qid.get(r["qid"])
        if toks is None:
            continue
        n += 1
        strict = gold.is_correct(r["answer"], toks, {})
        row_aliases = gold.aliases_for(aliases, r["qid"], base_by_qid.get(r["qid"]))
        aliased = gold.is_correct(r["answer"], toks, row_aliases)
        strict_correct += strict
        aliased_correct += aliased
        if aliased and not strict:
            flipped_wrong_to_right.append(r["qid"])
        if strict and not aliased:
            flipped_right_to_wrong.append(r["qid"])

    assert not flipped_right_to_wrong, (
        f"aliasing turned strict-correct answers wrong, which must never "
        f"happen: {flipped_right_to_wrong}")

    return {
        "answerable": n,
        "strict_correct": strict_correct,
        "aliased_correct": aliased_correct,
        "flipped_wrong_to_right": flipped_wrong_to_right,
        "flipped_right_to_wrong": flipped_right_to_wrong,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="append", default=None,
                     help="questions jsonl to pull answerable tokens from "
                          "(repeatable; default: man, Emacs and the typo "
                          "variant pool)")
    args = ap.parse_args()
    eval_paths = args.eval or DEFAULT_EVALS

    aliases = gold.load_aliases(ROOT / "data" / "eval" / "gold_aliases.json")
    tokens_by_qid, base_by_qid = _tokens_by_qid(eval_paths)

    out = {}
    for path in sorted(RESULTS.glob("*-answers.json")):
        data = json.loads(path.read_text())
        if "results" not in data:
            continue
        out[path.stem] = rescore_run(data, tokens_by_qid, aliases, base_by_qid)

    out_path = RESULTS / "gold-aliases-rescore.json"
    out_path.write_text(json.dumps(out, indent=2))

    for name, r in out.items():
        print(f"{name}: strict {r['strict_correct']}/{r['answerable']}  "
              f"aliased {r['aliased_correct']}/{r['answerable']}  "
              f"flipped {len(r['flipped_wrong_to_right'])} wrong->right, "
              f"{len(r['flipped_right_to_wrong'])} right->wrong")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
