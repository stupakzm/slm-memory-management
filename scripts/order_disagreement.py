#!/usr/bin/env python3
"""Count how often a second extract order changes the answer, per first-run bucket.

`--a NAME` and `--b NAME` are answers files `<results>/<NAME>-answers.json` for
the same rows with the same extracts in two orders (A first, B second). A's
answered rows are split into correct / wrong / unanswerable-answered (rows A
abstained are excluded). A qid disagrees when B abstained or B names a different
set of identifiers; when neither names any, when the normalised texts differ.

Correctness comes only from screen_report.score_run. --json writes the counts
plus the sha256 of every input file read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import gold  # noqa: E402


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sr = _load("screen_report", "screen_report.py")
gr = _load("grounding_report", "grounding_report.py")

RESULTS = ROOT / "data" / "eval" / "results"
BUCKETS = ("correct", "wrong", "unanswerable-answered")


def norm_text(answer: str) -> str:
    text = " ".join(gr.strip_citations(answer).lower().split())
    return text.rstrip(".")


def disagrees(a_row: dict, b_row: dict) -> bool:
    if b_row["abstained"]:
        return True
    ia = gr.tight_idents(a_row["answer"])
    ib = gr.tight_idents(b_row["answer"])
    if not ia and not ib:
        return norm_text(a_row["answer"]) != norm_text(b_row["answer"])
    return ia != ib


def bucket_of(score: dict) -> str:
    if not score["answerable"]:
        return "unanswerable-answered"
    return "correct" if score["correct"] else "wrong"


def count(a_rows: dict, b_rows: dict, scores: dict) -> dict:
    counts = {b: {"n": 0, "disagree": 0} for b in BUCKETS}
    for qid, row in a_rows.items():
        s = scores[qid]
        if s["abstained"]:
            continue
        c = counts[bucket_of(s)]
        c["n"] += 1
        c["disagree"] += int(disagrees(row, b_rows[qid]))
    return counts


def load_rows(path) -> dict:
    return {r["qid"]: r for r in json.loads(Path(path).read_text())["results"]}


def main(argv=None, results_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval", action="append", required=True)
    ap.add_argument("--aliases", default=None)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    rdir = Path(results_dir) if results_dir else RESULTS
    aliases_path = Path(a.aliases) if a.aliases else ROOT / "data" / "eval" / "gold_aliases_v2.json"
    a_path = rdir / f"{a.a}-answers.json"
    b_path = rdir / f"{a.b}-answers.json"
    for p in (a_path, b_path):
        if not p.exists():
            print(str(p), file=sys.stderr)
            return 2
    aliases = gold.load_aliases(aliases_path)
    try:
        eval_rows = sr.load_evals(a.eval)
        scores = sr.score_run(a_path, eval_rows, aliases)
        sr.score_run(b_path, eval_rows, aliases)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    a_rows, b_rows = load_rows(a_path), load_rows(b_path)
    if set(a_rows) != set(b_rows):
        diff = sorted(set(a_rows) ^ set(b_rows))
        print(f"qid sets differ between {a.a} and {a.b} (e.g. {diff[0]})", file=sys.stderr)
        return 2

    counts = count(a_rows, b_rows, scores)
    n = sum(c["n"] for c in counts.values())
    parts = " ".join(f"{b} {counts[b]['disagree']}/{counts[b]['n']}" for b in BUCKETS)
    print(f"{a.a} vs {a.b}: answered {n} {parts}")
    print(f"{a.a} vs {a.b}: refusing every disagreement would lose "
          f"{counts['correct']['disagree']} correct answers and catch "
          f"{counts['wrong']['disagree']} wrong and "
          f"{counts['unanswerable-answered']['disagree']} unanswerable-answered rows")

    if a.json:
        out = {"counts": counts,
               "sha256": {"eval": sr.sha256(a.eval[0]) if len(a.eval) == 1
                          else {str(e): sr.sha256(e) for e in a.eval},
                          "aliases": sr.sha256(aliases_path) if aliases_path.exists() else None,
                          "a": sr.sha256(a_path), "b": sr.sha256(b_path)}}
        Path(a.json).write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
