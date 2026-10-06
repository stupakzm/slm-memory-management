#!/usr/bin/env python3
"""Build the blind-judge files for rows whose labels differ between two runs.

Labels are one-token matches plus aliases, so a real difference between two runs
can hide in label noise. A blind judge reads only the rows where a control run
and an arm run were labelled differently (plus hidden controls), and never sees
a run name, qid, pair, score or label. This tool packages those rows; the judge
dispatch is not part of it and scripts/judge_score.py reads the judge's labels.

  --pair CONTROL:ARM   repeatable; answers files are <results>/<NAME>-answers.json
  --out-dir DIR        the blind directory: items-01.jsonl ... and rubric.md
  --key FILE           the key (id -> qid, run, pair, role), written OUTSIDE out-dir

Selection (answerable rows only): a row is selected iff correctness differs
between the two runs (screen_report.score_run). Each selected row contributes
the control answer and the arm answer as items, except a refusal side, which is
not an item (the scorer counts a refusal as wrong). Items with the same
(qid, answer text) are one item with several key entries. --controls rows that
both runs of the first pair answered correctly with evidence are added as hidden
controls (answers from the first pair's control run). All items are shuffled
with --seed, named J001..., and written in batches of --batch-size.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "data" / "eval" / "results"
SECTION_CUT = 6000

RUBRIC = """# Judging rubric

Each item shows a question, a model's answer, the reference answer token(s) and
the text of the gold manual section(s). Read only that. Label every item:

- **CORRECT:** the answer gives a command, option or key that does what the
  question asks, according to the gold section text, even if it is not the
  reference token. Extra correct detail is fine.
- **WRONG:** it does not do what was asked, or it states something the text
  contradicts.
- **UNCLEAR:** the text provided cannot settle it.

Output one JSON line per item, nothing else:

{"id": "J001", "label": "CORRECT|WRONG|UNCLEAR"}
"""


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sr = _load("screen_report", "scripts/screen_report.py")
rh = _load("reembed_headings", "scripts/reembed_headings.py")


def answers_file(results_dir: Path, name: str) -> Path:
    return Path(results_dir) / f"{name}-answers.json"


def load_answers(path) -> dict:
    return {r["qid"]: r for r in json.loads(Path(path).read_text())["results"]}


def gold_ids(eval_rows: dict, qid: str) -> list:
    row = eval_rows[qid]
    ids = row.get("gold_sec_ids")
    if not ids:
        base = row.get("variant_of") or row.get("paraphrase_of")
        ids = (eval_rows.get(base) or {}).get("gold_sec_ids") if base else None
    return list(ids or [])


def gold_sections(eval_rows: dict, qid: str, sections: dict) -> list:
    return [{"sec_id": s, "text": sections[s][:SECTION_CUT]}
            for s in gold_ids(eval_rows, qid) if s in sections]


def build_items(pairs, runs, answers, eval_rows, n_controls, seed):
    """([(qid, answer, [entry, ...])] unshuffled, differing rows, controls added)."""
    items: dict = {}

    def add(qid, answer, entry):
        items.setdefault((qid, answer), []).append(entry)

    differing = 0
    for ctl_name, arm_name in pairs:
        pair = f"{ctl_name}:{arm_name}"
        ctl, arm = runs[ctl_name], runs[arm_name]
        for qid in sorted(set(ctl) & set(arm)):
            if not ctl[qid]["answerable"] or ctl[qid]["correct"] == arm[qid]["correct"]:
                continue
            differing += 1
            for role, name, run in (("control", ctl_name, ctl), ("arm", arm_name, arm)):
                if run[qid]["abstained"]:
                    continue
                add(qid, answers[name][qid]["answer"],
                    {"qid": qid, "run": f"{name}-answers.json", "pair": pair, "role": role})
    controls = 0
    if pairs and n_controls > 0:
        ctl_name, arm_name = pairs[0]
        ctl, arm = runs[ctl_name], runs[arm_name]
        pool = sorted(q for q in set(ctl) & set(arm)
                      if ctl[q]["answerable"] and ctl[q]["correct"] and arm[q]["correct"]
                      and ctl[q]["evidence"] and arm[q]["evidence"])
        picked = random.Random(seed).sample(pool, min(n_controls, len(pool)))
        controls = len(picked)
        for qid in picked:
            add(qid, answers[ctl_name][qid]["answer"],
                {"qid": qid, "run": f"{ctl_name}-answers.json",
                 "pair": f"{ctl_name}:{arm_name}", "role": "hidden_control"})
    return [(q, a, e) for (q, a), e in items.items()], differing, controls


def write_pack(items, eval_rows, sections, out_dir: Path, key_path: Path,
               seed: int, batch_size: int) -> int:
    items = list(items)
    random.Random(seed).shuffle(items)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("items-*.jsonl"):
        old.unlink()
    views, keys = [], []
    for i, (qid, answer, entries) in enumerate(items, 1):
        jid = f"J{i:03d}"
        views.append({"id": jid, "question": eval_rows[qid]["question"], "answer": answer,
                      "reference_answer": eval_rows[qid].get("answer_contains") or [],
                      "gold_sections": gold_sections(eval_rows, qid, sections)})
        keys.append({"id": jid, "entries": entries})
    batches = [views[i:i + batch_size] for i in range(0, len(views), batch_size)]
    for n, batch in enumerate(batches, 1):
        (out_dir / f"items-{n:02d}.jsonl").write_text(
            "".join(json.dumps(v) + "\n" for v in batch))
    (out_dir / "rubric.md").write_text(RUBRIC)
    Path(key_path).write_text("".join(json.dumps(k) + "\n" for k in keys))
    return len(batches)


def main(argv=None, results_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval", action="append", required=True)
    ap.add_argument("--aliases", default=str(ROOT / "data" / "eval" / "gold_aliases_v2.json"))
    ap.add_argument("--pair", action="append", required=True)
    ap.add_argument("--man", default=str(ROOT / "data" / "corpus" / "man.jsonl"))
    ap.add_argument("--md", action="append", default=None)
    ap.add_argument("--controls", type=int, default=20)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--key", required=True)
    a = ap.parse_args(argv)

    rdir = Path(results_dir) if results_dir else RESULTS
    md_specs = a.md or [f"{ROOT / 'data' / 'corpus' / 'domains' / 'emacs'}:emacs"]
    md_dirs = [(Path(d), dom) for d, _, dom in (s.rpartition(":") for s in md_specs)]
    pairs = [tuple(p.split(":", 1)) for p in a.pair]
    if any(len(p) != 2 or not all(p) for p in pairs):
        print("--pair must be CONTROL:ARM", file=sys.stderr)
        return 2
    names = sorted({n for p in pairs for n in p})
    needed = [*map(Path, a.eval), Path(a.aliases), Path(a.man),
              *[d for d, _ in md_dirs], *[answers_file(rdir, n) for n in names]]
    for path in needed:
        if not path.exists():
            print(f"missing input: {path}", file=sys.stderr)
            return 2
    out_dir, key_path = Path(a.out_dir), Path(a.key)
    if out_dir.resolve() in key_path.resolve().parents:
        print(f"--key must be outside --out-dir: {key_path}", file=sys.stderr)
        return 2

    try:
        eval_rows = sr.load_evals(a.eval)
        aliases = sr.gold.load_aliases(a.aliases)
        runs = {n: sr.score_run(answers_file(rdir, n), eval_rows, aliases) for n in names}
        answers = {n: load_answers(answers_file(rdir, n)) for n in names}
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    docs = rh.load_docs(Path(a.man), md_dirs)
    sections = {s["sec_id"]: s["text"] for d in docs.values() for s in d["sections"]}

    items, differing, controls = build_items(pairs, runs, answers, eval_rows,
                                             a.controls, a.seed)
    batches = write_pack(items, eval_rows, sections, out_dir, key_path,
                         a.seed, a.batch_size)
    print(f"items {len(items)} (controls {controls}) from {differing} differing rows "
          f"in {len(pairs)} pair(s); batches {batches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
