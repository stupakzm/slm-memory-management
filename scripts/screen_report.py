#!/usr/bin/env python3
"""Score a screen from committed `data/eval/results/<name>-answers.json`.

Each --arm is paired against --control on one --eval file. Per arm: answerable
correct, lost, gained, net, an exact two-sided sign-test p over the discordant
pairs, evidence, abstention, abstention net, and the pre-registered verdict
(PASS iff net >= --min-net, lost <= --max-lost and abstention net >=
--min-abstention-net).

`correct` is recomputed from the stored answer text with today's aliases and
never trusted from the file (same principle as robustness_report.py);
evidence_retrieved and abstained are trusted as stored. --json writes the same
data plus the sha256 of every input so a frozen result can be replayed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import gold  # noqa: E402

RESULTS = ROOT / "data" / "eval" / "results"


def sign_test_p(g: int, l: int) -> float:
    n = g + l
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(max(g, l), n + 1)) / 2 ** n)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signed(n: int) -> str:
    return f"{n:+d}"


def load_eval(path) -> dict:
    rows = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["qid"]] = r
    return rows


def score_run(answers_path, eval_rows: dict, aliases: dict) -> dict:
    """{qid: {correct, evidence, abstained, answerable}} for one answers file."""
    out = {}
    for row in json.loads(Path(answers_path).read_text())["results"]:
        qid = row["qid"]
        if qid not in eval_rows:
            raise ValueError(f"{answers_path}: qid {qid} is not in the eval file")
        er = eval_rows[qid]
        al = gold.aliases_for(aliases, qid,
                              er.get("variant_of") or er.get("paraphrase_of"))
        correct = (not row["abstained"]) and gold.is_correct(
            row["answer"], er.get("answer_contains") or [], al)
        out[qid] = {"correct": bool(correct),
                    "evidence": bool(row.get("evidence_retrieved")),
                    "abstained": bool(row["abstained"]),
                    "answerable": er["kind"] == "answerable"}
    return out


def summarize(run: dict) -> dict:
    ans = [r for r in run.values() if r["answerable"]]
    una = [r for r in run.values() if not r["answerable"]]
    return {"answerable": len(ans), "unanswerable": len(una),
            "correct": sum(r["correct"] for r in ans),
            "evidence": sum(r["evidence"] for r in ans),
            "abstained": sum(r["abstained"] for r in una)}


def compare(ctl: dict, arm: dict, min_net=4, max_lost=1, min_abstention_net=0) -> dict:
    if set(ctl) != set(arm):
        raise ValueError("control and arm do not have the same qid set")
    lost = sorted(q for q in ctl if ctl[q]["answerable"]
                  and ctl[q]["correct"] and not arm[q]["correct"])
    gained = sorted(q for q in ctl if ctl[q]["answerable"]
                    and arm[q]["correct"] and not ctl[q]["correct"])
    abst_net = (sum(1 for q in ctl if not ctl[q]["answerable"]
                    and arm[q]["abstained"] and not ctl[q]["abstained"])
                - sum(1 for q in ctl if not ctl[q]["answerable"]
                      and ctl[q]["abstained"] and not arm[q]["abstained"]))
    net = len(gained) - len(lost)
    ok = net >= min_net and len(lost) <= max_lost and abst_net >= min_abstention_net
    return {**summarize(arm), "lost": len(lost), "gained": len(gained),
            "net": net, "p": sign_test_p(len(gained), len(lost)),
            "abstention_net": abst_net, "verdict": "PASS" if ok else "FAIL",
            "lost_qids": lost, "gained_qids": gained}


def format_report(control: str, cs: dict, arms: list) -> str:
    A, U = cs["answerable"], cs["unanswerable"]
    lines = [f"control {control}: correct {cs['correct']}/{A} "
             f"evidence {cs['evidence']}/{A} abstained {cs['abstained']}/{U}"]
    for name, r in arms:
        lines.append(
            f"{name} vs {control}: correct {r['correct']}/{r['answerable']} "
            f"lost {r['lost']} gained {r['gained']} net {signed(r['net'])} "
            f"p {r['p']:.3g} evidence {r['evidence']}/{r['answerable']} "
            f"abstained {r['abstained']}/{r['unanswerable']} "
            f"abstention_net {signed(r['abstention_net'])} verdict {r['verdict']}")
        if r["lost"]:
            lines.append("  lost: " + " ".join(r["lost_qids"]))
        if r["gained"]:
            lines.append("  gained: " + " ".join(r["gained_qids"]))
    return "\n".join(lines) + "\n"


def main(argv=None, results_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--aliases", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--min-net", type=int, default=4)
    ap.add_argument("--max-lost", type=int, default=1)
    ap.add_argument("--min-abstention-net", type=int, default=0)
    a = ap.parse_args(argv)

    rdir = Path(results_dir) if results_dir else RESULTS
    aliases_path = Path(a.aliases) if a.aliases else ROOT / "data" / "eval" / "gold_aliases.json"
    aliases = gold.load_aliases(aliases_path)
    eval_rows = load_eval(a.eval)

    def path_of(name):
        return rdir / f"{name}-answers.json"

    try:
        ctl = score_run(path_of(a.control), eval_rows, aliases)
        arms = [(n, score_run(path_of(n), eval_rows, aliases)) for n in a.arm]
        results = [(n, compare(ctl, run, a.min_net, a.max_lost,
                               a.min_abstention_net)) for n, run in arms]
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    cs = summarize(ctl)
    sys.stdout.write(format_report(a.control, cs, results))

    if a.json:
        hashes = {"eval": sha256(a.eval), a.control: sha256(path_of(a.control))}
        for n in a.arm:
            hashes[n] = sha256(path_of(n))
        data = {"control": {"name": a.control, **cs},
                "arms": {n: r for n, r in results},
                "thresholds": {"min_net": a.min_net, "max_lost": a.max_lost,
                               "min_abstention_net": a.min_abstention_net},
                "eval": str(a.eval),
                "aliases": str(aliases_path),
                "aliases_exists": aliases_path.exists(),
                "sha256": {"eval": hashes["eval"],
                           "aliases": sha256(aliases_path) if aliases_path.exists() else None,
                           "answers": {k: v for k, v in hashes.items() if k != "eval"}}}
        Path(a.json).write_text(json.dumps(data, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
