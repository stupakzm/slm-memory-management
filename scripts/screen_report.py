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


def load_evals(paths) -> dict:
    """Merge several eval files by qid; a qid in two files is an error."""
    rows = {}
    for path in paths:
        part = load_eval(path)
        dup = sorted(set(rows) & set(part))
        if dup:
            raise ValueError(f"{path}: qid {dup[0]} is already in an earlier --eval file")
        rows.update(part)
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
                    "answerable": er["kind"] == "answerable",
                    "top_score": float(row.get("top_score", float("-inf")))}
    return out


def arm_gate(answers_path) -> float:
    """The gate the answers file ran with (0 if its config has none)."""
    return float((json.loads(Path(answers_path).read_text()).get("config") or {})
                 .get("gate") or 0)


def replay_gate(run: dict, t: float) -> dict:
    """Rows whose top_score is below t become refusals (rows without one never do); evidence is unchanged."""
    return {q: ({**r, "abstained": True, "correct": False}
                if r["top_score"] != float("-inf") and r["top_score"] < t else r)
            for q, r in run.items()}


def match_abstention(ctl: dict, arm: dict, gate: float):
    """(threshold, replayed arm) at the lowest gate >= `gate` whose abstentions on
    unanswerable rows reach the control's; (None, max abstained) if none does."""
    target = summarize(ctl)["abstained"]
    scores = sorted({r["top_score"] + 1e-9 for r in arm.values()
                     if r["top_score"] != float("-inf")})
    cands = [gate] + [t for t in scores if t >= gate]
    for t in cands:
        replayed = replay_gate(arm, t)
        if summarize(replayed)["abstained"] >= target:
            return t, replayed
    return None, summarize(replay_gate(arm, cands[-1]))["abstained"]


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


def group_of(eval_row: dict, field: str, default: str) -> str:
    return eval_row.get(field) or default


def group_compare(ctl: dict, arm: dict, eval_rows: dict, field: str, default: str) -> dict:
    """{group: compare() restricted to that group's rows}."""
    names = sorted({group_of(eval_rows[q], field, default) for q in ctl})
    out = {}
    for g in names:
        qs = [q for q in ctl if group_of(eval_rows[q], field, default) == g]
        out[g] = compare({q: ctl[q] for q in qs}, {q: arm[q] for q in qs})
    return out


def _fields(r: dict) -> str:
    return (f"correct {r['correct']}/{r['answerable']} "
            f"lost {r['lost']} gained {r['gained']} net {signed(r['net'])} "
            f"p {r['p']:.3g} evidence {r['evidence']}/{r['answerable']} "
            f"abstained {r['abstained']}/{r['unanswerable']} "
            f"abstention_net {signed(r['abstention_net'])} verdict {r['verdict']}")


def _group_lines(r: dict, indent: str = "  ") -> list:
    return [indent + f"[{g}] correct {gr['correct']}/{gr['answerable']} "
            f"lost {gr['lost']} gained {gr['gained']} net {signed(gr['net'])} "
            f"p {gr['p']:.3g} evidence {gr['evidence']}/{gr['answerable']} "
            f"abstained {gr['abstained']}/{gr['unanswerable']} "
            f"abstention_net {signed(gr['abstention_net'])}"
            for g, gr in r.get("groups", {}).items()]


def format_report(control: str, cs: dict, arms: list) -> str:
    A, U = cs["answerable"], cs["unanswerable"]
    lines = [f"control {control}: correct {cs['correct']}/{A} "
             f"evidence {cs['evidence']}/{A} abstained {cs['abstained']}/{U}"]
    for name, r in arms:
        lines.append(f"{name} vs {control}: {_fields(r)}")
        if r["lost"]:
            lines.append("  lost: " + " ".join(r["lost_qids"]))
        if r["gained"]:
            lines.append("  gained: " + " ".join(r["gained_qids"]))
        lines.extend(_group_lines(r))
        m = r.get("matched")
        if m is not None:
            if m["threshold"] is None:
                lines.append(f"  matched-abstention: unreachable (max abstained "
                             f"{m['max_abstained']}/{r['unanswerable']})")
            else:
                lines.append(f"  matched-abstention gate {m['threshold']:.4f}: "
                             f"{_fields(m)}")
                lines.extend(_group_lines(m, "    "))
    return "\n".join(lines) + "\n"


def main(argv=None, results_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval", action="append", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--aliases", default=None)
    ap.add_argument("--group", default=None)
    ap.add_argument("--group-default", default="-")
    ap.add_argument("--json", default=None)
    ap.add_argument("--match-abstention", action="store_true")
    ap.add_argument("--min-net", type=int, default=4)
    ap.add_argument("--max-lost", type=int, default=1)
    ap.add_argument("--min-abstention-net", type=int, default=0)
    a = ap.parse_args(argv)

    rdir = Path(results_dir) if results_dir else RESULTS
    aliases_path = Path(a.aliases) if a.aliases else ROOT / "data" / "eval" / "gold_aliases.json"
    aliases = gold.load_aliases(aliases_path)

    def path_of(name):
        return rdir / f"{name}-answers.json"

    try:
        eval_rows = load_evals(a.eval)
        ctl = score_run(path_of(a.control), eval_rows, aliases)
        arms = [(n, score_run(path_of(n), eval_rows, aliases)) for n in a.arm]
        results = [(n, compare(ctl, run, a.min_net, a.max_lost,
                               a.min_abstention_net)) for n, run in arms]
        if a.group:
            for (n, run), (_, r) in zip(arms, results):
                r["groups"] = group_compare(ctl, run, eval_rows, a.group, a.group_default)
        if a.match_abstention:
            for (n, run), (_, r) in zip(arms, results):
                t, got = match_abstention(ctl, run, arm_gate(path_of(n)))
                if t is None:
                    r["matched"] = {"threshold": None, "max_abstained": got}
                    continue
                m = {"threshold": t, **compare(ctl, got, a.min_net, a.max_lost,
                                               a.min_abstention_net)}
                if a.group:
                    m["groups"] = group_compare(ctl, got, eval_rows, a.group,
                                                a.group_default)
                r["matched"] = m
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    cs = summarize(ctl)
    sys.stdout.write(format_report(a.control, cs, results))

    if a.json:
        hashes = {"eval": sha256(a.eval[0]) if len(a.eval) == 1
                  else {str(e): sha256(e) for e in a.eval}, a.control: sha256(path_of(a.control))}
        for n in a.arm:
            hashes[n] = sha256(path_of(n))
        data = {"control": {"name": a.control, **cs},
                "arms": {n: r for n, r in results},
                "thresholds": {"min_net": a.min_net, "max_lost": a.max_lost,
                               "min_abstention_net": a.min_abstention_net},
                "eval": str(a.eval[0]) if len(a.eval) == 1 else [str(e) for e in a.eval],
                "aliases": str(aliases_path),
                "aliases_exists": aliases_path.exists(),
                "sha256": {"eval": hashes["eval"],
                           "aliases": sha256(aliases_path) if aliases_path.exists() else None,
                           "answers": {k: v for k, v in hashes.items() if k != "eval"}}}
        Path(a.json).write_text(json.dumps(data, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
