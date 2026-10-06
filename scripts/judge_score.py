#!/usr/bin/env python3
"""Read the blind judge's labels and print each pair's judged net next to its labelled net.

  --key FILE          the key written by scripts/judge_pack.py
  --judgments FILE    repeatable; one JSON object {"id", "label"} per line, or an
                      {"items": [...]} object (or a JSON list) of such objects
  --eval FILE         repeatable; the eval files the pack was built from

For every pair in the key the labelled net is recomputed with
screen_report.compare, so both nets are printed from tracked code. Over the
rows whose labels differ between the runs, a side's judged label is the judge's
label for its item; a refusal side counts WRONG, an item with no judgment counts
UNCLEAR, and a row with an UNCLEAR side is excluded from lost/gained and
counted in `unclear`. A judgment id that is not in the key exits 2.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RESULTS = ROOT / "data" / "eval" / "results"
LABELS = ("CORRECT", "WRONG", "UNCLEAR")


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sr = _load("screen_report", "scripts/screen_report.py")


def read_judgments(paths) -> dict:
    """{id: label}; a later judgment of the same id replaces an earlier one."""
    out = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            objs = obj if isinstance(obj, list) else obj["items"] if "items" in obj else [obj]
            for o in objs:
                label = str(o["label"]).strip().upper()
                if label not in LABELS:
                    raise ValueError(f"{path}: label {o['label']!r} for {o['id']} is not one of "
                                     + "/".join(LABELS))
                out[o["id"]] = label
    return out


def read_key(path) -> list:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def judge_pair(ctl: dict, arm: dict, labelled: dict, side_label) -> dict:
    """Judged lost/gained/unclear over the rows whose labels differ in `labelled`."""
    lost = gained = unclear = 0
    for qid in labelled["lost_qids"] + labelled["gained_qids"]:
        c, m = side_label("control", qid, ctl[qid]), side_label("arm", qid, arm[qid])
        if "UNCLEAR" in (c, m):
            unclear += 1
        elif m == "CORRECT" and c == "WRONG":
            gained += 1
        elif c == "CORRECT" and m == "WRONG":
            lost += 1
    return {"lost": lost, "gained": gained, "unclear": unclear,
            "rows": len(labelled["lost_qids"]) + len(labelled["gained_qids"])}


def main(argv=None, results_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--key", required=True)
    ap.add_argument("--judgments", action="append", required=True)
    ap.add_argument("--eval", action="append", required=True)
    ap.add_argument("--aliases", default=str(ROOT / "data" / "eval" / "gold_aliases_v2.json"))
    a = ap.parse_args(argv)

    rdir = Path(results_dir) if results_dir else RESULTS
    for path in [a.key, *a.judgments, *a.eval, a.aliases]:
        if not Path(path).exists():
            print(f"missing input: {path}", file=sys.stderr)
            return 2
    try:
        key = read_key(a.key)
        labels = read_judgments(a.judgments)
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"bad input: {e}", file=sys.stderr)
        return 2
    ids = {k["id"] for k in key}
    unknown = sorted(set(labels) - ids)
    if unknown:
        print(f"judgment id not in the key: {unknown[0]}", file=sys.stderr)
        return 2

    side_ids, hidden, pairs = {}, set(), []
    for k in key:
        for e in k["entries"]:
            if e["role"] == "hidden_control":
                hidden.add(k["id"])
            else:
                side_ids[(e["pair"], e["role"], e["qid"])] = k["id"]
            if e["pair"] not in pairs:
                pairs.append(e["pair"])

    try:
        eval_rows = sr.load_evals(a.eval)
        aliases = sr.gold.load_aliases(a.aliases)
        names = sorted({n for p in pairs for n in p.split(":", 1)})
        runs = {n: sr.score_run(rdir / f"{n}-answers.json", eval_rows, aliases) for n in names}
        results = []
        for pair in pairs:
            cn, an = pair.split(":", 1)
            labelled = sr.compare(runs[cn], runs[an])

            def side_label(role, qid, run_row, pair=pair):
                if run_row["abstained"]:
                    return "WRONG"
                return labels.get(side_ids.get((pair, role, qid)), "UNCLEAR")

            results.append((pair, labelled, judge_pair(runs[cn], runs[an], labelled, side_label)))
    except (ValueError, OSError) as e:
        print(str(e), file=sys.stderr)
        return 2

    for pair, lab, j in results:
        print(f"{pair} rows {j['rows']} labelled lost {lab['lost']} gained {lab['gained']} "
              f"net {sr.signed(lab['net'])} | judged lost {j['lost']} gained {j['gained']} "
              f"net {sr.signed(j['gained'] - j['lost'])} "
              f"p {sr.sign_test_p(j['gained'], j['lost']):.3g} unclear {j['unclear']}")
    right = sum(labels.get(i) == "CORRECT" for i in hidden)
    print(f"hidden controls judged CORRECT {right}/{len(hidden)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
