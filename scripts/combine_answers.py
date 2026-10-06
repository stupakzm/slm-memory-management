#!/usr/bin/env python3
"""Join a base run and an alternative run of the same rows into one answers file.

`--base NAME` and `--alt NAME` are answers files `<results>/<NAME>-answers.json`.
A base row is replaced by the alternative run's row for that qid when the base
refused after passing the gate (abstained true, gated false) and its gate score
(top_score) is at least `--min-score`. Every other row, including every gated
row and any row lacking `gated` or `top_score`, is kept unchanged. The output
`<out>-answers.json` is a normal answers file for screen_report.py; a sidecar
`<out>-combine.json` records the sha256 of both inputs and the replaced qids.
An existing output file is never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "eval" / "results"


def replaceable(row: dict, min_score: float) -> bool:
    if "gated" not in row or "top_score" not in row:
        return False
    return (bool(row.get("abstained")) and not row["gated"]
            and float(row["top_score"]) >= min_score)


def main(argv=None, results_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--alt", required=True)
    ap.add_argument("--min-score", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results-dir", default=str(RESULTS))
    args = ap.parse_args(argv)
    d = Path(results_dir if results_dir is not None else args.results_dir)

    base_p = d / f"{args.base}-answers.json"
    alt_p = d / f"{args.alt}-answers.json"
    out_p = d / f"{args.out}-answers.json"
    side_p = d / f"{args.out}-combine.json"
    for p in (base_p, alt_p):
        if not p.exists():
            print(p, file=sys.stderr)
            return 2
    for p in (out_p, side_p):
        if p.exists():
            print(f"{p} already exists; not overwriting", file=sys.stderr)
            return 2

    base_raw, alt_raw = base_p.read_bytes(), alt_p.read_bytes()
    base, alt = json.loads(base_raw), json.loads(alt_raw)
    alt_rows = {r["qid"]: r for r in alt["results"]}
    base_qids = [r["qid"] for r in base["results"]]
    if set(base_qids) != set(alt_rows) or len(base_qids) != len(alt_rows):
        print(f"qid sets differ between {args.base} and {args.alt}", file=sys.stderr)
        return 2

    rows, replaced = [], []
    for r in base["results"]:
        if replaceable(r, args.min_score):
            new = copy.deepcopy(alt_rows[r["qid"]])
            new["combined_from"] = "alt"
            rows.append(new)
            replaced.append(r["qid"])
        else:
            rows.append(r)

    config = dict(base.get("config") or {})
    config["combined"] = {"base": args.base, "alt": args.alt,
                          "min_score": args.min_score, "replaced": len(replaced)}
    out = {"name": args.out, "k": base.get("k"), "n": len(rows),
           "config": config, "results": rows}
    out_p.write_text(json.dumps(out, indent=1))
    side_p.write_text(json.dumps({
        "base": {"name": args.base, "sha256": hashlib.sha256(base_raw).hexdigest()},
        "alt": {"name": args.alt, "sha256": hashlib.sha256(alt_raw).hexdigest()},
        "min_score": args.min_score,
        "replaced_qids": sorted(replaced)}, indent=1))
    print(f"{args.out}: replaced {len(replaced)} of {len(rows)} rows "
          f"(base {args.base}, alt {args.alt}, min score {args.min_score})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
