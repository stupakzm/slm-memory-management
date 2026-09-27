#!/usr/bin/env python3
"""R3a (tsk_20260927_robreport): turn one eval_answers.py run into the
robustness map - the same question asked as a typo, a synonym, tersely,
casually, without naming the tool, or as an older paraphrase, measured
against its own clean base *in the same run*.

Runs no model: this is pure analysis over a committed answers file
(--answers, an eval_answers.py `*-answers.json`) and the pool of questions
jsonl files that run drew from (--pool, repeatable).

Why re-score rather than trust the stored `correct`
-----------------------------------------------------
Stored `correct`/`evidence_retrieved` flags predate later alias work
(gold_aliases.json keeps growing - see scripts/rescore_answers.py, the same
principle). This script never trusts a stored `correct`; it recomputes it
from the stored answer text with today's aliases, exactly like
rescore_answers.py does, but per-kind and paired rather than flat. Evidence
is trusted from the run (it was measured against a fixed retrieval, so it
cannot be rederived from the answer text alone) and abstention is trusted
too (it is either the model's own words or the gate, already resolved by
eval_answers.py: `abstained = gated or abstained(text)`).

kind assignment (per result row, via its pool row)
-----------------------------------------------------
"clean" (no variant_of, no paraphrase_of) / "paraphrase" (paraphrase_of set)
/ else the pool row's own variant_kind (typo1, typo3, synonym, casual,
terse, no-name, or one of the older typo/terse/no-tool rows) - see
`classify_kind`.

Pairing
-----------------------------------------------------
For every non-clean row, if its base (variant_of or paraphrase_of) is also
in this run, the pair counts toward kept/lost/gained/both_wrong (net =
gained - lost) for correctness (answerable pairs), evidence (answerable
pairs), and abstention (unanswerable pairs, lost = base abstained but the
variant answered). A base missing from the run counts the variant under
"unpaired" and is left out of every paired number. The two-sided exact
McNemar p is computed from (lost, gained) alone.

Stdlib only. The JSON output is deterministic: `json.dumps(..., sort_keys=True)`
sorts every dict's keys regardless of build order, and every list this script
emits (qids, domain names) is explicitly sorted before it is written, so two
runs on the same inputs are byte-identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import gold  # noqa: E402

CLEAN = "clean"
PARAPHRASE = "paraphrase"


def classify_kind(pool_row: dict) -> str:
    """"clean": neither variant_of nor paraphrase_of set. "paraphrase":
    paraphrase_of set. Else the pool row's own variant_kind, as-is."""
    if not pool_row.get("variant_of") and not pool_row.get("paraphrase_of"):
        return CLEAN
    if pool_row.get("paraphrase_of"):
        return PARAPHRASE
    return pool_row["variant_kind"]


def load_pool(paths: list) -> dict:
    """qid -> pool row, across every file. Errors out (SystemExit) on a qid
    that appears more than once anywhere in the pool - an ambiguous lookup
    that must never happen, not something to silently pick a winner for."""
    pool: dict = {}
    for path in paths:
        p = Path(path)
        for lineno, line in enumerate(p.open(encoding="utf-8"), 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row["qid"]
            if qid in pool:
                raise SystemExit(
                    f"duplicate qid in pool: {qid!r} ({path}:{lineno} and "
                    f"an earlier row)")
            pool[qid] = row
    return pool


def apply_overlay(results: list, overlay_results: list) -> list:
    """R4a (tsk_20260927_spellnorm): `results` (from --answers) with every
    qid that also appears in `overlay_results` REPLACED by the overlay's own
    row for that qid - rows not in the overlay are returned unchanged, in
    `results`' own order and count. Every overlay qid must already be a
    qid in `results`; SystemExit otherwise (an overlay is a replacement for
    known rows, not a way to inject new ones), as must a duplicate qid
    within the overlay itself."""
    by_qid = {r["qid"]: r for r in results}
    seen = set()
    for r in overlay_results:
        qid = r["qid"]
        if qid in seen:
            raise SystemExit(f"duplicate qid in --overlay: {qid!r}")
        seen.add(qid)
        if qid not in by_qid:
            raise SystemExit(f"--overlay qid {qid!r} not found in --answers")
        by_qid[qid] = r
    return [by_qid[r["qid"]] for r in results]


def mcnemar_p(lost: int, gained: int) -> float:
    """Exact two-sided McNemar p on the discordant pairs (lost, gained)
    under the binomial(n, 0.5) null, n = lost + gained. 1.0 when there are
    no discordant pairs at all."""
    n = lost + gained
    if n == 0:
        return 1.0
    m = min(lost, gained)
    tail = sum(comb(n, k) for k in range(0, m + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def paired_counts(pairs: list) -> dict:
    """pairs: list of (base_bool, variant_bool). kept = both true, lost =
    base true / variant false, gained = base false / variant true,
    both_wrong = both false. net = gained - lost."""
    kept = sum(1 for b, v in pairs if b and v)
    lost = sum(1 for b, v in pairs if b and not v)
    gained = sum(1 for b, v in pairs if not b and v)
    both_wrong = sum(1 for b, v in pairs if not b and not v)
    return {
        "kept": kept, "lost": lost, "gained": gained, "both_wrong": both_wrong,
        "net": gained - lost, "p": mcnemar_p(lost, gained),
    }


def new_bucket() -> dict:
    return {
        "answerable": {"n": 0, "evidence": 0, "correct": 0},
        "unanswerable": {"n": 0, "abstained": 0},
        # non-clean kinds only; filled in later
        "correct_pairs": [], "evidence_pairs": [], "abstained_pairs": [],
        "unpaired": 0,
    }


def add_row(bucket: dict, row: dict) -> None:
    if row["akind"] == "answerable":
        a = bucket["answerable"]
        a["n"] += 1
        a["evidence"] += bool(row["evidence"])
        a["correct"] += bool(row["correct"])
    else:
        u = bucket["unanswerable"]
        u["n"] += 1
        u["abstained"] += bool(row["abstained"])


def finish_bucket(bucket: dict, kind: str) -> dict:
    out = {"answerable": bucket["answerable"], "unanswerable": bucket["unanswerable"]}
    if kind != CLEAN:
        out["paired"] = {
            "correct": paired_counts(bucket["correct_pairs"]),
            "evidence": paired_counts(bucket["evidence_pairs"]),
            "abstained": paired_counts(bucket["abstained_pairs"]),
        }
        out["unpaired"] = {"n": bucket["unpaired"]}
    return out


def build_report(pool: dict, results: list, aliases: dict) -> dict:
    computed: dict = {}
    for r in results:
        qid = r["qid"]
        if qid in computed:
            raise SystemExit(f"duplicate qid in results: {qid!r}")
        if qid not in pool:
            raise SystemExit(f"qid {qid!r} in --answers has no pool row (check --pool)")
        prow = pool[qid]
        akind = prow["kind"]
        base_qid = prow.get("variant_of") or prow.get("paraphrase_of")
        abstained = bool(r["abstained"])
        correct = evidence = None
        if akind == "answerable":
            if "evidence_retrieved" not in r:
                raise SystemExit(
                    f"qid {qid!r}: pool row is answerable but the answers file has "
                    f"no evidence_retrieved (recorded kind={r.get('kind')!r}) - "
                    f"the --answers run and --pool have drifted out of sync")
            toks = prow["answer_contains"]
            row_aliases = gold.aliases_for(aliases, qid, base_qid)
            correct = (not abstained) and gold.is_correct(r["answer"], toks, row_aliases)
            evidence = bool(r["evidence_retrieved"])
        computed[qid] = {
            "kind": classify_kind(prow),
            "domain": prow.get("domain") or "linux",
            "akind": akind,
            "abstained": abstained,
            "correct": correct,
            "evidence": evidence,
            "base_qid": base_qid,
        }

    kinds = sorted({c["kind"] for c in computed.values()})
    overall: dict = {k: new_bucket() for k in kinds}
    by_domain: dict = {k: {} for k in kinds}

    for qid, c in computed.items():
        k, d = c["kind"], c["domain"]
        add_row(overall[k], c)
        dom_bucket = by_domain[k].setdefault(d, new_bucket())
        add_row(dom_bucket, c)

    for qid, c in computed.items():
        if c["kind"] == CLEAN:
            continue
        base_qid = c["base_qid"]
        base = computed.get(base_qid) if base_qid else None
        paired = base is not None and base["akind"] == c["akind"]
        buckets = [overall[c["kind"]], by_domain[c["kind"]][c["domain"]]]
        if not paired:
            for b in buckets:
                b["unpaired"] += 1
            continue
        if c["akind"] == "answerable":
            for b in buckets:
                b["correct_pairs"].append((base["correct"], c["correct"]))
                b["evidence_pairs"].append((base["evidence"], c["evidence"]))
        else:
            for b in buckets:
                b["abstained_pairs"].append((base["abstained"], c["abstained"]))

    report_kinds = {}
    for k in kinds:
        report_kinds[k] = {
            "overall": finish_bucket(overall[k], k),
            "domains": {
                d: finish_bucket(by_domain[k][d], k)
                for d in sorted(by_domain[k])
            },
        }

    return {"kinds": report_kinds}


def print_table(report: dict) -> None:
    kinds = report["kinds"]
    print(f"{'kind':10} {'domain':8} {'ans n':>6} {'correct':>8} {'evidence':>9}"
          f" {'unans n':>8} {'abstained':>10}")
    for kind in sorted(kinds):
        o = kinds[kind]["overall"]
        a, u = o["answerable"], o["unanswerable"]
        print(f"{kind:10} {'overall':8} {a['n']:6d} {a['correct']:8d} {a['evidence']:9d}"
              f" {u['n']:8d} {u['abstained']:10d}")
        for domain in sorted(kinds[kind]["domains"]):
            b = kinds[kind]["domains"][domain]
            a, u = b["answerable"], b["unanswerable"]
            print(f"{'':10} {domain:8} {a['n']:6d} {a['correct']:8d} {a['evidence']:9d}"
                  f" {u['n']:8d} {u['abstained']:10d}")
        if "paired" in o:
            for metric in ("correct", "evidence", "abstained"):
                p = o["paired"][metric]
                print(f"  paired[{metric:9}] kept={p['kept']:4d} lost={p['lost']:4d} "
                      f"gained={p['gained']:4d} both_wrong={p['both_wrong']:4d} "
                      f"net={p['net']:+4d} p={p['p']:.4f}")
            print(f"  unpaired n={o['unpaired']['n']}")
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--answers", required=True, help="eval_answers.py output json")
    ap.add_argument("--pool", nargs="+", required=True,
                     help="jsonl question files the run drew from")
    ap.add_argument("--out", required=True, help="where to write the JSON report")
    ap.add_argument("--aliases", default=str(ROOT / "data" / "eval" / "gold_aliases.json"))
    ap.add_argument("--overlay", default=None,
                     help="an eval_answers.py output whose results REPLACE the --answers "
                          "results with the same qid (phase 11 R4a, tsk_20260927_spellnorm) "
                          "- every overlay qid must already exist in --answers. Without "
                          "--overlay the output is unchanged (byte-identical).")
    args = ap.parse_args()

    pool = load_pool(args.pool)
    aliases = gold.load_aliases(args.aliases)
    data = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    results = data["results"]
    if args.overlay:
        overlay_data = json.loads(Path(args.overlay).read_text(encoding="utf-8"))
        results = apply_overlay(results, overlay_data["results"])

    report = build_report(pool, results, aliases)
    report = {
        "answers": args.answers,
        "pool": sorted(args.pool),
        "aliases": args.aliases,
        "n_results": len(results),
        "kinds": report["kinds"],
    }
    if args.overlay:
        report["overlay"] = args.overlay

    print_table(report)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
