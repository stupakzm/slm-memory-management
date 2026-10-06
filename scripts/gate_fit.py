#!/usr/bin/env python3
"""Fit a calibrated gate per run and replay it on held-out stored answers.

The plain gate is one threshold on the top-1 score (0.65). Every retrieval
change moves the score distribution, so this scorer fits, per run, a small
logistic model on features read from that run's own retrieval cache
(`<results>/<CACHE>-retrieved.json`), on a fixed family-preserving fit split;
freezes it; and replays it as a gate over the held-out split's stored answers
(`<results>/<NAME>-answers.json`). No GPU, no model: stored answers are only
turned into refusals, never regenerated.

SPLIT  family key = variant_of or paraphrase_of or qid; bucket =
       int(sha256(key), 16) % 10; bucket <= 6 -> fit, else held. Variants of
       one question never straddle the split; the split depends on the eval
       rows only, so it is identical for every run.
FEATURES (from the cache entry's gate_hits): top1, gap, same_doc, dist1,
       mean5, unk (share of 4+ letter question words absent from --vocab).
LABEL  1 iff the row is answerable and its stored evidence_retrieved is True.
MODEL  standardise with fit-row mean / population std; L2 logistic regression,
       zero start, full-batch gradient descent, lr 0.1, 2000 iterations, L2
       gradient (1.0 / n_fit_rows) * w on the weights (never the bias).
OPERATING POINT  the target is the number of unanswerable fit rows the
       control refuses under its own plain gate; each run's threshold t is the
       lowest candidate at which its fit-split unanswerable refusals (stored
       abstained or p < t) reach the target. A row with no hits is never
       fitted and always refused. The baseline a calibrated gate has to beat is
       the top-1-only gate: the same threshold search over the raw top-1
       feature instead of the model's probability, with the same target.

Correctness is recomputed by screen_report.score_run (imported, never
re-implemented). --json writes the same numbers plus the model and the
sha256 of every input.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import gold, normalize  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "screen_report", ROOT / "scripts" / "screen_report.py")
sr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sr)

RESULTS = ROOT / "data" / "eval" / "results"

FEATURES = ("top1", "gap", "same_doc", "dist1", "mean5", "unk")
SPLIT_RULE = ("family key = variant_of or paraphrase_of or qid; bucket = "
              "int(sha256(key), 16) % 10; bucket <= 6 -> fit, bucket >= 7 -> held")
LR = 0.1
ITERS = 2000
L2 = 1.0
EPS = 1e-12


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------

def family_key(eval_row: dict) -> str:
    return eval_row.get("variant_of") or eval_row.get("paraphrase_of") or eval_row["qid"]


def bucket_of(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 10


def split_of(eval_row: dict) -> str:
    return "fit" if bucket_of(family_key(eval_row)) <= 6 else "held"


# --------------------------------------------------------------------------
# features and label
# --------------------------------------------------------------------------

def gate_value(hit: dict) -> float:
    """The number the gate thresholds (the rule of smm.retrieve.gate_score)."""
    return float(hit["rerank_score"] if "rerank_score" in hit else hit.get("score", 0.0))


def gate_hits_of(entry) -> list:
    return entry["gate_hits"] if isinstance(entry, dict) else entry


def unknown_share(question: str, vocab) -> float:
    if not vocab:
        return 0.0
    words = [w for w in re.findall(r"[a-z]+", question.lower()) if len(w) >= 4]
    if not words:
        return 0.0
    return sum(1 for w in words if w not in vocab) / len(words)


def features(entry, question: str, vocab):
    """One float vector in FEATURES order, or None when there are no hits."""
    hits = gate_hits_of(entry) if entry is not None else []
    if not hits:
        return None
    vals = [gate_value(h) for h in hits[:5]]
    top1 = vals[0]
    gap = top1 - gate_value(hits[1]) if len(hits) > 1 else top1
    same_doc = sum(1 for h in hits[:5] if h.get("doc_id") == hits[0].get("doc_id"))
    d = hits[0].get("distance")
    dist1 = float(d) if d is not None else 0.0
    return [top1, gap, float(same_doc), dist1, sum(vals) / len(vals),
            unknown_share(question, vocab)]


def label_of(row: dict) -> int:
    return 1 if row["answerable"] and row["evidence"] else 0


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def standardiser(X: list) -> tuple:
    """(means, stds) over the given rows; population std, 0 -> 1."""
    n = len(X)
    k = len(FEATURES)
    if n == 0:
        return [0.0] * k, [1.0] * k
    means = [sum(r[j] for r in X) / n for j in range(k)]
    stds = [math.sqrt(sum((r[j] - means[j]) ** 2 for r in X) / n) for j in range(k)]
    return means, [s if s > 0 else 1.0 for s in stds]


def standardise(x: list, means: list, stds: list) -> list:
    return [(v - m) / s for v, m, s in zip(x, means, stds)]


def sigmoid(z: float) -> float:
    z = max(-60.0, min(60.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def fit_logistic(Z: list, y: list) -> tuple:
    """(weights, bias) on already-standardised rows. Deterministic."""
    k = len(FEATURES)
    w, b = [0.0] * k, 0.0
    n = len(Z)
    if n == 0:
        return w, b
    reg = L2 / n
    for _ in range(ITERS):
        gw, gb = [0.0] * k, 0.0
        for z, t in zip(Z, y):
            e = sigmoid(b + sum(wj * zj for wj, zj in zip(w, z))) - t
            gb += e
            for j in range(k):
                gw[j] += e * z[j]
        b -= LR * gb / n
        w = [wj - LR * (g / n + reg * wj) for wj, g in zip(w, gw)]
    return w, b


def predict(model: dict, x: list) -> float:
    z = standardise(x, model["means"], model["stds"])
    return sigmoid(model["bias"] + sum(wj * zj for wj, zj in zip(model["weights"], z)))


def choose_threshold(rows: list, target: int) -> tuple:
    """rows: (p or None, abstained) for the fit-split unanswerable rows.
    Returns (t, reachable): t is the lowest candidate at which refusals
    (abstained, no p, or p < t) reach target; else the largest candidate."""
    cands = sorted({p + EPS for p, _ in rows if p is not None})
    for t in cands:
        if sum(1 for p, a in rows if a or p is None or p < t) >= target:
            return t, True
    return (cands[-1] if cands else 0.0), False


def replay(run: dict, probs: dict, t: float) -> dict:
    """Rows with p < t, or with no features, become refusals; the rest keep their stored values."""
    out = {}
    for q, r in run.items():
        p = probs.get(q)
        out[q] = {**r, "abstained": True, "correct": False} if p is None or p < t else r
    return out


def fit_run(run: dict, feats: dict, eval_rows: dict, target: int) -> dict:
    """Fit on this run's fit rows, pick the operating point, replay on all rows."""
    fit_q = [q for q in run if split_of(eval_rows[q]) == "fit" and feats[q] is not None]
    X = [feats[q] for q in fit_q]
    means, stds = standardiser(X)
    w, b = fit_logistic([standardise(x, means, stds) for x in X],
                        [label_of(run[q]) for q in fit_q])
    model = {"means": means, "stds": stds, "weights": w, "bias": b}
    probs = {q: (predict(model, feats[q]) if feats[q] is not None else None) for q in run}
    una = [q for q in run if split_of(eval_rows[q]) == "fit" and not run[q]["answerable"]]
    t, reachable = choose_threshold(
        [(probs[q], run[q]["abstained"]) for q in una if feats[q] is not None]
        + [(None, run[q]["abstained"]) for q in una if feats[q] is None], target)
    return {"model": model, "probs": probs, "threshold": t, "reachable": reachable,
            "calibrated": replay(run, probs, t)}


def top1_run(run: dict, feats: dict, eval_rows: dict, target: int) -> dict:
    """The top-1-only gate: threshold the top1 feature, tuned to the same fit-row target."""
    probs = {q: (feats[q][0] if feats[q] is not None else None) for q in run}
    una = [q for q in run if split_of(eval_rows[q]) == "fit" and not run[q]["answerable"]]
    t, reachable = choose_threshold([(probs[q], run[q]["abstained"]) for q in una], target)
    return {"threshold": t, "reachable": reachable, "replayed": replay(run, probs, t)}


# --------------------------------------------------------------------------
# io and report
# --------------------------------------------------------------------------

def parse_run(spec: str) -> tuple:
    name, _, cache = spec.partition(":")
    return name, cache or name


def subset(run: dict, qids) -> dict:
    return {q: run[q] for q in qids}


def weights_line(name: str, model: dict) -> str:
    parts = " ".join(f"{f} {w:+.4f}" for f, w in zip(FEATURES, model["weights"]))
    return f"{name} weights bias {model['bias']:+.4f} {parts}"


def main(argv=None, results_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval", action="append", required=True)
    ap.add_argument("--aliases", default=None)
    ap.add_argument("--control", required=True)
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--min-net", type=int, default=4)
    ap.add_argument("--max-lost", type=int, default=1000000000)
    ap.add_argument("--min-abstention-net", type=int, default=0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    rdir = Path(results_dir) if results_dir else RESULTS
    aliases_path = Path(a.aliases) if a.aliases else ROOT / "data" / "eval" / "gold_aliases_v2.json"
    specs = [parse_run(a.control)] + [parse_run(s) for s in a.arm]
    names = [n for n, _ in specs]
    answers_path = {n: rdir / f"{n}-answers.json" for n in names}
    cache_path = {n: rdir / f"{c}-retrieved.json" for n, c in specs}

    needed = [Path(p) for p in a.eval] + [aliases_path] + list(answers_path.values()) \
        + list(cache_path.values()) + ([Path(a.vocab)] if a.vocab else [])
    for p in needed:
        if not p.exists():
            print(str(p), file=sys.stderr)
            return 2

    try:
        eval_rows = sr.load_evals(a.eval)
        aliases = gold.load_aliases(aliases_path)
        vocab = normalize.load_vocab(a.vocab) if a.vocab else None
        runs = {n: sr.score_run(answers_path[n], eval_rows, aliases) for n in names}
        ctl_name = names[0]
        ctl = runs[ctl_name]
        for n in names[1:]:
            sr.compare(ctl, runs[n])  # raises ValueError on a qid-set mismatch
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    qids = sorted(ctl)
    caches = {n: json.loads(cache_path[n].read_text()) for n in names}
    missing = [(n, q) for n in names for q in sorted(runs[n]) if q not in caches[n]]
    for n, q in missing:
        print(f"{cache_path[n]}: qid {q} has no entry", file=sys.stderr)
    if missing:
        return 2
    fit_q = [q for q in qids if split_of(eval_rows[q]) == "fit"]
    held_q = [q for q in qids if split_of(eval_rows[q]) == "held"]
    fit_s, held_s = sr.summarize(subset(ctl, fit_q)), sr.summarize(subset(ctl, held_q))
    target = fit_s["abstained"]
    lines = [f"split fit {len(fit_q)} rows ({fit_s['answerable']} answerable, "
             f"{fit_s['unanswerable']} unanswerable) held {len(held_q)} rows "
             f"({held_s['answerable']} answerable, {held_s['unanswerable']} unanswerable)",
             f"target control refuses {target}/{fit_s['unanswerable']} "
             f"unanswerable fit rows at its own gate"]

    ctl_held = subset(ctl, held_q)
    kw = (a.min_net, a.max_lost, a.min_abstention_net)
    out = {"split_rule": SPLIT_RULE, "features": list(FEATURES), "control": ctl_name,
           "target": target, "runs": {}}
    for n, c in specs:
        cache = caches[n]
        feats = {q: features(cache.get(q), eval_rows[q].get("question", ""), vocab)
                 for q in qids}
        res = fit_run(runs[n], feats, eval_rows, target)
        cal = res["calibrated"]
        fs = sr.summarize(subset(cal, fit_q))
        lines.append(weights_line(n, res["model"]))
        if not res["reachable"]:
            lines.append(f"{n} fit: target unreachable")
        lines.append(f"{n} fit threshold {res['threshold']:.6f} refused "
                     f"{fs['abstained']}/{fs['unanswerable']} "
                     f"correct {fs['correct']}/{fs['answerable']}")
        entry = {"cache": c, **res["model"], "threshold": res["threshold"],
                 "reachable": res["reachable"],
                 "fit": {"refused": fs["abstained"], "unanswerable": fs["unanswerable"],
                         "correct": fs["correct"], "answerable": fs["answerable"]}}
        t1 = top1_run(runs[n], feats, eval_rows, target)
        t1s = sr.summarize(subset(t1["replayed"], fit_q))
        if not t1["reachable"]:
            lines.append(f"{n} fit top1: target unreachable")
        lines.append(f"{n} fit top1 threshold {t1['threshold']:.6f} refused "
                     f"{t1s['abstained']}/{t1s['unanswerable']}")
        t1_cmp = sr.compare(ctl_held, subset(t1["replayed"], held_q), *kw)
        entry["top1_matched"] = {"threshold": t1["threshold"], "reachable": t1["reachable"],
                                 "held": t1_cmp}
        cal_held = subset(cal, held_q)
        cal_cmp = sr.compare(ctl_held, cal_held, *kw)
        if n == ctl_name:
            lines.append(f"{n} held calibrated vs {ctl_name} plain: {sr._fields(cal_cmp)}")
            entry["held_calibrated"] = cal_cmp
            lines.append(f"{n} held top1-matched vs {ctl_name} plain: {sr._fields(t1_cmp)}")
        else:
            plain_cmp = sr.compare(ctl_held, subset(runs[n], held_q), *kw)
            lines.append(f"{n} held plain vs {ctl_name}: {sr._fields(plain_cmp)}")
            lines.append(f"{n} held calibrated vs {ctl_name}: {sr._fields(cal_cmp)}")
            entry["held_plain"], entry["held_calibrated"] = plain_cmp, cal_cmp
            lines.append(f"{n} held top1-matched vs {ctl_name}: {sr._fields(t1_cmp)}")
        out["runs"][n] = entry
    sys.stdout.write("\n".join(lines) + "\n")

    if a.json:
        out["thresholds"] = {"min_net": a.min_net, "max_lost": a.max_lost,
                             "min_abstention_net": a.min_abstention_net}
        out["sha256"] = {"eval": {str(e): sr.sha256(e) for e in a.eval},
                         "aliases": sr.sha256(aliases_path),
                         "vocab": sr.sha256(a.vocab) if a.vocab else None,
                         "answers": {n: sr.sha256(answers_path[n]) for n in names},
                         "caches": {n: sr.sha256(cache_path[n]) for n in names}}
        Path(a.json).write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
