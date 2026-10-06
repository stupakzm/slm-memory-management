#!/usr/bin/env python3
"""Replay stored answers against their stored retrieval caches and count ungrounded identifiers.

Per run (`--run NAME[:CACHE]`, answers `<NAME>-answers.json`, cache
`<CACHE>-retrieved.json`, CACHE defaults to NAME): among answered rows, split
into correct / wrong / unanswerable-answered, how many name an identifier (a
flag, key chord or command) that occurs in none of the top-5 extracts the
reader saw. Two identifier patterns are reported, `draft` and `tight`. Per
cache it also counts extracts that open with a lowercase fragment.

Correctness comes only from screen_report.score_run (recomputed from the stored
answer text, never trusted from the file). --json writes the counts plus the
sha256 of every input file read.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import gold  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "screen_report", ROOT / "scripts" / "screen_report.py")
sr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sr)

RESULTS = ROOT / "data" / "eval" / "results"
BUCKETS = ("correct", "wrong", "unanswerable-answered")
PATTERNS = ("draft", "tight")
STRIP = ".,;:)'\"`"

IDENT = re.compile(r"(?<![\w-])(--?[A-Za-z0-9][\w-]*(?:=\S+)?|[CMs]-\S+|[a-z]+(?:-[a-z0-9]+){1,})")
LONG_FLAG = re.compile(r"(?<![\w-])--[A-Za-z][A-Za-z0-9-]*")
SHORT_FLAG = re.compile(r"(?<![\w-])-[A-Za-z0-9](?![\w-])")
CHORD = re.compile(r"(?<![\w-])(?:[CMs]-)+[^\s,;:)'\"`]+")
MX_CMD = re.compile(r"M-x\s+([a-z][a-z0-9-]*)")
TICKED_FLAG = re.compile(r"(?<=`)-[A-Za-z][A-Za-z0-9]+(?=`)")


def strip_citations(answer: str) -> str:
    return re.sub(r"\[[1-9]\]", "", answer)


def draft_idents(answer: str) -> set:
    out = set()
    for m in IDENT.findall(strip_citations(answer)):
        tok = m.split("=", 1)[0].rstrip(STRIP)
        if len(tok) > 1:
            out.add(tok)
    return out


def tight_idents(answer: str) -> set:
    text = strip_citations(answer)
    out = {m.rstrip("-") for m in LONG_FLAG.findall(text)}
    out |= set(SHORT_FLAG.findall(text))
    out |= {m.rstrip(STRIP) for m in CHORD.findall(text)}
    out |= set(MX_CMD.findall(text))
    out |= set(TICKED_FLAG.findall(text))
    return {t for t in out if t}


EXTRACTORS = {"draft": draft_idents, "tight": tight_idents}


def unpack_hits(entry) -> list:
    """A cache entry is a list of hits or {"hits": [...], "gate_hits": [...]}."""
    return entry["hits"] if isinstance(entry, dict) else entry


def seen_hits(cache: dict, qid: str, read_k: int = 0) -> list:
    """The reader's list (eval_answers.select_hits, cap_per_doc == 0): hits[:read_k], or all hits if read_k is 0."""
    hits = list(unpack_hits(cache.get(qid, [])))
    return hits[:read_k] if read_k > 0 else hits


def seen_text(hits: list) -> str:
    return " ".join(h.get("prefix", "") + h["text"] for h in hits)


def ord_of(hit: dict) -> int:
    try:
        return int(hit.get("ord", 0))
    except (TypeError, ValueError):
        return 0


def is_mid_fragment(hit: dict) -> bool:
    first = hit["text"][:1]
    return ord_of(hit) > 0 and first.isalnum() and first.islower()


def bucket_of(score: dict) -> str:
    if not score["answerable"]:
        return "unanswerable-answered"
    return "correct" if score["correct"] else "wrong"


def grounding_counts(rows: list, scores: dict, cache: dict, read_k: int = 0) -> dict:
    """{pattern: {bucket: {"n", "flagged"}}} over answered rows."""
    out = {p: {b: {"n": 0, "flagged": 0} for b in BUCKETS} for p in PATTERNS}
    for row in rows:
        sc = scores[row["qid"]]
        if sc["abstained"]:
            continue
        seen = seen_text(seen_hits(cache, row["qid"], read_k))
        b = bucket_of(sc)
        for p in PATTERNS:
            cell = out[p][b]
            cell["n"] += 1
            if any(i not in seen for i in EXTRACTORS[p](row["answer"])):
                cell["flagged"] += 1
    return out


def extract_starts(rows: list, cache: dict, read_k: int = 0) -> dict:
    n = mid = 0
    for row in rows:
        for h in seen_hits(cache, row["qid"], read_k):
            n += 1
            mid += is_mid_fragment(h)
    return {"n": n, "mid": mid}


def parse_run(spec: str):
    name, _, cache = spec.partition(":")
    return name, (cache or name)


def run_lines(name: str, cache_name: str, counts: dict, starts: dict) -> list:
    lines = []
    for p in PATTERNS:
        c = counts[p]
        lines.append(f"{name} [{p}] " + " ".join(
            f"{b} {c[b]['flagged']}/{c[b]['n']}" for b in BUCKETS))
        lines.append(f"{name} [{p}] catch: wrong flagged {c['wrong']['flagged']}/{c['wrong']['n']}, "
                     f"correct flagged {c['correct']['flagged']}/{c['correct']['n']}")
    lines.append(f"{cache_name} extract-starts: lowercase fragment (ord>0) "
                 f"{starts['mid']}/{starts['n']}")
    return lines


def main(argv=None, results_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval", action="append", required=True)
    ap.add_argument("--aliases", default=None)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    rdir = Path(results_dir) if results_dir else RESULTS
    aliases_path = Path(a.aliases) if a.aliases else ROOT / "data" / "eval" / "gold_aliases_v2.json"
    aliases = gold.load_aliases(aliases_path)
    try:
        eval_rows = sr.load_evals(a.eval)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    runs, skipped, ans_hash, cache_hash = {}, {}, {}, {}
    for spec in a.run:
        name, cache_name = parse_run(spec)
        ans_path = rdir / f"{name}-answers.json"
        cache_path = rdir / f"{cache_name}-retrieved.json"
        reason = None
        for p in (ans_path, cache_path):
            if not p.exists():
                reason = f"missing {p.name}"
                break
        if reason is None:
            data = json.loads(ans_path.read_text())
            cfg = data.get("config") or {}
            if int(cfg.get("cap_per_doc") or 0) > 0:
                reason = "config cap_per_doc nonzero"
            read_k = int(cfg.get("read_k") or 0)
        if reason is None:
            try:
                scores = sr.score_run(ans_path, eval_rows, aliases)
            except ValueError:
                reason = "qid not in eval"
        if reason is not None:
            print(f"{name}: skipped ({reason})")
            skipped[name] = reason
            continue
        cache = json.loads(cache_path.read_text())
        rows = data["results"]
        counts = grounding_counts(rows, scores, cache, read_k)
        starts = extract_starts(rows, cache, read_k)
        sys.stdout.write("\n".join(run_lines(name, cache_name, counts, starts)) + "\n")
        runs[name] = {**counts, "extract_starts": starts}
        ans_hash[name] = sr.sha256(ans_path)
        cache_hash[name] = sr.sha256(cache_path)

    if a.json:
        out = {"runs": runs, "skipped": skipped,
               "sha256": {"eval": sr.sha256(a.eval[0]) if len(a.eval) == 1
                          else {str(e): sr.sha256(e) for e in a.eval},
                          "aliases": sr.sha256(aliases_path) if aliases_path.exists() else None,
                          "answers": ans_hash, "caches": cache_hash}}
        Path(a.json).write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
