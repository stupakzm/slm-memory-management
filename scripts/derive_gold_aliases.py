#!/usr/bin/env python3
"""Derive gold-token aliases (tsk_20260926_a51d0707), mechanically, from the
man pages alone.

Why: answer correctness is `all(t in text for t in row["answer_contains"])`
(scripts/eval_answers.py). Gold tokens are mostly long-form options
(`--no-clobber`), so a correct answer using the short form documented on the
same option line (`-n`) is scored wrong. That noise sits under every number
phase 6's verifier was judged against.

This script reads ONLY data/corpus/man.jsonl and data/eval/questions.jsonl -
never an answer, never a result - and for each answerable question's gold
token, searches the gold doc's option lines for three kinds of match:

  synonym     the token is itself one of the options on an option line
              (`--no-clobber` on `-n, --no-clobber`); the other options on
              that line, with `=ARG`/` ARG` stripped, become its aliases.
  argument    the token is the argument placeholder on an option line
              (`identity_file` on `-i identity_file`); that line's options
              become its aliases.
  description the token is a phrase that occurs verbatim in an option
              entry's description text (`sort by time` under `-t`); that
              entry's options become its aliases.

Every alias is recorded with the rule that produced it, the section it came
from, and the option line itself - so any alias can be traced back to the
one man page sentence that licenses it, and none of this ever looks at a
model's answer.

Usage: derive_gold_aliases.py [--corpus data/corpus/man.jsonl]
                               [--eval data/eval/questions.jsonl]
                               [--out data/eval/gold_aliases.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_MAX_OPT_INDENT = 7


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def parse_option_line(line: str) -> list[tuple[str, str | None]]:
    """Split one option line into (flag, arg_or_None) pairs.

    `-L, --dereference` -> [("-L", None), ("--dereference", None)]
    `--exclude-from=FILE` -> [("--exclude-from", "FILE")]
    `-i identity_file` -> [("-i", "identity_file")]
    `-C NUM, -NUM, --context=NUM` -> [("-C","NUM"),("-NUM",None),("--context","NUM")]
    """
    out = []
    for unit in line.split(","):
        u = unit.strip()
        if not u:
            continue
        if "=" in u:
            flag, arg = u.split("=", 1)
            out.append((flag.strip(), arg.strip()))
        elif " " in u:
            flag, arg = u.split(None, 1)
            out.append((flag.strip(), arg.strip()))
        else:
            out.append((u, None))
    return out


def parse_option_entries(text: str) -> list[dict]:
    """Find every option entry in a section's text: an option line (starting,
    after at most ~7 leading spaces, with `-`) plus the following
    more-indented lines, up to the next option line or a dedent."""
    lines = text.split("\n")
    entries: list[dict] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        indent = _indent(line)
        if stripped and indent <= _MAX_OPT_INDENT and stripped.startswith("-"):
            opt_indent = indent
            desc_lines: list[str] = []
            j = i + 1
            while j < n:
                nxt = lines[j]
                nxt_stripped = nxt.strip()
                if not nxt_stripped:
                    desc_lines.append(nxt)
                    j += 1
                    continue
                if _indent(nxt) > opt_indent:
                    desc_lines.append(nxt)
                    j += 1
                else:
                    break
            units = parse_option_line(stripped)
            entries.append({
                "line": stripped,
                "units": units,
                "flags": [f for f, _ in units],
                "description": "\n".join(desc_lines).strip(),
            })
            i = j
        else:
            i += 1
    return entries


def alias_matches_for_token(doc: dict, token: str) -> list[dict]:
    """Every (alias, rule, sec_id, line) match for `token` in `doc`."""
    matches: list[dict] = []
    seen = set()

    def add(alias: str, rule: str, sec_id: str, line: str) -> None:
        key = (alias, rule, sec_id, line)
        if key in seen:
            return
        seen.add(key)
        matches.append({"alias": alias, "rule": rule, "sec_id": sec_id, "line": line})

    for sec in doc.get("sections", []):
        sec_id = sec.get("sec_id", "")
        for entry in parse_option_entries(sec.get("text", "")):
            flags = entry["flags"]
            line = entry["line"]

            # synonym: token is one of this line's options.
            if token in flags:
                for other in flags:
                    if other != token:
                        add(other, "synonym", sec_id, line)

            # argument: token is an argument placeholder on this line.
            if any(arg == token for _flag, arg in entry["units"] if arg is not None):
                for flag in flags:
                    add(flag, "argument", sec_id, line)

            # description: token is a phrase in this entry's description.
            if entry["description"] and token in entry["description"]:
                for flag in flags:
                    add(flag, "description", sec_id, line)

    return matches


def load_corpus(path: Path) -> dict:
    docs = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            docs[d["doc_id"]] = d
    return docs


def load_questions(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def derive(corpus_path: Path, eval_path: Path) -> tuple[dict, dict]:
    """Returns (aliases, stats). aliases: {qid: {token: [alias-dict, ...]}},
    tokens with no alias omitted. stats: counts per rule, qids touched, and
    tokens with more than 3 aliases."""
    docs = load_corpus(Path(corpus_path))
    rows = load_questions(Path(eval_path))

    aliases: dict[str, dict] = {}
    per_rule = {"synonym": 0, "argument": 0, "description": 0}
    touched: set[str] = set()
    over3: list[tuple[str, str, int]] = []

    for row in rows:
        if row.get("kind") != "answerable":
            continue
        doc = docs.get(row.get("doc"))
        if doc is None:
            continue
        qid = row["qid"]
        per_token: dict[str, list] = {}
        for token in row.get("answer_contains", []):
            m = alias_matches_for_token(doc, token)
            if not m:
                continue
            per_token[token] = m
            touched.add(qid)
            for entry in m:
                per_rule[entry["rule"]] += 1
            if len(m) > 3:
                over3.append((qid, token, len(m)))
        aliases[qid] = per_token

    stats = {
        "per_rule": per_rule,
        "questions_touched": len(touched),
        "over3": over3,
    }
    return aliases, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus/man.jsonl")
    ap.add_argument("--eval", default="data/eval/questions.jsonl")
    ap.add_argument("--out", default="data/eval/gold_aliases.json")
    args = ap.parse_args()

    corpus_path = ROOT / args.corpus
    eval_path = ROOT / args.eval
    out_path = ROOT / args.out

    if not corpus_path.exists():
        print(f"no corpus at {corpus_path}", file=sys.stderr)
        return 2
    if not eval_path.exists():
        print(f"no eval set at {eval_path}", file=sys.stderr)
        return 2

    aliases, stats = derive(corpus_path, eval_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aliases, indent=2, sort_keys=True))

    print(f"wrote {out_path}")
    print(f"questions touched: {stats['questions_touched']}")
    print("aliases per rule:")
    for rule, n in sorted(stats["per_rule"].items()):
        print(f"  {rule:12} {n}")
    if stats["over3"]:
        print("\ntokens with more than 3 aliases (review):")
        for qid, token, n in stats["over3"]:
            print(f"  {qid:8} {token!r:30} {n} aliases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
