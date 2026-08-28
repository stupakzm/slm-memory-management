#!/usr/bin/env python3
"""Validate the phase 4 tool eval set against the corpus. No gold by assertion.

`resolve_gold.py` refuses to let a phase 0 question claim an answer the machine does
not actually document. This is the same rule for tool questions, and it has more to
check, because a tool question asserts three separable things:

- that a proposed command's option really exists in this machine's version of the
  tool - `dd` here documents `status=LEVEL`, not `status=progress`, and phase 0 only
  caught that because something checked;
- that a `thin-evidence` question's tool really is absent. That claim is the entire
  point of those questions, and it silently rots the first time someone installs
  rsync;
- that a `lookup` question names a page that exists at the section it names.

It also refuses a request that leaks its own expected command token, for the same
reason phase 0 did: a question containing `--exclude-from` tests string matching
rather than retrieval.

  .venv/bin/python scripts/resolve_tools.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Short flags are spelling, not evidence: `-r` appears in every page ever written, so
# only a long option or a literal keyword is checked against the document.
LONG_OPT = re.compile(r"^(--[a-z][a-z0-9-]+|[a-z_]+=|%[A-Za-z]|list-[a-z-]+|enable|stop)$")


def load_corpus(path: Path) -> dict:
    docs = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            docs[d["doc_id"]] = d
    return docs


def doc_text(doc: dict) -> str:
    return "\n".join(s["heading"] + "\n" + s["text"] for s in doc["sections"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus/man.jsonl")
    ap.add_argument("--tools", default="data/eval/tool_questions.jsonl")
    ap.add_argument("--eval", default="data/eval/questions.jsonl")
    args = ap.parse_args()

    docs = load_corpus(ROOT / args.corpus)
    names = {d["name"] for d in docs.values()}
    phase0 = {json.loads(l)["qid"]: json.loads(l)
              for l in (ROOT / args.eval).open(encoding="utf-8")}
    rows = [json.loads(l) for l in (ROOT / args.tools).open(encoding="utf-8")]

    problems, checked = [], 0
    for r in rows:
        qid, req = r["qid"], r["request"]

        if r.get("source_qid") and r["source_qid"] not in phase0:
            problems.append(f"{qid}: source_qid {r['source_qid']} not in phase 0 set")
        if r.get("source_qid") and r.get("doc"):
            src = phase0.get(r["source_qid"])
            if src and src.get("doc") and src["doc"] != r["doc"]:
                problems.append(f"{qid}: doc {r['doc']} disagrees with source "
                                f"{r['source_qid']} ({src['doc']})")

        if r["kind"] == "thin-evidence":
            # The claim under test, named explicitly rather than guessed from the
            # wording: *this* tool is not documented on this machine. Scanning the
            # request for any word that happens to be a page name does not work -
            # "Sync ~/photos with rsync" contains `sync`, which is a real page.
            tool = r.get("missing_tool")
            if not tool:
                problems.append(f"{qid}: thin-evidence question declares no missing_tool")
            elif tool in names:
                problems.append(f"{qid}: claims {tool!r} is absent, but it has a page "
                                f"on this machine")
            else:
                checked += 1
            continue

        if r["kind"] == "lookup":
            page = f"{r['expect_args']['page']}.{r['expect_args']['section']}"
            if page not in docs:
                problems.append(f"{qid}: lookup names {page}, which is not in the corpus")
            checked += 1
            continue

        if not r.get("doc"):
            continue
        if r["doc"] not in docs:
            problems.append(f"{qid}: doc {r['doc']} not in corpus")
            continue

        text = doc_text(docs[r["doc"]])
        for group in r.get("command_contains") or []:
            testable = [t for t in group if LONG_OPT.match(t)]
            if testable and not any(t in text for t in testable):
                problems.append(f"{qid}: none of {testable} appear in {r['doc']}")
            # A request must not hand over its own answer.
            for t in group:
                if len(t) > 3 and t.lower() in req.lower():
                    problems.append(f"{qid}: request leaks expected token {t!r}")
            checked += 1

    print(f"{len(rows)} tool questions, {checked} grounded claims checked")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  {p}")
        return 1
    print("all clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
