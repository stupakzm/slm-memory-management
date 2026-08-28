#!/usr/bin/env python3
"""Verify the eval set against the corpus and fill in gold section ids.

Every answerable question must name a real doc containing a section where all of
its answer tokens appear; every "tool not installed" question must name a tool
that genuinely has no page. This is what stops the eval set from encoding the
author's recall instead of what is actually on the machine.

Usage: resolve_gold.py [--write]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus" / "man.jsonl"
EVAL = ROOT / "data" / "eval" / "questions.jsonl"


def load_corpus():
    docs, names = {}, {}
    for line in CORPUS.open(encoding="utf-8"):
        d = json.loads(line)
        docs[d["doc_id"]] = d
        names.setdefault(d["name"], []).append(d["doc_id"])
        for a in d["aliases"]:
            names.setdefault(a, []).append(d["doc_id"])
    return docs, names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write resolved gold ids back")
    args = ap.parse_args()

    docs, names = load_corpus()
    rows = [json.loads(l) for l in EVAL.open(encoding="utf-8")]

    errors, warnings = [], []
    ids = {r["qid"] for r in rows}
    for r in rows:
        base_id = r.get("variant_of") or r.get("paraphrase_of")
        if base_id and base_id not in ids:
            errors.append(f"{r['qid']}: derives from unknown question {base_id!r}")

        if r["kind"] == "answerable":
            doc = docs.get(r["doc"])
            if doc is None:
                errors.append(f"{r['qid']}: no such doc {r['doc']!r}")
                continue
            toks = r["answer_contains"]
            matches = [s["sec_id"] for s in doc["sections"] if all(t in s["text"] for t in toks)]
            if not matches:
                near = [t for t in toks if not any(t in s["text"] for s in doc["sections"])]
                errors.append(f"{r['qid']}: no section of {r['doc']} holds all of {toks} (missing: {near})")
                continue
            if len(matches) > 5:
                warnings.append(f"{r['qid']}: {len(matches)} sections match {toks} - tokens too generic")
            r["gold_sec_ids"] = matches
            # The primary is the section a reader would be sent to: an options or
            # command reference before prose, shortest before longest.
            def rank(sec_id: str) -> tuple:
                heading = sec_id.split("#", 1)[1].upper()
                for i, kw in enumerate(("OPTIONS", "COMMANDS", "ACTIONS", "EXPRESSION", "DESCRIPTION")):
                    if kw in heading:
                        return (i, len(heading))
                return (9, len(heading))
            r["gold_primary"] = sorted(matches, key=rank)[0]
            # Leak check: the question must not contain its own answer token.
            q = r["question"].lower()
            leaked = [t for t in toks if len(t) > 3 and t.lower() in q]
            if leaked:
                warnings.append(f"{r['qid']}: question leaks answer token(s) {leaked}")
        else:
            reason = r.get("unanswerable_reason")
            detail = r.get("unanswerable_detail") or ""
            if reason == "tool-not-installed":
                if detail in names:
                    errors.append(f"{r['qid']}: {detail!r} IS installed ({names[detail][0]}) - not unanswerable")
            elif reason == "out-of-corpus":
                # This check did not exist, and that is how a wrong label survived
                # three phases: u22 claimed pip was out of corpus while pip.1,
                # pip-install.1 and pip3-install.1 were all indexed, and every phase
                # scored the documented answer as a hallucination. An out-of-corpus
                # question must now name the page it would need, and that page must
                # genuinely be absent.
                if not detail:
                    errors.append(f"{r['qid']}: out-of-corpus but names no needed page")
                elif detail in docs:
                    errors.append(f"{r['qid']}: claims {detail!r} is out of corpus, "
                                  f"but it is indexed")

    print(f"questions            : {len(rows)}")
    print(f"  answerable         : {sum(1 for r in rows if r['kind']=='answerable')}")
    print(f"  unanswerable       : {sum(1 for r in rows if r['kind']=='unanswerable')}"
          f" ({sum(1 for r in rows if r['kind']=='unanswerable')/len(rows):.0%})")
    print(f"  paraphrases        : {sum(1 for r in rows if r['paraphrase_of'])}")
    vk = Counter(r.get("variant_kind") for r in rows if r.get("variant_kind"))
    print(f"  query-noise variants: {sum(vk.values())} {dict(vk)}")
    resolved = [r for r in rows if r["kind"] == "answerable" and r["gold_sec_ids"]]
    print(f"  gold resolved      : {len(resolved)}")
    gold_docs = Counter(r["doc"] for r in resolved)
    print(f"  distinct gold docs : {len(gold_docs)}")
    print(f"  tags               : {dict(Counter(t for r in rows for t in r['tags']).most_common())}")

    if warnings:
        print(f"\n--- {len(warnings)} warning(s) ---")
        for w in warnings:
            print("  ! " + w)
    if errors:
        print(f"\n--- {len(errors)} ERROR(s) ---")
        for e in errors:
            print("  x " + e)

    if args.write and not errors:
        with EVAL.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("\nwrote resolved gold ids")
    elif args.write:
        print("\nNOT written - fix errors first")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
