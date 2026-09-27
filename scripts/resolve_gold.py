#!/usr/bin/env python3
"""Verify the eval set against the corpus and fill in gold section ids.

Every answerable question must name a real doc containing a section where all of
its answer tokens appear; every "tool not installed" question must name a tool
that genuinely has no page. This is what stops the eval set from encoding the
author's recall instead of what is actually on the machine.

A row may also carry "gold_hint": a list of strings that narrow which section
counts as gold (useful when the real answer token alone matches many sections)
without inflating what eval_answers/gold.is_correct requires of a model's
answer. gold_hint strings are required alongside answer_contains tokens when
locating the gold section, but are never scored and never leak-checked.

Usage: resolve_gold.py [--write]
       resolve_gold.py [--corpus PATH] [--eval PATH] [--write]
       resolve_gold.py --md-corpus DIR --domain D [--eval PATH] [--write]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus" / "man.jsonl"
EVAL = ROOT / "data" / "eval" / "questions.jsonl"


def load_corpus(corpus_path: Path):
    docs, names = {}, {}
    for line in corpus_path.open(encoding="utf-8"):
        d = json.loads(line)
        docs[d["doc_id"]] = d
        names.setdefault(d["name"], []).append(d["doc_id"])
        for a in d["aliases"]:
            names.setdefault(a, []).append(d["doc_id"])
    return docs, names


def load_md_corpus(md_dir: Path, domain: str):
    """Load a directory of .md files (one '## <node>' heading per section) the
    same way scripts/ingest.py would, keyed by doc_id. Aliases are empty: a
    freshly-ingested markdown corpus has no man-page alias table."""
    sys.path.insert(0, str(ROOT / "src"))
    from smm import ingest  # noqa: E402

    docs, names = {}, {}
    for f in sorted(md_dir.glob("*.md")):
        d = ingest.from_file(f, domain)
        docs[d["doc_id"]] = d
        names.setdefault(d["name"], []).append(d["doc_id"])
    return docs, names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write resolved gold ids back")
    ap.add_argument("--corpus", type=Path, default=CORPUS,
                     help="man-page corpus jsonl (ignored with --md-corpus)")
    ap.add_argument("--eval", type=Path, default=EVAL, help="questions jsonl to check")
    ap.add_argument("--md-corpus", type=Path, default=None,
                     help="directory of domain .md files, one '## <node>' per section")
    ap.add_argument("--domain", default=None, help="domain name for --md-corpus")
    args = ap.parse_args()

    md_mode = args.md_corpus is not None
    if md_mode:
        if not args.domain:
            ap.error("--md-corpus requires --domain")
        docs, names = load_md_corpus(args.md_corpus, args.domain)
    else:
        docs, names = load_corpus(args.corpus)
    rows = [json.loads(l) for l in args.eval.open(encoding="utf-8")]

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
            hints = r.get("gold_hint") or []
            needed = toks + hints
            matches = [s["sec_id"] for s in doc["sections"] if all(t in s["text"] for t in needed)]
            if not matches:
                near = [t for t in toks if not any(t in s["text"] for s in doc["sections"])]
                errors.append(f"{r['qid']}: no section of {r['doc']} holds all of {needed} (missing: {near})")
                continue
            if len(matches) > 5:
                warnings.append(f"{r['qid']}: {len(matches)} sections match {needed} - tokens too generic")
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
                if md_mode:
                    # Stronger than the name check: the Emacs FAQ names many
                    # third-party packages in prose without them being "installed"
                    # as a doc, so a name-only check would pass questions whose
                    # answer is sitting right there in the text.
                    detail_lower = detail.lower()
                    hit = next((d["doc_id"] for d in docs.values()
                                if any(detail_lower in s["text"].lower()
                                       for s in d["sections"])), None)
                    if hit:
                        errors.append(f"{r['qid']}: {detail!r} IS mentioned in the corpus "
                                      f"({hit}) - not unanswerable")
                elif detail in names:
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
        with args.eval.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("\nwrote resolved gold ids")
    elif args.write:
        print("\nNOT written - fix errors first")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
