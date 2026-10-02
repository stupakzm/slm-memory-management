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
  description the token is a *phrase* (it does not start with `-`) that
              occurs verbatim in an option entry's description text
              (`sort by time` under `-t`); that entry's options become its
              aliases. A flag token (`-t`, `--location`) never gets a
              description alias - man pages cross-reference other flags by
              name in prose (`-t`'s description mentions `--full-time`), and
              that is not a synonym.

Every alias is recorded with the rule that produced it, the section it came
from, and the option line itself - so any alias can be traced back to the
one man page sentence that licenses it, and none of this ever looks at a
model's answer. Every alias, regardless of rule, is itself a flag (starts
with `-`); anything else is dropped.

Two real-corpus defects fixed here (attempt 2):
  - an option line's inline description (`-t     sort by time, newest
    first`, description on the same physical line, separated by a run of
    2+ spaces or a tab) was being fed whole into the comma/space option
    parser, producing bogus "options" like `newest`. The line is now split
    at the first such run before parsing: the left part is the option
    spec, the right part joins the entry's description text.
  - the description rule was firing for flag tokens too (`-t`'s
    description mentioning `--full-time` made `--full-time` an alias of
    `-t`, and vice versa), which is backwards - see the `description` rule
    above.

R1 (tsk_20260927_keyalias) adds a second, independent mode for the Emacs
md-corpus eval set: a "keybinding" rule that pairs a documented key
sequence with the command it runs, from the corpus text of the row's own
gold section(s) alone - never from an answer. Two shapes license a pair:

  inline pair       'KEY' (‘COMMAND’), close together (same sentence, at
                     most a few words, or one line break) - e.g. "typing
                     ‘C-M-w’ (‘append-next-kill’) right beforehand".
  definition list    one or more header lines that are exactly ‘...’ at
                     column 0, followed by an indented description whose
                     first (‘COMMAND’) names the entry's command - every
                     header is a key of that command - e.g. "‘% m REGEXP
                     <RET>’\\n‘* % REGEXP <RET>’\\n     Mark ... REGEXP
                     (‘dired-mark-files-regexp’)."

Direction follows the row's own token: a command token's aliases are its
keys; a key token's alias is the command. Keys are normalised (trailing
ALL-CAPS placeholder words and `<RET>` stripped) and must carry a
modifier (C-/M-/s-/H-/A-), be a function key (`<F3>`), or start with a
prefix character (`%`, `*`) - a bare single character or plain word is
never accepted as a key, so it can never inflate what counts correct.

Alternatives (--alternatives FILE, --md-corpus mode only): a one-token label
under-credits a correct answer that names a *different documented command*
for the same need, or a key the keybinding rule cannot parse because it sits
in prose. Deciding which neighbour really answers a question is judgment, so
FILE is adjudicated by hand: a JSON object keyed by qid, each value a list of
{"token", "alias", "sec_id", "line", "why"} (a row may map to []). It is
merged as two rules, after that row's keybinding entries:

  alternative      the FILE entry itself, licensed by `line`, a verbatim line
                   of the row's own gold section `sec_id`.
  alternative-key  mechanical: when an alternative alias is command-shaped,
                   every keybinding pair in the row's gold sections whose
                   command equals it adds its key (with `via` = the command).

Deduped on (alias, sec_id) per token; an alias equal to the token is dropped.
Without --alternatives the output is byte-identical to the keybinding mode.
--check polices the FILE: the qid must be in --eval and answerable, `token`
in answer_contains, `sec_id` in gold_sec_ids, `line` verbatim in that
section, `alias` a substring of `line` and command- or key-shaped, `why`
non-empty, and no alias (alternative or alternative-key) may occur in the
question text (case-insensitive) - an answer echoing the question must never
score. --check also flags a stale --out: for every qid in this --eval the
--out entry must equal the fresh derivation (without --alternatives only the
`keybinding` entries are compared). Merging with --alternatives refuses to
write if any violation is found.

Usage: derive_gold_aliases.py [--corpus data/corpus/man.jsonl]
                               [--eval data/eval/questions.jsonl]
                               [--out data/eval/gold_aliases.json]
       derive_gold_aliases.py --md-corpus DIR --eval EVAL.jsonl
                               [--merge] [--check] [--alternatives FILE]
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
_INLINE_SPLIT_RE = re.compile(r"\t| {2,}")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _split_option_spec(line: str) -> tuple[str, str]:
    """Split an option line at the first run of 2+ spaces or a tab: the left
    part is the option spec (parsed for flags/args), the right part is
    inline description text belonging to the same entry (`-t     sort by
    time, newest first` -> ("-t", "sort by time, newest first")). No such
    run means the whole line is the option spec and there is no inline
    description."""
    m = _INLINE_SPLIT_RE.search(line)
    if not m:
        return line, ""
    return line[:m.start()].rstrip(), line[m.end():].strip()


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
            opt_spec, inline_desc = _split_option_spec(stripped)
            desc_lines: list[str] = [inline_desc] if inline_desc else []
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
            units = parse_option_line(opt_spec)
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
        if not alias.startswith("-"):
            return  # every alias, in every rule, must itself be a flag
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

            # description: token is a phrase (not a flag) in this entry's
            # description text. A flag token never gets a description alias:
            # man pages routinely name other flags in prose (-t's
            # description mentions --full-time), and that is not a synonym.
            if (not token.startswith("-") and entry["description"]
                    and token in entry["description"]):
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


# --- R1 (tsk_20260927_keyalias): keybinding rule for the Emacs md-corpus set ---

LQ = "‘"  # ‘
RQ = "’"  # ’

_HEADER_RE = re.compile(rf"^{LQ}([^{LQ}{RQ}]+){RQ}$")
_COMMAND_SHAPE = r"[a-z0-9]+(?:-[a-z0-9]+)+"
_COMMAND_SHAPE_RE = re.compile(rf"^{_COMMAND_SHAPE}$")
_CMD_PARENS_RE = re.compile(rf"\({LQ}({_COMMAND_SHAPE}){RQ}\)")
_INLINE_PAIR_RE = re.compile(
    rf"{LQ}([^{LQ}{RQ}]{{1,40}}){RQ}"       # ‘KEY’
    rf"([^{LQ}{RQ}]{{0,60}})"                # short gap, no other quotes crossed
    rf"\({LQ}({_COMMAND_SHAPE}){RQ}\)"        # (‘COMMAND’)
)
_PLACEHOLDER_WORD_RE = re.compile(r"^[A-Z][A-Z0-9]*$")
_MODIFIER_RE = re.compile(r"(?:^|\s)(?:C|M|s|H|A)-")
_FUNCKEY_RE = re.compile(r"^<[A-Za-z][A-Za-z0-9]*>$")
_INLINE_GAP_MAX_WORDS = 4


def normalize_key(raw: str) -> str:
    """Strip trailing argument placeholders: ALL-CAPS words (REGEXP, BUFFER,
    KEY, R) and the literal `<RET>`. "C-x k BUFFER <RET>" -> "C-x k"."""
    toks = raw.split()
    while toks:
        last = toks[-1]
        if last == "<RET>" or _PLACEHOLDER_WORD_RE.match(last):
            toks.pop()
            continue
        break
    return " ".join(toks)


def is_key_shaped(key: str) -> bool:
    """A modifier (C-/M-/s-/H-/A-), a function key (<F3>), or a prefix
    character (%, *) - never a bare single character or a plain word.
    `M-x <command-name>` is Emacs' "run this command by name" prefix, not a
    modifier keystroke, and is never a key even though it starts with `M-`."""
    if not key or len(key) <= 1:
        return False  # a bare single character (including a bare '%' or '*')
    if key == "M-x" or key.startswith("M-x "):
        return False
    if _MODIFIER_RE.search(key):
        return True
    if _FUNCKEY_RE.match(key):
        return True
    if key[0] in "%*":
        return True
    return False


def find_definition_groups(text: str) -> list[dict]:
    """One or more consecutive column-0 lines that are exactly ‘...’,
    followed by the indented (or blank) description up to the next
    column-0 non-blank line. Each header maps to the FIRST (‘command’)
    found anywhere in that description block."""
    lines = text.split("\n")
    i, n = 0, len(lines)
    out: list[dict] = []
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if stripped and _indent(line) == 0 and _HEADER_RE.match(stripped):
            headers: list[tuple[str, str]] = []
            while i < n:
                s2 = lines[i].strip()
                if s2 and _indent(lines[i]) == 0 and _HEADER_RE.match(s2):
                    headers.append((_HEADER_RE.match(s2).group(1), s2))
                    i += 1
                else:
                    break
            # The description block is indented relative to the header (0);
            # a plain prose paragraph elsewhere in the section is *also*
            # indented (this corpus wraps body text at 3 spaces) but less
            # than a definition's own description (5 spaces here) - so the
            # block's own indent, fixed from its first non-blank line, is
            # the boundary: a shallower non-blank line ends the block even
            # though its indent is still > 0.
            block: list[str] = []
            desc_indent: int | None = None
            while i < n:
                nxt = lines[i]
                if not nxt.strip():
                    block.append(nxt)
                    i += 1
                    continue
                nxt_indent = _indent(nxt)
                if nxt_indent == 0:
                    break
                if desc_indent is None:
                    desc_indent = nxt_indent
                if nxt_indent < desc_indent:
                    break
                block.append(nxt)
                i += 1
            block_text = "\n".join(block)
            m = _CMD_PARENS_RE.search(block_text)
            if m:
                command = m.group(1)
                cmd_line = next(
                    (bl.strip() for bl in block if _CMD_PARENS_RE.search(bl)), ""
                )
                for key_raw, header_line in headers:
                    out.append({
                        "key_raw": key_raw, "command": command,
                        "key_line": header_line, "cmd_line": cmd_line,
                    })
        else:
            i += 1
    return out


def find_inline_pairs(text: str) -> list[dict]:
    """‘KEY’ (‘COMMAND’), separated by at most a few words and at most one
    line break."""
    out: list[dict] = []
    for m in _INLINE_PAIR_RE.finditer(text):
        key_raw, gap, command = m.group(1), m.group(2), m.group(3)
        if len(gap.split()) > _INLINE_GAP_MAX_WORDS or gap.count("\n") > 1:
            continue
        line = m.group(0).strip()
        out.append({"key_raw": key_raw, "command": command,
                     "key_line": line, "cmd_line": line})
    return out


def keybinding_pairs_in_section(text: str) -> list[dict]:
    return find_definition_groups(text) + find_inline_pairs(text)


def keybinding_matches_for_row(sections: dict, row: dict) -> dict:
    """{token: [alias-dict, ...]} for `row`'s answer_contains tokens, using
    only the corpus text of `row`'s own gold_sec_ids."""
    per_token: dict[str, list] = {}
    for sec_id in row.get("gold_sec_ids", []):
        text = sections.get(sec_id)
        if text is None:
            continue
        pairs = []
        for p in keybinding_pairs_in_section(text):
            pairs.append({
                "key": normalize_key(p["key_raw"]), "command": p["command"],
                "key_line": p["key_line"], "cmd_line": p["cmd_line"],
            })
        for token in row.get("answer_contains", []):
            for p in pairs:
                if token == p["command"] and is_key_shaped(p["key"]):
                    per_token.setdefault(token, []).append({
                        "alias": p["key"], "rule": "keybinding",
                        "sec_id": sec_id, "line": p["key_line"],
                    })
                elif token == p["key"] and _COMMAND_SHAPE_RE.match(p["command"]):
                    per_token.setdefault(token, []).append({
                        "alias": p["command"], "rule": "keybinding",
                        "sec_id": sec_id, "line": p["cmd_line"],
                    })
    for token, entries in per_token.items():
        seen = set()
        deduped = []
        for e in entries:
            key = (e["alias"], e["sec_id"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(e)
        per_token[token] = deduped
    return per_token


def load_md_corpus(md_dir: Path) -> dict:
    """{doc_id: {"doc": doc-dict, "sections": {sec_id: text}}} for every
    *.md in md_dir, converted the way the Emacs eval set's docs were:
    smm.ingest.from_file(Path(f), 'emacs')."""
    sys.path.insert(0, str(ROOT / "src"))
    from smm import ingest  # noqa: E402

    docs = {}
    for f in sorted(Path(md_dir).glob("*.md")):
        d = ingest.from_file(f, "emacs")
        docs[d["doc_id"]] = {
            "doc": d,
            "sections": {s["sec_id"]: s["text"] for s in d["sections"]},
        }
    return docs


def load_alternatives(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _valid_alt_entries(entries) -> list[dict]:
    return [e for e in entries if isinstance(e, dict)
            and all(isinstance(e.get(k), str) and e.get(k)
                    for k in ("token", "alias", "sec_id", "line"))]


def apply_alternatives(sections: dict, row: dict, entries: list,
                       per_token: dict) -> None:
    """Append `alternative` entries (and the mechanically derived
    `alternative-key` ones) for `row` to per_token, after its keybinding
    entries. Dedupe on (alias, sec_id) per token; alias == token is dropped.
    Validation is --check's job; entries naming a token outside the row's
    answer_contains are skipped here."""
    answer = set(row.get("answer_contains", []))
    for e in _valid_alt_entries(entries):
        token, alias = e["token"], e["alias"]
        if token not in answer or alias == token:
            continue
        got = per_token.setdefault(token, [])
        seen = {(x["alias"], x["sec_id"]) for x in got}

        def add(entry: dict) -> None:
            k = (entry["alias"], entry["sec_id"])
            if entry["alias"] != token and k not in seen:
                seen.add(k)
                got.append(entry)

        add({"alias": alias, "rule": "alternative",
             "sec_id": e["sec_id"], "line": e["line"]})
        if _COMMAND_SHAPE_RE.match(alias):
            for sec_id in row.get("gold_sec_ids", []):
                text = sections.get(sec_id)
                if text is None:
                    continue
                for p in keybinding_pairs_in_section(text):
                    key = normalize_key(p["key_raw"])
                    if p["command"] == alias and is_key_shaped(key):
                        add({"alias": key, "rule": "alternative-key",
                             "sec_id": sec_id, "line": p["key_line"],
                             "via": alias})
        if not got:
            del per_token[token]


def derive_keybinding(docs: dict, rows: list[dict],
                      alternatives: dict | None = None) -> tuple[dict, dict]:
    """Returns (aliases, stats), same shape as derive()."""
    aliases: dict[str, dict] = {}
    per_rule = {"keybinding": 0}
    if alternatives is not None:
        per_rule.update({"alternative": 0, "alternative-key": 0})
    touched: set[str] = set()

    for row in rows:
        if row.get("kind") != "answerable":
            continue
        doc = docs.get(row.get("doc"))
        if doc is None:
            continue
        qid = row["qid"]
        per_token = keybinding_matches_for_row(doc["sections"], row)
        if alternatives is not None and isinstance(alternatives.get(qid), list):
            apply_alternatives(doc["sections"], row, alternatives[qid], per_token)
        if per_token:
            touched.add(qid)
            for entries in per_token.values():
                for e in entries:
                    per_rule[e["rule"]] += 1
        aliases[qid] = per_token

    stats = {"per_rule": per_rule, "questions_touched": len(touched)}
    return aliases, stats


def check_alternatives(alternatives: dict, aliases: dict, docs: dict,
                       rows: list[dict]) -> list[str]:
    """Violations in the adjudicated FILE and in what it derived. `aliases`
    is the fresh derivation (for the alternative-key question-echo check)."""
    rows_by_qid = {r["qid"]: r for r in rows}
    violations: list[str] = []
    if not isinstance(alternatives, dict):
        return ["alternatives file is not a JSON object keyed by qid"]
    for qid, entries in alternatives.items():
        row = rows_by_qid.get(qid)
        if row is None:
            violations.append(f"{qid}: alternatives for a qid not in this --eval")
            continue
        if row.get("kind") != "answerable":
            violations.append(f"{qid}: alternatives for a non-answerable row")
            continue
        if not isinstance(entries, list):
            violations.append(f"{qid}: alternatives value is not a list")
            continue
        sections = docs.get(row.get("doc"), {}).get("sections", {})
        gold_secs = set(row.get("gold_sec_ids", []))
        for e in entries:
            if not isinstance(e, dict):
                violations.append(f"{qid}: alternative entry is not an object")
                continue
            token, alias = e.get("token"), e.get("alias")
            sec_id, line = e.get("sec_id"), e.get("line")
            tag = f"{qid}/{token}"
            if not all(isinstance(x, str) and x for x in (token, alias, sec_id, line)):
                violations.append(f"{tag}: token/alias/sec_id/line must be non-empty strings")
                continue
            if token not in row.get("answer_contains", []):
                violations.append(f"{tag}: token not in the row's answer_contains")
            if not (isinstance(e.get("why"), str) and e["why"].strip()):
                violations.append(f"{tag}: alias {alias!r} has no why")
            if sec_id not in gold_secs:
                violations.append(f"{tag}: sec_id {sec_id!r} not in gold_sec_ids")
            else:
                text = sections.get(sec_id)
                if text is None:
                    violations.append(f"{tag}: no such section {sec_id!r}")
                elif line not in text:
                    violations.append(
                        f"{tag}: line {line!r} not verbatim in section {sec_id!r}")
            if alias not in line:
                violations.append(f"{tag}: alias {alias!r} not a substring of its line")
            if not (_COMMAND_SHAPE_RE.match(alias)
                    or is_key_shaped(normalize_key(alias))):
                violations.append(
                    f"{tag}: alias {alias!r} is neither command- nor key-shaped")
    for qid, per_token in aliases.items():
        question = rows_by_qid.get(qid, {}).get("question", "").lower()
        for token, entries in per_token.items():
            for e in entries:
                if (e.get("rule") in ("alternative", "alternative-key")
                        and e["alias"].lower() in question):
                    violations.append(
                        f"{qid}/{token}: {e['rule']} alias {e['alias']!r} "
                        f"occurs in the question text")
    return violations


def check_stale(aliases: dict, existing: dict, rows: list[dict],
                with_alternatives: bool) -> list[str]:
    """For each qid in this --eval the --out entry must equal the fresh
    derivation; without --alternatives only `keybinding` entries compare."""
    def view(entry):
        out = {}
        for token, entries in (entry or {}).items():
            if not with_alternatives:
                entries = [e for e in entries if e.get("rule") == "keybinding"]
            if entries:
                out[token] = entries
        return out

    violations = []
    for qid in sorted({r["qid"] for r in rows}):
        if view(existing.get(qid)) != view(aliases.get(qid)):
            violations.append(
                f"{qid}: --out entry is stale (differs from a fresh derivation)")
    return violations


def check_keybinding(aliases: dict, docs: dict, rows: list[dict]) -> list[str]:
    """Every violation of the invariants R1 promises. Empty means --check
    passes."""
    rows_by_qid = {r["qid"]: r for r in rows}
    violations: list[str] = []
    for qid, per_token in aliases.items():
        row = rows_by_qid.get(qid)
        if row is None:
            violations.append(f"{qid}: not a row in this --eval")
            continue
        gold_secs = set(row.get("gold_sec_ids", []))
        doc = docs.get(row.get("doc"), {})
        sections = doc.get("sections", {})
        for token, entries in per_token.items():
            for e in entries:
                if e.get("rule") != "keybinding":
                    continue
                alias, sec_id, line = e["alias"], e["sec_id"], e.get("line", "")
                if sec_id not in gold_secs:
                    violations.append(
                        f"{qid}/{token}: sec_id {sec_id!r} not in gold_sec_ids")
                    continue
                text = sections.get(sec_id)
                if text is None:
                    violations.append(f"{qid}/{token}: no such section {sec_id!r}")
                    continue
                if token not in text:
                    violations.append(
                        f"{qid}/{token}: token not found in section {sec_id!r}")
                if alias not in text:
                    violations.append(
                        f"{qid}/{token}: alias {alias!r} not found in section {sec_id!r}")
                if line and line not in text:
                    violations.append(
                        f"{qid}/{token}: line {line!r} not verbatim in section {sec_id!r}")
                if _COMMAND_SHAPE_RE.match(token):
                    # command token -> alias is the key
                    if not is_key_shaped(alias):
                        violations.append(
                            f"{qid}/{token}: alias {alias!r} is not key-shaped")
                else:
                    # key token -> alias is the command
                    if not _COMMAND_SHAPE_RE.match(alias):
                        violations.append(
                            f"{qid}/{token}: alias {alias!r} is not command-shaped")
    return violations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/corpus/man.jsonl")
    ap.add_argument("--eval", default="data/eval/questions.jsonl")
    ap.add_argument("--out", default="data/eval/gold_aliases.json")
    ap.add_argument("--md-corpus", default=None,
                     help="directory of Emacs manual .md files (keybinding rule mode)")
    ap.add_argument("--merge", action="store_true",
                     help="merge into --out, replacing only this --eval's qids, "
                          "leaving every other qid byte-identical (--md-corpus only)")
    ap.add_argument("--check", action="store_true",
                     help="validate derived keybinding aliases; exit 1 on any "
                          "violation (--md-corpus only)")
    ap.add_argument("--alternatives", default=None,
                     help="JSON file of adjudicated alternative answers "
                          "(--md-corpus only)")
    args = ap.parse_args()

    eval_path = ROOT / args.eval
    out_path = ROOT / args.out

    if not eval_path.exists():
        print(f"no eval set at {eval_path}", file=sys.stderr)
        return 2

    if args.md_corpus:
        md_dir = ROOT / args.md_corpus
        if not md_dir.exists():
            print(f"no md corpus at {md_dir}", file=sys.stderr)
            return 2
        docs = load_md_corpus(md_dir)
        rows = load_questions(eval_path)
        alternatives = None
        if args.alternatives:
            alt_path = ROOT / args.alternatives
            if not alt_path.exists():
                print(f"no alternatives file at {alt_path}", file=sys.stderr)
                return 2
            alternatives = load_alternatives(alt_path)
        aliases, stats = derive_keybinding(docs, rows, alternatives)

        if args.check:
            violations = check_keybinding(aliases, docs, rows)
            if alternatives is not None:
                violations += check_alternatives(alternatives, aliases, docs, rows)
            existing = (json.loads(out_path.read_text())
                        if out_path.exists() else {})
            violations += check_stale(aliases, existing, rows,
                                      alternatives is not None)
            for v in violations:
                print("VIOLATION: " + v)
            if violations:
                print(f"\n{len(violations)} violation(s)")
                return 1
            extra = "".join(f", {stats['per_rule'][r]} {r}"
                            for r in ("alternative", "alternative-key")
                            if r in stats["per_rule"])
            print(f"check OK: {stats['per_rule']['keybinding']} keybinding "
                  f"aliases{extra}, {stats['questions_touched']} questions, "
                  f"0 violations")
            return 0

        if alternatives is not None:
            violations = check_alternatives(alternatives, aliases, docs, rows)
            for v in violations:
                print("VIOLATION: " + v)
            if violations:
                print(f"\n{len(violations)} violation(s); nothing written")
                return 1

        print(f"keybinding aliases derived: {stats['per_rule']['keybinding']}")
        if alternatives is not None:
            for r in ("alternative", "alternative-key"):
                print(f"{r} aliases derived: {stats['per_rule'][r]}")
        print(f"questions touched: {stats['questions_touched']}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.merge:
            existing = json.loads(out_path.read_text()) if out_path.exists() else {}
            eval_qids = {r["qid"] for r in rows}
            merged = {k: v for k, v in existing.items() if k not in eval_qids}
            merged.update(aliases)
            out_path.write_text(json.dumps(merged, indent=2, sort_keys=True))
            print(f"merged into {out_path}")
        else:
            out_path.write_text(json.dumps(aliases, indent=2, sort_keys=True))
            print(f"wrote {out_path}")
        return 0

    corpus_path = ROOT / args.corpus

    if not corpus_path.exists():
        print(f"no corpus at {corpus_path}", file=sys.stderr)
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
