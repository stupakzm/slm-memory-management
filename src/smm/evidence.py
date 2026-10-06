"""Show the line of a cited extract that contains an identifier the answer names.

Stdlib + smm.grammar only. The answer says "use --foo [1]"; this finds the
first line of extract 1 that actually contains --foo, so a reader can check the
claim without opening the page, and says where to read more.
"""

from __future__ import annotations

import re

from smm import grammar

MAX_LINE = 200

_LONG = re.compile(r"--[A-Za-z][A-Za-z0-9-]*")
_SHORT = re.compile(r"(?<![\w-])-[A-Za-z0-9](?![\w-])")
_CHORD = re.compile(r"(?:[CMs]-)+\S+")
_MX = re.compile(r"M-x\s+(\S+)")
_MAN_DOC = re.compile(r"^(.+)\.(\d[A-Za-z]*)$")


def _strip(tok: str) -> str:
    return tok.rstrip(".,;:)'\"`")


def _identifiers(answer: str) -> list[str]:
    """Flags, key chords and M-x commands the answer names, citations removed."""
    text = grammar.CITE_RE.sub("", answer)
    found: list[str] = []
    found += [_strip(m) for m in _LONG.findall(text)]
    found += _SHORT.findall(text)
    found += [_strip(m) for m in _CHORD.findall(text)]
    found += [_strip(m) for m in _MX.findall(text)]
    return list(dict.fromkeys(i for i in found if i))


def _line_has(line: str, ident: str) -> bool:
    if ident.startswith("-") and not ident.startswith(("C-", "M-", "s-")):
        return re.search(r"(?<![\w-])" + re.escape(ident) + r"(?![\w-])", line) is not None
    return ident in line


def pointer_for(hit: dict) -> str:
    doc_id = hit["doc_id"]
    m = _MAN_DOC.match(doc_id)
    if m:
        return f"man {m.group(2)} {m.group(1)}"
    sec_id = hit.get("sec_id") or ""
    section = sec_id.split("#", 1)[1] if "#" in sec_id else doc_id
    return f"Info manual {doc_id.split('.', 1)[0]}, section {section}"


def evidence_lines(answer: str, hits: list[dict]) -> list[dict]:
    if answer.strip() == grammar.REFUSAL:
        return []
    idents = _identifiers(answer)
    if not idents:
        return []
    out = []
    for n in grammar.citations(answer):
        if not 1 <= n <= len(hits):
            continue
        hit = hits[n - 1]
        body = f"{hit.get('prefix', '')}{hit['text']}"
        for raw in body.split("\n"):
            line = raw.strip()
            if line and any(_line_has(line, i) for i in idents):
                out.append({"n": n, "line": line[:MAX_LINE], "pointer": pointer_for(hit)})
                break
    return out
