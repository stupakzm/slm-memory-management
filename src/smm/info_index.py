"""Parse the index node of a GNU info file into (entry, node, line) triples.

A makeinfo index node lists one entry per `* ` line, wrapped when the entry text is
long (the real shape, from the Emacs manual's `Concept Index`):

    * .emacs file:                           Init File.          (line    6)
    * ? in display:                          International Chars.
                                                                 (line   18)

The entry text is everything between `* ` and the last colon followed by whitespace
and a node name; the node is the text up to the period before `(line N)`. Entries can
contain colons and punctuation; node names do not contain colons.

Stdlib only. scripts/extract_info.py walks the same `\\x1f`-separated segments; this
module re-implements the walk rather than importing a script.
"""

from __future__ import annotations

import re

SEP = "\x1f"
_HEADER = re.compile(r"File:[^\n]*?Node:\s*([^,\n]+)")
_ENTRY = re.compile(r"^(.+):\s+([^:]+?)\.\s+\(line\s+(\d+)\)$")
_LINE_END = re.compile(r"\(line\s+\d+\)\s*$")


def _node_of(segment: str) -> str | None:
    m = _HEADER.match(segment.lstrip("\n"))
    return m.group(1).strip() if m else None


def _entries(body: str):
    cur = None
    for line in body.split("\n"):
        if line.startswith("* "):
            if cur is not None:
                yield cur
            cur = line[2:].strip()
        elif (cur is not None and line[:1] in (" ", "\t") and line.strip()
              and not _LINE_END.search(cur)):
            cur += " " + line.strip()
        else:
            if cur is not None:
                yield cur
            cur = None
    if cur is not None:
        yield cur


def parse_index(info_text: str, node_name: str = "Concept Index") -> list[dict]:
    """[{"entry", "node", "line"}] of the segment(s) whose header names `node_name`,
    in file order, exact duplicates dropped."""
    out, seen = [], set()
    for seg in info_text.split(SEP):
        if _node_of(seg) != node_name:
            continue
        for text in _entries(seg):
            m = _ENTRY.match(text)
            if not m:
                continue
            item = (m.group(1).strip(), m.group(2).strip(), int(m.group(3)))
            if item in seen:
                continue
            seen.add(item)
            out.append({"entry": item[0], "node": item[1], "line": item[2]})
    return out


def chunk_for(chunks: list[tuple], offset: int) -> tuple | None:
    """chunks: (chunk_id, char_start, char_end) of one document. The chunk with the
    largest char_start <= offset whose char_end > offset; else the chunk with the
    smallest char_start > offset; None when there are no chunks."""
    inside = [c for c in chunks if c[1] <= offset < c[2]]
    if inside:
        return max(inside, key=lambda c: c[1])
    after = [c for c in chunks if c[1] > offset]
    return min(after, key=lambda c: c[1]) if after else None


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def find_section(offsets: list[tuple[int, str]], node: str):
    """The (offset, heading) entry whose heading equals `node` ignoring case and runs
    of whitespace, else None."""
    want = _norm(node)
    for off, heading in offsets:
        if _norm(heading) == want:
            return (off, heading)
    return None
