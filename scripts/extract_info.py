#!/usr/bin/env python3
"""Convert GNU info manuals (single-file, gzip-compressed) into the markdown
shape `src/smm/ingest.py` already sections: `## <heading>` lines that
`_sections_from_markdown` turns into `sec_id = doc_id#slug(heading)`.

Usage: extract_info.py DIR --out OUT

For every DIR/*.info.gz (processed in sorted order), writes OUT/<manual>.md,
where <manual> is the filename stem before `.info.gz`.

Info-file shape this handles (see docstrings below for the real examples that
justify each rule):
  - `\\x1f` (ASCII unit separator) splits the decompressed text into segments.
    The first segment is a makeinfo preamble and is dropped; `Tag Table:`,
    `End Tag Table`, and `Local Variables:` segments are not nodes and are
    dropped too - all three are recognised because, unlike a node segment,
    they do not start with `File:` once leading newlines are stripped.
  - A node segment's first line is `File: NAME.info,  Node: X,  Next: Y, ...`.
    The node name is the text after `Node: ` up to the next comma. That
    header line is dropped; the rest of the segment is the node's body.
  - Nodes whose name ends in "Index" are skipped entirely.
  - A `* Menu:` line (alone on its line) starts a menu block: Texinfo table-
    of-contents scaffolding (entries, wrapped descriptions, group-label
    lines, and in Top nodes a "-- The Detailed Node Listing --" divider with
    its own labelled sub-lists) that is not prose and gets stripped. Real
    prose that happens to follow a menu (footnotes, or ordinary paragraphs)
    is kept - see `strip_menu` for how the two are told apart.
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smm.ingest import slug  # noqa: E402  (same slug() ingest.py will apply)

HEADER_LINE_RE = re.compile(r"^File:\s*[^,]+,\s*Node:\s*([^,]+),")
MENU_MARK_RE = re.compile(r"^\* Menu:[ \t]*\r?\n", re.M)
MD_HEADING_LINE_RE = re.compile(r"^(#{1,6})(\s)", re.M)


def iter_nodes(text: str):
    """Yield (node_name, raw_body) for every real node segment, in order.

    Drops the preamble segment (index 0) and any `Tag Table:` / `End Tag
    Table` / `Local Variables:` segment - none of those start with `File:`
    once leading newlines are stripped, which is how they are told apart
    from a real node segment.
    """
    segments = text.split("\x1f")
    for seg in segments[1:]:
        s = seg.lstrip("\n")
        if not s.startswith("File:"):
            continue
        nl = s.find("\n")
        header = s if nl == -1 else s[:nl]
        m = HEADER_LINE_RE.match(header)
        if not m:
            continue
        name = m.group(1).strip()
        body = "" if nl == -1 else s[nl + 1:]
        yield name, body


def strip_menu(body: str) -> str:
    """Drop the `* Menu:` table-of-contents block; keep real prose after it.

    Real menus (`* Name::  description`, with indented wrapped-description
    continuation lines, occasional group-label lines like "Important General
    Concepts", and in Top nodes a "-- The Detailed Node Listing --" divider
    with its own labelled sub-lists) always have another `* entry::` line
    somewhere further down, right up until the menu's last entry. Real prose
    that follows a menu (a footnote block, or an ordinary paragraph) never
    does - no `* ` line ever appears again in the node. That difference is
    what tells the two apart line by line.
    """
    m = MENU_MARK_RE.search(body)
    if not m:
        return body
    pre = body[: m.start()]
    lines = body[m.end():].splitlines(keepends=True)

    # suffix_has_star[i]: does any line at index >= i start with "* "?
    suffix_has_star = [False] * (len(lines) + 1)
    for i in range(len(lines) - 1, -1, -1):
        suffix_has_star[i] = suffix_has_star[i + 1] or lines[i].rstrip("\n").startswith("* ")

    cut = None
    prev_entry = False
    for i, line in enumerate(lines):
        content = line.rstrip("\n")
        if content.strip() == "":
            prev_entry = False
            continue
        if content.startswith("* "):
            prev_entry = True
            continue
        if prev_entry and (content.startswith(" ") or content.startswith("\t")):
            # wrapped continuation of the previous entry's description
            continue
        prev_entry = False
        if suffix_has_star[i + 1]:
            # divider / group-label / explanatory line - more entries follow
            continue
        cut = i
        break

    kept_after = "".join(lines[cut:]) if cut is not None else ""
    return pre + kept_after


def neutralise_headings(body: str) -> str:
    """A body line matching ^#{1,6}\\s would be misread as a markdown heading
    by ingest.py's `_MD_HEADING`; indent it by one space so it is not."""
    return MD_HEADING_LINE_RE.sub(r" \1\2", body)


def disambiguate(names: list[str]) -> list[str]:
    """Make slug(name) unique within a manual, deterministically.

    `slug()` collapses non-alphanumerics, so distinct node names can collide
    (info.info.gz's "Help" and "Help-]" both slug to "help"). The second (and
    any later) name to collide gets " (N)" appended until its slug is free.
    """
    used: set[str] = set()
    out = []
    for name in names:
        s = slug(name)
        if s not in used:
            used.add(s)
            out.append(name)
            continue
        k = 2
        while True:
            candidate = f"{name} ({k})"
            cs = slug(candidate)
            if cs not in used:
                used.add(cs)
                out.append(candidate)
                break
            k += 1
    return out


def convert_manual(gz_path: Path) -> tuple[str, int, int]:
    """Return (markdown_text, kept_node_count, skipped_index_count)."""
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    kept_names = []
    kept_bodies = []
    skipped = 0
    for name, raw_body in iter_nodes(text):
        if name.endswith("Index"):
            skipped += 1
            continue
        body = strip_menu(raw_body)
        body = neutralise_headings(body)
        kept_names.append(name)
        kept_bodies.append(body.strip())

    final_names = disambiguate(kept_names)
    parts = [f"## {name}\n\n{body}\n\n" for name, body in zip(final_names, kept_bodies)]
    return "".join(parts), len(final_names), skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dir", type=Path, help="directory containing *.info.gz files")
    ap.add_argument("--out", type=Path, required=True, help="output directory for *.md files")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    gz_files = sorted(args.dir.glob("*.info.gz"), key=lambda p: p.name)

    total_nodes = 0
    total_skipped = 0
    for gz_path in gz_files:
        manual = gz_path.name[: -len(".info.gz")]
        md_text, kept, skipped = convert_manual(gz_path)
        (args.out / f"{manual}.md").write_text(md_text, encoding="utf-8")
        total_nodes += kept
        total_skipped += skipped

    print(f"{len(gz_files)} files, {total_nodes} nodes, {total_skipped} index nodes skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
