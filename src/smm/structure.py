"""Split a man-page section into the entries a human sees in it.

Phase 2 closed the half of phase 1's failure split it was aimed at - wrong document
- and left the other half untouched: 16 of the 36 remaining misses retrieve the
right page and rank the answer chunk 6th to 20th, or miss it entirely. The mechanism
is not subtle. A 1,000-char window slid blindly through `sshd_config.5`'s 65 KB
DESCRIPTION cuts between `PermitRootLogin` and its description as readily as around
it, and a chunk holding the tail of one directive and the head of the next is about
nothing in particular.

The fix does not need a model, because groff already marked the boundaries. A
tagged paragraph (`.TP`) - which is how every man page renders an option, a config
directive, a mount flag - comes out as a lone tag line at column 0 followed by an
indented block:

    --exclude-from=FILE
           Read a list of patterns from FILE.

Ordinary prose is also at column 0, but as a *run* of lines. So the entire parser is
one question asked of each column-0 line: is the next non-blank line indented? If
yes it is a tag and the indented block is its body; if no it is prose. That rests on
the same groff invariant the section parser already rests on - body text is always
indented - which `corpus/manpages.py` documents and which held across all 4,114
pages with zero failures.

Nested entries stay with their parent. `--backup[=CONTROL]` documents its own values
(`none, off`, `t, numbered`) as a deeper tagged list; those belong to the option
being described, not to four separate options.
"""

from __future__ import annotations

from typing import Iterator


def _indented(line: str) -> bool:
    return line[:1] == " "


def entries(text: str) -> list[tuple[str | None, str]]:
    """Section body -> [(tag, block)], in document order, covering the whole text.

    `tag` is the entry's tag line for a tagged paragraph, and None for prose. The
    block always includes the tag line, so blocks concatenate back to the section.
    """
    lines = text.split("\n")
    out: list[tuple[str | None, str]] = []
    i, n = 0, len(lines)

    def emit(tag: str | None, a: int, b: int) -> None:
        block = "\n".join(lines[a:b]).rstrip()
        if block.strip():
            out.append((tag, block))

    while i < n:
        if not lines[i].strip():
            i += 1
            continue
        if _indented(lines[i]):
            # An indented block with no tag above it: a continuation of something
            # already emitted, or a section that opens indented. Keep it whole.
            j = i
            while j < n and (not lines[j].strip() or _indented(lines[j])):
                j += 1
            emit(None, i, j)
            i = j
            continue

        k = i + 1
        while k < n and not lines[k].strip():
            k += 1
        if k < n and _indented(lines[k]):
            # Tagged paragraph: this line tags the indented block that follows.
            j = k
            while j < n and (not lines[j].strip() or _indented(lines[j])):
                j += 1
            emit(lines[i].strip(), i, j)
            i = j
        else:
            # Prose: consume the run of column-0 lines up to the next tagged entry.
            j = i
            while j < n:
                if not lines[j].strip():
                    j += 1
                    continue
                if _indented(lines[j]):
                    break
                nxt = j + 1
                while nxt < n and not lines[nxt].strip():
                    nxt += 1
                if nxt < n and _indented(lines[nxt]):
                    break          # lines[j] tags the next entry; stop before it
                j = nxt
            emit(None, i, j)
            i = j
    return out


def heading_path(sec: dict) -> str:
    """`OPTIONS / Local file selection` - the trail a reader would have followed.

    Read straight off `sec_id`, which the corpus extractor already builds as
    `doc_id#HEADING/Subheading`. The `parent` field holds the parent's *heading*
    rather than its id, so walking it would mean a second lookup for no gain.
    """
    return sec["sec_id"].split("#", 1)[-1].replace("/", " / ")


def doc_units(doc: dict) -> Iterator[tuple[dict, str | None, str]]:
    """(section, tag, block) for every entry in a document, in reading order."""
    for sec in doc["sections"]:
        if not sec["text"].strip():
            yield sec, None, sec["heading"]
            continue
        for tag, block in entries(sec["text"]):
            yield sec, tag, block
