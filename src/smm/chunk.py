"""Chunking, in two versions the eval can put side by side.

`chunk_doc` is phase 1: fixed-size, structure-blind, a window slid over the page
flattened back to plain text. The section tree was withheld from it on purpose, so
that phase 3 could show whether structure helps rather than assume it.

`chunk_doc_structured` is phase 3. It cuts where the document says to cut - one
tagged entry per unit, adjacent entries packed up to the same nominal size the flat
chunker uses. Holding the size fixed is the point: the only variable between the two
is *where the boundaries fall*, so a difference in recall cannot be a difference in
how much text an embedding had to cover.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

from .structure import doc_units, entries, heading_path

CHUNK_CHARS = 1000
OVERLAP_CHARS = 150

# An entry longer than this is windowed rather than kept whole. Set above the target
# so the common oversized option - a paragraph or two past the limit - still travels
# as one unit; only genuine outliers like sshd_config's 3 KB `Match` get cut.
MAX_ENTRY_CHARS = 1500


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    ord: int
    char_start: int
    char_end: int
    text: str
    prefix: str = ""
    sec_id: str = ""
    tag: str = ""

    @property
    def embed_text(self) -> str:
        return f"{self.prefix}{self.text}" if self.prefix else self.text


def flatten(doc: dict) -> str:
    """Page as plain text, headings inline, exactly as a reader would see it."""
    parts = []
    for sec in doc["sections"]:
        parts.append(sec["heading"])
        if sec["text"]:
            parts.append(sec["text"])
    return "\n".join(parts)


def doc_prefix(doc: dict) -> str:
    """`tar(1) - an archiving utility` — document identity, not hierarchy.

    A window from the middle of a long page otherwise never mentions which command
    it documents. Whether this helps is measured, not assumed: index with and
    without and compare.
    """
    summary = doc.get("summary", "")
    head = f"{doc['name']}({doc['section']})"
    return f"{head} - {summary}\n" if summary else f"{head}\n"


def split_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> Iterator[tuple[int, int]]:
    """Character windows, nudged to the nearest newline so lines stay intact."""
    n = len(text)
    if n == 0:
        return
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            nl = text.rfind("\n", start + size // 2, end)
            if nl != -1:
                end = nl + 1
        yield start, end
        if end >= n:
            break
        start = max(end - overlap, start + 1)


def chunk_doc(doc: dict, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS,
              with_prefix: bool = True) -> list[Chunk]:
    text = flatten(doc)
    prefix = doc_prefix(doc) if with_prefix else ""
    out = []
    for i, (a, b) in enumerate(split_text(text, size, overlap)):
        body = text[a:b].strip()
        if not body:
            continue
        out.append(Chunk(f"{doc['doc_id']}:{i}", doc["doc_id"], i, a, b, body, prefix))
    return out


def chunk_corpus(corpus: Path, **kw) -> Iterator[Chunk]:
    with corpus.open(encoding="utf-8") as fh:
        for line in fh:
            yield from chunk_doc(json.loads(line), **kw)


def to_dict(c: Chunk) -> dict:
    return asdict(c)


def section_prefix(doc: dict, sec: dict) -> str:
    """Document identity plus the heading trail: `tar(1) - ...` / `OPTIONS / ...`.

    The phase 1 ablation established that naming the document in every chunk is
    worth +5.6 points of recall@5, because a window from the middle of a page
    otherwise never says what it documents. The heading trail is the same argument
    one level down, and it is free: the extractor already parsed it. Whether it
    actually helps is a separate index and a separate eval, not an assumption.
    """
    return f"{doc_prefix(doc)}{heading_path(sec)}\n"


TAGGED_SECTION_RATIO = 0.5


def chunk_doc_structured(doc: dict, size: int = CHUNK_CHARS,
                         max_entry: int = MAX_ENTRY_CHARS,
                         with_prefix: bool = True, with_path: bool = False,
                         overlap: int = OVERLAP_CHARS,
                         tagged_ratio: float = TAGGED_SECTION_RATIO) -> list[Chunk]:
    """Entry-packed where the section is a list of tagged entries, windowed where not.

    The decision is per section, and that is the whole design:

    - An `OPTIONS`-shaped section - most of its text inside tagged paragraphs - is
      cut on its entries. An option is self-describing, so a chunk holding two whole
      options is about two things, and a chunk holding the tail of one and the head
      of the next is about nothing. Entries pack up to `size` and never split below
      `max_entry`, so a tag never separates from its description. This is the failure
      phase 3 exists to fix.
    - Any other section is windowed exactly as the flat chunker windows it.

    The second rule cost an index build to learn. Cutting on structure everywhere put
    `signal.7`'s table of signal numbers in a chunk of its own: correct as structure,
    useless as an embedding, because a bare column of `SIGHUP 1 1 1 1` means nothing
    without the paragraph above it naming the columns. A tagged option is
    self-describing; a table is not. Concept pages (man7) lost 5 of 16 questions to
    that distinction before it was drawn.

    `char_start`/`char_end` are offsets into the *section*, not the page: the flat
    chunker's offsets are into the flattened page, nothing compares the two, and a
    section-relative offset is what locates an entry again.
    """
    out: list[Chunk] = []
    base = doc_prefix(doc) if with_prefix else ""

    def emit(sec, body, tags, start, end):
        body = body.strip()
        if not body:
            return
        prefix = "" if not with_prefix else (section_prefix(doc, sec) if with_path else base)
        out.append(Chunk(f"{doc['doc_id']}:{len(out)}", doc["doc_id"], len(out),
                         start, end, body, prefix, sec["sec_id"],
                         " | ".join(t for t in tags if t)))

    for sec in doc["sections"]:
        text = sec["text"]
        if not text.strip():
            emit(sec, sec["heading"], [], 0, 0)
            continue
        units = entries(text)
        tagged = sum(len(b) for t, b in units if t)
        if not units or tagged < tagged_ratio * len(text):
            for a, b in split_text(text, size, overlap):
                emit(sec, text[a:b], [], a, b)
            continue

        buf, tags, start, pos = [], [], 0, 0
        for tag, block in units:
            if len(block) > max_entry:
                emit(sec, "\n\n".join(buf), tags, start, pos)
                buf, tags = [], []
                # Oversized entry: window it, repeating the tag so no fragment is
                # left anonymous.
                for a, b in split_text(block, size, overlap):
                    piece = block[a:b].strip()
                    if piece:
                        emit(sec, piece if not tag or piece.startswith(tag) else f"{tag}\n{piece}",
                             [tag], pos + a, pos + b)
                pos += len(block) + 2
                start = pos
                continue
            if buf and sum(len(x) + 2 for x in buf) + len(block) > size:
                emit(sec, "\n\n".join(buf), tags, start, pos)
                buf, tags, start = [], [], pos
            if not buf:
                start = pos
            buf.append(block)
            tags.append(tag)
            pos += len(block) + 2
        emit(sec, "\n\n".join(buf), tags, start, pos)
    return out


def chunk_corpus_structured(corpus: Path, **kw) -> Iterator[Chunk]:
    with corpus.open(encoding="utf-8") as fh:
        for line in fh:
            yield from chunk_doc_structured(json.loads(line), **kw)
