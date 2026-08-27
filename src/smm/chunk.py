"""Phase 1 chunking: deliberately boring, fixed-size, structure-blind.

The extracted section tree is *not* used here on purpose. If the baseline already
had hierarchy in it, phase 3 could never show whether hierarchy helps. This
flattens each page back to plain text and slides a fixed window over it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator

CHUNK_CHARS = 1000
OVERLAP_CHARS = 150


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    ord: int
    char_start: int
    char_end: int
    text: str
    prefix: str = ""

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
