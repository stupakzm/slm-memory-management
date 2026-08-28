"""Turn arbitrary user material into documents the existing pipeline already handles.

The cheapest way to reuse phases 1-4 is to make user material look like a man page:
a document with an id, a summary, and an ordered list of sections. Then `chunk.py`,
`store.py`, the reranker, the gate and the citation grammar all work unchanged, and
phase 5 is an ingest path plus a namespace rather than a second system.

What it cannot reuse is the reason man pages were easy. `corpus/manpages.py` rests on
a groff invariant - body text is always indented - that holds for 4,114 pages and
gives sections for free. A pasted note has no such guarantee, so the splitter here is
explicitly a guess, in three tiers:

1. Markdown headings, if the text has any. Authored structure, same as a man heading.
2. Otherwise a blank-line-delimited paragraph walk, packed into sections of roughly
   `SECTION_CHARS`. Not structure, just a stable unit.
3. Failing both, one section for the whole document.

Tier 2 is the honest weak point and it is worth naming: the project's whole argument
against corpus-agnostic RAG is that man pages come with hierarchy already authored,
so a domain that arrives without one gets a worse deal here, and phase 3 measured
what unreliable boundaries cost. User material is retrieved on the same footing as
man pages but it is not cut as well, and no amount of pipeline reuse changes that.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path

SECTION_CHARS = 2000
DEFAULT_DOMAIN = "linux"

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$", re.M)
_SLUG = re.compile(r"[^a-z0-9]+")


def slug(text: str, limit: int = 48) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return _SLUG.sub("-", text.lower()).strip("-")[:limit] or "untitled"


def _sections_from_markdown(text: str) -> list[tuple[int, str, str]]:
    """(level, heading, body) for a document that carries its own headings."""
    marks = list(_MD_HEADING.finditer(text))
    if not marks:
        return []
    out = []
    preamble = text[: marks[0].start()].strip()
    if preamble:
        out.append((1, "Introduction", preamble))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end():end].strip()
        out.append((len(m.group(1)), m.group(2).strip(), body))
    return out


def _sections_from_paragraphs(text: str, size: int) -> list[tuple[int, str, str]]:
    """No authored structure: pack paragraphs into stable, roughly equal sections.

    The heading is the section's first line, truncated. It is a label, not a claim
    about the document's structure, and it exists so a chunk prefix has something to
    say about where it came from.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, buf, n = [], [], 0
    for p in paras:
        if buf and n + len(p) > size:
            body = "\n\n".join(buf)
            out.append((1, body.splitlines()[0][:60], body))
            buf, n = [], 0
        buf.append(p)
        n += len(p)
    if buf:
        body = "\n\n".join(buf)
        out.append((1, body.splitlines()[0][:60], body))
    return out


def from_text(text: str, title: str, domain: str, doc_id: str | None = None,
              summary: str = "", source: str = "",
              section_chars: int = SECTION_CHARS) -> dict:
    """A user document in the shape `chunk.py` and `store.py` already accept."""
    text = text.replace("\r\n", "\n").strip()
    doc_id = doc_id or f"{slug(title)}.{domain}"
    parts = _sections_from_markdown(text) or _sections_from_paragraphs(text, section_chars)
    if not parts:
        parts = [(1, title, text)]
    sections = []
    for i, (level, heading, body) in enumerate(parts):
        sections.append({
            "sec_id": f"{doc_id}#{slug(heading) or i}",
            "heading": heading, "level": level, "parent": None, "text": body,
        })
    return {
        "doc_id": doc_id,
        # `name`/`section` exist because doc_prefix() builds `tar(1) - summary` from
        # them; a user document reads as `recipes(cooking) - ...` for the same reason.
        "name": slug(title), "section": domain,
        "title": title, "summary": summary or title,
        "aliases": [], "see_also": [], "sections": sections,
        "domain": domain, "source": source,
        "digest": hashlib.sha256(text.encode()).hexdigest()[:16],
    }


def dedupe_ids(docs: list[dict]) -> list[dict]:
    """Make doc_ids unique within a batch, in place.

    `slug()` truncates, so two distinct sources can land on one id - four of 1,602
    GNOME help pages did, and the collision surfaced only as a constraint violation
    deep in the vector store. Disambiguation is a short digest of the source rather
    than a counter, so the id a document gets does not depend on the order the batch
    happened to be read in.
    """
    seen: dict[str, int] = {}
    for d in docs:
        if d["doc_id"] not in seen:
            seen[d["doc_id"]] = 1
            continue
        tag = hashlib.sha256((d.get("source") or d["title"]).encode()).hexdigest()[:4]
        base, _, suffix = d["doc_id"].rpartition(".")
        d["doc_id"] = f"{base}-{tag}.{suffix}"
        for sec in d["sections"]:
            sec["sec_id"] = f"{d['doc_id']}#{sec['sec_id'].split('#', 1)[-1]}"
    return docs


def from_file(path: Path, domain: str, title: str | None = None, **kw) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    return from_text(raw, title or path.stem, domain, source=str(path), **kw)


def from_url(url: str, domain: str, title: str | None = None, timeout: int = 60,
             **kw) -> dict:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "slm-memory-management/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", "replace")
    return from_text(raw, title or url.rsplit("/", 1)[-1], domain, source=url, **kw)
