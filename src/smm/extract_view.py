"""The extract view the reader sees (--read-view repaired in scripts/eval_answers.py).

Windows are cut by character count and nudged only at their END to a newline
(chunk.split_text), so a window usually starts mid-line, inside an option's
description and without the option's name. This module repairs only what the
READER reads: the line fragment the window opens inside is completed from the
previous window of the same page, the option / section heading line above it
is put on top, and consecutive windows of one page are merged. Retrieval,
evidence scoring and the gate keep reading the original hits.

Pure functions over hit dicts (chunk_id, doc_id, ord, prefix, text, ...); stdlib only.
"""

from __future__ import annotations

KEY_LEN = 40          # leading characters of a window searched for in the previous window
MIN_KEY = 20          # shortest window text that can be located
HEADER_MAX = 160
FRAG_MAX = 200
MIN_OVERLAP = 20


def _header_above(text: str) -> str:
    """The last non-empty line of `text` that does not start with a space or tab
    (an option header or a section heading), or ''."""
    for line in reversed(text.split("\n")):
        if line.strip() and line[0] not in " \t":
            return line
    return ""


def repair_extract(hit: dict, prev_text: str | None) -> dict:
    """A NEW hit whose text opens on a complete line under its option/section
    header, or an equal copy when no repair applies. `prev_text` is the stripped
    text of the same doc's window ord-1 (None -> unchanged)."""
    out = dict(hit)
    t = hit["text"]
    if prev_text is None or len(t) < MIN_KEY:
        return out
    pos = prev_text.rfind(t[:KEY_LEN])
    if pos < 0:
        return out
    line_start = prev_text.rfind("\n", 0, pos) + 1
    frag = prev_text[line_start:pos]
    if frag == "":
        return out
    if len(frag) > FRAG_MAX:
        frag = frag[-FRAG_MAX:]
    header = _header_above(prev_text[:line_start])[:HEADER_MAX]
    out["text"] = (header + "\n" if header else "") + frag + t
    return out


def _remainder(first: str, second: str) -> str | None:
    """`second` without the longest prefix (>= MIN_OVERLAP characters) that is
    also a suffix of `first`. A repaired `second` opens with a header line that
    `first` does not end with, so the match is retried after its first line."""
    starts = [0]
    nl = second.find("\n")
    if 0 <= nl <= HEADER_MAX:
        starts.append(nl + 1)
    for j in starts:
        rest = second[j:]
        for k in range(min(len(first), len(rest)), MIN_OVERLAP - 1, -1):
            if first.endswith(rest[:k]):
                return rest[k:]
    return None


def _join(first: str, second: str) -> str:
    rem = _remainder(first, second)
    return first + "\n" + second if rem is None else first + rem


def merge_consecutive(hits: list[dict]) -> list[dict]:
    """Merge hits that are consecutive windows (same doc_id, ord differing by 1)
    of one page. The merged extract takes the position and keys of the
    earlier-ranked member; the overlap between the windows is not repeated."""
    out: list[dict] = []
    span: list[tuple] = []  # (doc_id, lowest ord, highest ord) per output item
    for h in hits:
        h = dict(h)
        doc, o = h.get("doc_id"), int(h["ord"])
        for i, (d, lo, hi) in enumerate(span):
            if d != doc:
                continue
            if o == hi + 1:
                out[i]["text"] = _join(out[i]["text"], h["text"])
                span[i] = (d, lo, o)
                break
            if o == lo - 1:
                out[i]["text"] = _join(h["text"], out[i]["text"])
                span[i] = (d, o, hi)
                break
        else:
            out.append(h)
            span.append((doc, o, o))
    return out


def repair_hits(hits: list[dict], prev_lookup) -> list[dict]:
    """The reader's view of `hits`. prev_lookup(doc_id, ord) -> stripped window
    text or None."""
    return merge_consecutive([
        repair_extract(h, prev_lookup(h["doc_id"], int(h["ord"]) - 1)
                       if int(h["ord"]) > 0 else None)
        for h in hits])
