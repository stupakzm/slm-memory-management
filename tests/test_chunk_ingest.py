"""Tests for src/smm/chunk.py and src/smm/ingest.py (tsk_20260928_chunktests).

Covers: split_text's window/overlap/newline-snap geometry, chunk_doc's id/prefix/
embed_text shape, chunk_doc_structured's per-section decision (entry-packed vs
windowed, oversized-entry re-tagging, the empty-section heading fallback), and
ingest.py's slug(), from_text()'s markdown/paragraph-packing tiers, and
dedupe_ids()'s order-independent suffixing and sec_id rewrite.

Hermetic by design (blk_test_env_constraints): src/smm/structure.py is
stdlib-only, so chunk imports cleanly with no stub. ingest.from_url is not
called anywhere in this file - the only network-capable path in ingest.py -
so no urllib stub is needed either.

Fixtures are built by hand from src/smm/structure.py's `entries()` contract:
a column-0 line followed by an indented line is a tagged entry (the whole
tag+indented run is its block); a run of column-0 lines with no indented
continuation is prose. See structure.py's docstring and the entries() tests
in tests/test_defects.py / tests/test_depth.py for the same convention used
here.
"""

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smm import chunk
from smm import ingest
from smm import structure


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


# --------------------------------------------------------------------------
# chunk.split_text
# --------------------------------------------------------------------------

def test_split_text_empty():
    windows = list(chunk.split_text(""))
    check(windows == [], f"empty text must yield no windows: {windows}")


def test_split_text_windows_cover_text_with_overlap():
    # No newlines, so the newline-snap never fires and the arithmetic is exact.
    text = "a" * 205
    windows = list(chunk.split_text(text, size=50, overlap=10))
    check(windows == [(0, 50), (40, 90), (80, 130), (120, 170), (160, 205)],
          f"exact window sequence: {windows}")
    check(windows[0][0] == 0, "first window must start at 0")
    check(windows[-1][1] == len(text), "last window must end at len(text)")
    for (pa, pb), (na, nb) in zip(windows, windows[1:]):
        check(na < pb, f"consecutive windows must overlap: {(pa, pb)} -> {(na, nb)}")
        check(pb - na <= 10, f"overlap must not exceed `overlap`: {(pa, pb)} -> {(na, nb)}")


def test_split_text_snaps_to_newline():
    # Newline at index 15; window of size 20 starting at 0 has its back half
    # at [10, 20), so the newline at 15 falls inside it and the window end
    # must snap to just after it (16), not stay at the raw 20.
    text = "a" * 15 + "\n" + "b" * 30
    windows = list(chunk.split_text(text, size=20, overlap=5))
    check(windows[0] == (0, 16), f"first window must snap to just after the newline: {windows[0]}")
    check(text[windows[0][1] - 1] == "\n", "window end must land right after the newline")


# --------------------------------------------------------------------------
# chunk.chunk_doc
# --------------------------------------------------------------------------

def test_chunk_doc_prefix_and_ids():
    doc = {
        "doc_id": "tar.1",
        "name": "tar",
        "section": "1",
        "summary": "an archiving utility",
        "sections": [
            {"heading": "NAME", "text": "tar - an archiving utility", "sec_id": "tar.1#NAME"},
        ],
    }
    chunks = chunk.chunk_doc(doc)
    check(len(chunks) == 1, f"short doc must fit in one chunk: {len(chunks)}")
    c = chunks[0]
    check(c.chunk_id == "tar.1:0", f"chunk_id must be '<doc_id>:<i>': {c.chunk_id}")
    check(c.ord == 0, f"ord must be the window index: {c.ord}")
    check(c.prefix == "tar(1) - an archiving utility\n",
          f"prefix must be 'name(section) - summary\\n': {c.prefix!r}")
    check(c.embed_text == c.prefix + c.text, f"embed_text must be prefix+text: {c.embed_text!r}")

    chunks_noprefix = chunk.chunk_doc(doc, with_prefix=False)
    c2 = chunks_noprefix[0]
    check(c2.prefix == "", f"with_prefix=False must leave prefix empty: {c2.prefix!r}")
    check(c2.embed_text == c2.text, f"embed_text with no prefix must equal text: {c2.embed_text!r}")


# --------------------------------------------------------------------------
# chunk.chunk_doc_structured
# --------------------------------------------------------------------------

def test_structured_tagged_entries_not_split():
    text = "--foo\n    Do foo things.\n\n--bar\n    Do bar things and stuff.\n"
    doc = {"doc_id": "x.1", "name": "x", "section": "1",
           "sections": [{"heading": "OPTIONS", "text": text, "sec_id": "x.1#OPTIONS"}]}
    chunks = chunk.chunk_doc_structured(doc, size=30, max_entry=1000, overlap=5, with_prefix=False)
    check(len(chunks) == 2, f"must cut on the entry boundary, not mid-entry: {len(chunks)}")
    check(chunks[0].tag == "--foo" and chunks[1].tag == "--bar",
          f"each chunk's tag must name its own entry: {[c.tag for c in chunks]}")
    check(chunks[0].text == "--foo\n    Do foo things.", f"chunk 0 body: {chunks[0].text!r}")
    check(chunks[1].text == "--bar\n    Do bar things and stuff.", f"chunk 1 body: {chunks[1].text!r}")
    check("--bar" not in chunks[0].text, "chunk 0 must not contain the next entry")
    check("--foo" not in chunks[1].text, "chunk 1 must not contain the previous entry")


def test_structured_untagged_section_windowed():
    prose = "This is plain prose text with no tagged entries at all, just words. " * 5
    doc = {"doc_id": "p.1", "name": "p", "section": "1",
           "sections": [{"heading": "DESCRIPTION", "text": prose, "sec_id": "p.1#DESCRIPTION"}]}
    chunks = chunk.chunk_doc_structured(doc, size=60, overlap=10, with_prefix=False)
    expected = list(chunk.split_text(prose, size=60, overlap=10))
    check(len(chunks) == len(expected), f"one chunk per window: {len(chunks)} vs {len(expected)}")
    for c, (a, b) in zip(chunks, expected):
        check(c.char_start == a and c.char_end == b,
              f"structured offsets must match split_text's: {(c.char_start, c.char_end)} vs {(a, b)}")
        check(c.text == prose[a:b].strip(), f"structured body must match the same slice: {c.text!r}")
        check(c.tag == "", f"an untagged window must have no tag: {c.tag!r}")


def test_structured_oversized_entry_repeats_tag():
    tag = "--big"
    body_lines = "\n".join(
        f"    line {i} of a long description that keeps going and going." for i in range(30))
    text = f"{tag}\n{body_lines}\n"
    doc = {"doc_id": "b.1", "name": "b", "section": "1",
           "sections": [{"heading": "OPTIONS", "text": text, "sec_id": "b.1#OPTIONS"}]}
    units = structure.entries(text)
    check(len(units) == 1 and units[0][0] == tag, f"fixture must be a single tagged entry: {units}")
    block = units[0][1]
    check(len(block) > 200, "fixture must actually exceed max_entry")
    chunks = chunk.chunk_doc_structured(doc, size=300, overlap=50, max_entry=200, with_prefix=False)
    expected_windows = list(chunk.split_text(block, size=300, overlap=50))
    check(len(chunks) == len(expected_windows),
          f"an oversized entry must be windowed into this many fragments: {len(chunks)} vs {len(expected_windows)}")
    for c in chunks:
        check(c.tag == tag, f"every fragment must keep the entry's tag: {c.tag!r}")
        check(c.text.startswith(tag), f"every fragment must start with its tag: {c.text[:20]!r}")


def test_structured_empty_section_emits_heading():
    doc = {"doc_id": "e.1", "name": "e", "section": "1",
           "sections": [{"heading": "BUGS", "text": "   ", "sec_id": "e.1#BUGS"}]}
    chunks = chunk.chunk_doc_structured(doc, with_prefix=False)
    check(len(chunks) == 1, f"an empty section must emit exactly one chunk: {len(chunks)}")
    c = chunks[0]
    check(c.text == "BUGS", f"its body must be the heading: {c.text!r}")
    check(c.char_start == 0 and c.char_end == 0, f"offsets must be (0, 0): {(c.char_start, c.char_end)}")
    check(c.sec_id == "e.1#BUGS", f"sec_id must carry through: {c.sec_id!r}")


# --------------------------------------------------------------------------
# ingest.slug
# --------------------------------------------------------------------------

def test_slug():
    check(ingest.slug("Hello, World!") == "hello-world",
          f"lowercase, non-alnum runs collapse to one '-': {ingest.slug('Hello, World!')!r}")
    check(ingest.slug("café") == "cafe", f"NFKD must strip the accent: {ingest.slug('café')!r}")
    check(ingest.slug("a" * 100, limit=10) == "a" * 10,
          f"must truncate to `limit`: {ingest.slug('a' * 100, limit=10)!r}")
    check(ingest.slug("") == "untitled", f"empty text must fall back to 'untitled': {ingest.slug('')!r}")
    check(ingest.slug("!!!") == "untitled",
          f"all-punctuation text must also fall back: {ingest.slug('!!!')!r}")


# --------------------------------------------------------------------------
# ingest.from_text
# --------------------------------------------------------------------------

def test_from_text_markdown_sections():
    md_text = ("Some preamble text here.\n\n"
               "# Title One\nBody one paragraph.\n\n"
               "## Sub Two\nBody two paragraph.\n")
    doc = ingest.from_text(md_text, title="Doc Title", domain="test")
    check(doc["doc_id"] == "doc-title.test", f"doc_id shape: {doc['doc_id']!r}")
    secs = doc["sections"]
    check(len(secs) == 3, f"preamble + 2 headings = 3 sections: {len(secs)}")
    check(secs[0]["heading"] == "Introduction" and secs[0]["text"] == "Some preamble text here.",
          f"preamble becomes 'Introduction': {secs[0]!r}")
    check(secs[0]["sec_id"] == "doc-title.test#introduction", f"sec_id shape: {secs[0]['sec_id']!r}")
    check(secs[1]["heading"] == "Title One" and secs[1]["level"] == 1 and
          secs[1]["text"] == "Body one paragraph.",
          f"first heading section: {secs[1]!r}")
    check(secs[1]["sec_id"] == "doc-title.test#title-one", f"sec_id shape: {secs[1]['sec_id']!r}")
    check(secs[2]["heading"] == "Sub Two" and secs[2]["level"] == 2 and
          secs[2]["text"] == "Body two paragraph.",
          f"second, deeper heading section: {secs[2]!r}")

    # CRLF must be normalised away before anything else happens.
    doc_crlf = ingest.from_text(md_text.replace("\n", "\r\n"), title="Doc Title", domain="test")
    check(doc_crlf["sections"] == doc["sections"], "CRLF input must produce identical sections")
    check(all("\r" not in s["text"] and "\r" not in s["heading"] for s in doc_crlf["sections"]),
          "no section may retain a bare \\r")


def test_from_text_paragraph_packing():
    p1, p2, p3, p4 = ("A" * 30, "B" * 30, "C" * 30, "D" * 30)
    text = "\n\n".join([p1, p2, p3, p4])
    doc = ingest.from_text(text, title="Pack Doc", domain="notes", section_chars=70)
    secs = doc["sections"]
    # size 70: p1+p2 (60) fits, +p3 would be 90 so it flushes; same for p3+p4.
    check(len(secs) == 2, f"two paragraphs must pack per section at this size: {len(secs)}")
    check(secs[0]["text"] == p1 + "\n\n" + p2, f"section 0 body: {secs[0]['text']!r}")
    check(secs[1]["text"] == p3 + "\n\n" + p4, f"section 1 body: {secs[1]['text']!r}")
    check(secs[0]["heading"] == p1, f"heading is the first line: {secs[0]['heading']!r}")

    # Heading truncation to 60, on a single-line paragraph longer than that.
    text2 = ("X" * 100) + "\n\n" + "tail paragraph"
    doc2 = ingest.from_text(text2, title="Long Doc", domain="notes", section_chars=1000)
    check(len(doc2["sections"]) == 1, "both paragraphs must pack into one section at this size")
    heading = doc2["sections"][0]["heading"]
    check(heading == "X" * 60 and len(heading) == 60,
          f"heading must be the first line truncated to 60: {heading!r}")


# --------------------------------------------------------------------------
# ingest.dedupe_ids
# --------------------------------------------------------------------------

def test_dedupe_ids_order_independent_and_sec_ids_rewritten():
    a = ingest.from_text("Body A", title="Same Title", domain="linux", source="src-A")
    b = ingest.from_text("Body B", title="Same Title", domain="linux", source="src-B")
    c = ingest.from_text("Body C", title="Same Title", domain="linux", source="src-C")
    check(a["doc_id"] == b["doc_id"] == c["doc_id"] == "same-title.linux",
          "fixture must actually collide before dedupe")

    run1 = ingest.dedupe_ids(copy.deepcopy([a, b, c]))
    run2 = ingest.dedupe_ids(copy.deepcopy([c, a, b]))

    def by_source(docs, source):
        return next(d for d in docs if d["source"] == source)

    tag_b = hashlib.sha256(b"src-B").hexdigest()[:4]
    b1 = by_source(run1, "src-B")
    b2 = by_source(run2, "src-B")
    check(b1["doc_id"] == b2["doc_id"] == f"same-title-{tag_b}.linux",
          f"a colliding doc's suffix must depend only on its own source, not batch order: "
          f"{b1['doc_id']!r} vs {b2['doc_id']!r}")

    # First-seen in each order keeps the bare id; the rest get renamed.
    check(by_source(run1, "src-A")["doc_id"] == "same-title.linux",
          "first-seen in run1 (A) must be untouched")
    check(by_source(run2, "src-C")["doc_id"] == "same-title.linux",
          "first-seen in run2 (C) must be untouched")

    # Every sec_id must be rewritten to carry the document's new doc_id.
    for run in (run1, run2):
        for d in run:
            for sec in d["sections"]:
                check(sec["sec_id"].startswith(d["doc_id"] + "#"),
                      f"sec_id must be rewritten to the new doc_id: {sec['sec_id']!r} for {d['doc_id']!r}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  pass  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001 - a crashing test is still a failure to report
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
