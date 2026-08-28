"""Corpus identity, so that "which corpus was this number measured on" has an answer.

Every number in this repository is relative to the documentation installed on one
machine at one moment, and phase 4 established that the moment moves. Re-extracting
picked up 33 `cmake-*` pages that had been installed since August, and separately a
one-line fix to the extractor changed 1,084 documents. Neither showed up as an
error. An index silently built from a different corpus than the eval it is scored
against is the kind of defect that invalidates a comparison without ever failing.

So a corpus gets a digest, and the digest travels: into the index metadata at build
time and into every eval result. `scripts/corpus_fingerprint.py --check` then answers
"has the machine moved under me", and the diff says which documents.

The digest covers three things that can each change a number:

- the *content*, as a sorted list of per-document digests, so a changed page and an
  added page are both visible and distinguishable;
- the *extractor*, hashed from its source, because parsing is where the pip bug
  lived and a corpus file alone cannot show that its producer changed;
- the *sections* requested, since a corpus of man1 is not a corpus of man1,5,7,8.

It deliberately does not cover chunking or embedding. Those belong to an index, and
two indexes over one corpus should share a corpus identity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = 1


def _sha(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def doc_digest(doc: dict) -> str:
    """Identity of one document: its id and the exact text a chunker would see."""
    body = "\n".join(f"{s['sec_id']}\n{s['heading']}\n{s['text']}"
                     for s in doc["sections"])
    return _sha(doc["doc_id"], doc.get("summary", ""), body)


def extractor_digest(source: Path) -> str:
    return hashlib.sha256(source.read_bytes()).hexdigest()[:16]


def compute(corpus: Path, extractor: Path | None = None) -> dict:
    docs, chars, sections = {}, 0, 0
    with corpus.open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            docs[d["doc_id"]] = doc_digest(d)
            sections += len(d["sections"])
            chars += sum(len(s["text"]) for s in d["sections"])
    digest = _sha(*(f"{k}:{v}" for k, v in sorted(docs.items())))
    return {
        "schema": SCHEMA,
        "digest": digest[:16],
        "n_docs": len(docs),
        "n_sections": sections,
        "n_chars": chars,
        "extractor": extractor_digest(extractor) if extractor else None,
        "docs": docs,
    }


def summary(fp: dict) -> str:
    return (f"{fp['digest']}  {fp['n_docs']} docs, {fp['n_sections']} sections, "
            f"{fp['n_chars']/1e6:.2f} MB, extractor {fp['extractor']}")


def diff(old: dict, new: dict) -> dict:
    a, b = old["docs"], new["docs"]
    return {
        "added": sorted(set(b) - set(a)),
        "removed": sorted(set(a) - set(b)),
        "changed": sorted(k for k in set(a) & set(b) if a[k] != b[k]),
        "extractor_changed": old.get("extractor") != new.get("extractor"),
    }


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def save(fp: dict, path: Path) -> None:
    path.write_text(json.dumps(fp, indent=2, sort_keys=True))


def stale_warning(meta: dict, corpus: Path, extractor: Path | None = None) -> str | None:
    """A line to print when an index was built from a different corpus than is here.

    Returns None when they agree. Indexes built before fingerprinting existed carry
    no digest at all, and say so rather than pretending to match - an unknown
    provenance is a weaker claim than a matching one, not an equal one.
    """
    recorded = meta.get("corpus_digest")
    if not recorded:
        return ("index predates corpus fingerprinting; its corpus cannot be verified "
                "against the one on disk")
    cur = compute(corpus, extractor)["digest"]
    if recorded == cur:
        return None
    return (f"INDEX/CORPUS MISMATCH: index built from corpus {recorded}, "
            f"corpus on disk is {cur}. Numbers from this run are not comparable with "
            f"runs on the other corpus. See scripts/corpus_fingerprint.py --check.")
