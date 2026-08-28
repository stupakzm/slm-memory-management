"""Single-file vector + text store on sqlite-vec. No server, no daemon."""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import sqlite_vec


def connect(path: Path, dim: int | None = None) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    if dim is not None:
        create(db, dim)
    return db


def create(db: sqlite3.Connection, dim: int) -> None:
    db.execute(
        """CREATE TABLE IF NOT EXISTS chunks(
               rowid      INTEGER PRIMARY KEY,
               chunk_id   TEXT UNIQUE NOT NULL,
               doc_id     TEXT NOT NULL,
               ord        INTEGER NOT NULL,
               char_start INTEGER NOT NULL,
               char_end   INTEGER NOT NULL,
               prefix     TEXT NOT NULL DEFAULT '',
               sec_id     TEXT NOT NULL DEFAULT '',
               tag        TEXT NOT NULL DEFAULT '',
               domain     TEXT NOT NULL DEFAULT 'linux',
               text       TEXT NOT NULL)"""
    )
    db.execute("CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id)")
    # Migration, not just creation: an index built before phase 5 has a `chunks`
    # table that CREATE TABLE IF NOT EXISTS will not touch, so the column has to be
    # added explicitly before anything indexes it. Only writers reach here - readers
    # call connect() without a dim - so opening an old index to query it never
    # rewrites it.
    if "domain" not in {r[1] for r in db.execute("PRAGMA table_info(chunks)")}:
        db.execute("ALTER TABLE chunks ADD COLUMN domain TEXT NOT NULL DEFAULT 'linux'")
    db.execute("CREATE INDEX IF NOT EXISTS chunks_domain ON chunks(domain)")
    db.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding float[{dim}])")
    db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
    db.commit()


def set_meta(db: sqlite3.Connection, **kw) -> None:
    db.executemany("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
                   [(k, str(v)) for k, v in kw.items()])
    db.commit()


def get_meta(db: sqlite3.Connection) -> dict:
    return dict(db.execute("SELECT key,value FROM meta").fetchall())


def pack(vec) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def add(db: sqlite3.Connection, rows: list[tuple[dict, list[float]]]) -> None:
    """rows: [(chunk_dict, embedding), ...]"""
    cur = db.cursor()
    for c, emb in rows:
        cur.execute(
            "INSERT OR IGNORE INTO chunks"
            "(chunk_id,doc_id,ord,char_start,char_end,prefix,sec_id,tag,domain,text)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (c["chunk_id"], c["doc_id"], c["ord"], c["char_start"], c["char_end"],
             c.get("prefix", ""), c.get("sec_id", ""), c.get("tag", ""),
             c.get("domain", "linux"), c["text"]),
        )
        rid = cur.execute("SELECT rowid FROM chunks WHERE chunk_id=?", (c["chunk_id"],)).fetchone()[0]
        # vec0 does not honour INSERT OR REPLACE on an existing rowid - it raises a
        # UNIQUE violation - so the old vector goes first. This is what makes an
        # interrupted ingest safe to re-run.
        cur.execute("DELETE FROM vec_chunks WHERE rowid=?", (rid,))
        cur.execute("INSERT INTO vec_chunks(rowid,embedding) VALUES(?,?)", (rid, pack(emb)))
    db.commit()


def has_structure(db: sqlite3.Connection) -> bool:
    """Phase 1 and 2 indexes predate the sec_id/tag columns; both still open here.

    Asked on every search rather than cached: sqlite3.Connection takes no
    attributes, and a PRAGMA against an open schema costs nothing next to the
    vector scan it precedes.
    """
    return "sec_id" in {r[1] for r in db.execute("PRAGMA table_info(chunks)")}


def has_domain(db: sqlite3.Connection) -> bool:
    return "domain" in {r[1] for r in db.execute("PRAGMA table_info(chunks)")}


def domains(db: sqlite3.Connection) -> dict[str, int]:
    if not has_domain(db):
        return {"linux": count(db)}
    return dict(db.execute("SELECT domain, count(*) FROM chunks GROUP BY domain"))


def search(db: sqlite3.Connection, query_vec: list[float], k: int = 5,
           domain: str | None = None, max_overfetch: int = 4096) -> list[dict]:
    """Top-k by cosine distance, optionally restricted to one domain.

    sqlite-vec's `k` is a global top-k over the whole table, so a namespace is served
    by asking for more than k and discarding the rest. The multiplier grows until k
    in-domain rows come back or the table is exhausted - a fixed multiple silently
    under-delivers for a domain that is a small share of the index, which is exactly
    the situation user-fed material starts in.
    """
    struct = "c.sec_id, c.tag" if has_structure(db) else "'' , ''"
    dom = "c.domain" if has_domain(db) else "'linux'"
    want = k if domain is None else min(max(k * 8, 64), max_overfetch)
    while True:
        rows = db.execute(
            f"""SELECT c.chunk_id, c.doc_id, c.text, c.prefix, {struct}, c.ord,
                       {dom}, v.distance
                  FROM vec_chunks v JOIN chunks c ON c.rowid = v.rowid
                 WHERE v.embedding MATCH ? AND k = ?
                 ORDER BY v.distance""",
            (pack(query_vec), want),
        ).fetchall()
        if domain is not None:
            rows = [r for r in rows if r[7] == domain]
        if domain is None or len(rows) >= k or want >= max_overfetch:
            break
        want = min(want * 4, max_overfetch)
    return [
        {"chunk_id": r[0], "doc_id": r[1], "text": r[2], "prefix": r[3],
         "sec_id": r[4], "tag": r[5], "ord": r[6], "domain": r[7],
         "distance": r[8], "score": 1.0 - r[8] / 2.0}
        for r in rows[:k]
    ]


def count(db: sqlite3.Connection) -> int:
    return db.execute("SELECT count(*) FROM chunks").fetchone()[0]


def neighbours(db: sqlite3.Connection, doc_id: str, sec_id: str,
               ord_: int, span: int = 1) -> list[dict]:
    """Chunks adjacent to one chunk *within its own section*, in reading order.

    Structure-aware chunks are smaller than the flat windows they replace - median
    633 characters against 962 - which is the point for retrieval and a loss for
    generation, since the model sees a third less text for the same k. Expansion
    gives the context back without giving the precision away: the ranking still
    happens on the tight chunk, and only what the model reads grows.

    The section boundary is the stop. Two adjacent chunks in different sections are
    adjacent on the page and unrelated in meaning, which is the whole premise of
    cutting on structure in the first place.
    """
    rows = db.execute(
        """SELECT chunk_id, doc_id, text, prefix, sec_id, tag, ord
             FROM chunks
            WHERE doc_id = ? AND sec_id = ? AND ord BETWEEN ? AND ?
            ORDER BY ord""",
        (doc_id, sec_id, ord_ - span, ord_ + span),
    ).fetchall()
    return [{"chunk_id": r[0], "doc_id": r[1], "text": r[2], "prefix": r[3],
             "sec_id": r[4], "tag": r[5], "ord": r[6]} for r in rows]
