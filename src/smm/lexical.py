"""BM25 half of the hybrid, on SQLite's built-in FTS5. No extra dependency.

Phase 1 showed dense retrieval sends 59% of its misses to a *plausible wrong
document* - `grep` questions answered from `git-grep.1`, `find` questions from
`git-whatchanged.1`. Those queries name their tool, and a lexical index cannot
drift the way an embedding can.

Two tokenizer notes, both load-bearing:

- Dashes stay separators. Man pages spell the answer `--exclude-from`; the phase 0
  questions deliberately never do (`resolve_gold.py` rejects a question that leaks
  its own answer token). So a query says "exclude" and must match `--exclude-from`,
  which only happens while `-` splits. Making dashes token characters would index
  the flag as one atom and match nothing anybody types.
- Porter stemming, so "listing" finds "list" and "mounted" finds "mount".

Queries are natural-language sentences, not FTS5 expressions, so they are torn down
to bare terms and OR'd. bm25's IDF already discounts "how" and "the"; the stoplist
only removes terms so common they cost time without changing the ranking.
"""

from __future__ import annotations

import re
import sqlite3

TOKENIZER = "porter unicode61"

# Question scaffolding. Deliberately short: anything with discriminating power in a
# man-page corpus - "file", "run", "set" - is left in and handled by IDF.
STOP = frozenset("""a an the is are was were be been being do does did doing done
how what when where which who whom why can could should would will shall may might
must i my me you your it its this that these those there here of to in on at by for
with from into onto out up down over under again then than so such as and or but if
not no nor only own same too very s t just now want need way get got make made use
using used something anything thing please""".split())

TERM_RE = re.compile(r"[A-Za-z0-9_.%/=]+")


def terms(query: str) -> list[str]:
    """Query sentence -> bare searchable terms, order preserved, deduplicated."""
    out, seen = [], set()
    for raw in TERM_RE.findall(query.replace("-", " ")):
        t = raw.strip("._/=%").lower()
        if len(t) < 2 or t in STOP or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def match_expr(query: str) -> str | None:
    """FTS5 MATCH expression: every term OR'd, each quoted so punctuation is inert."""
    ts = terms(query)
    if not ts:
        return None
    return " OR ".join(f'"{t}"' for t in ts)


def has_index(db: sqlite3.Connection) -> bool:
    row = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
    ).fetchone()
    return row is not None


def create(db: sqlite3.Connection, tokenizer: str = TOKENIZER) -> None:
    """FTS5 over the existing `chunks` rows. External-content: no text is duplicated.

    The prefix is indexed alongside the body for the same reason it is embedded -
    a window from the middle of a page otherwise never names its own command, and
    naming the tool is most of what BM25 contributes here.
    """
    db.execute(
        f"""CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                prefix, text,
                content='chunks', content_rowid='rowid',
                tokenize="{tokenizer}")"""
    )
    db.commit()


def populate(db: sqlite3.Connection) -> int:
    """(Re)build the FTS index from `chunks`. Idempotent."""
    db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")
    db.execute(
        "INSERT INTO chunks_fts(rowid, prefix, text) SELECT rowid, prefix, text FROM chunks"
    )
    db.commit()
    return db.execute("SELECT count(*) FROM chunks_fts").fetchone()[0]


def search(db: sqlite3.Connection, query: str, k: int = 50) -> list[dict]:
    """Top-k by BM25. `score` is negated bm25(), so larger is better, as with dense.

    The prefix column is weighted below the body: matching the tool name should
    promote a page, not let a 240 KB option dump rank on its own header.
    """
    expr = match_expr(query)
    if not expr:
        return []
    rows = db.execute(
        """SELECT c.chunk_id, c.doc_id, c.text, c.prefix, bm25(chunks_fts, 0.5, 1.0)
             FROM chunks_fts f JOIN chunks c ON c.rowid = f.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY bm25(chunks_fts, 0.5, 1.0)
            LIMIT ?""",
        (expr, k),
    ).fetchall()
    return [
        {"chunk_id": r[0], "doc_id": r[1], "text": r[2], "prefix": r[3],
         "bm25": r[4], "score": -r[4]}
        for r in rows
    ]
