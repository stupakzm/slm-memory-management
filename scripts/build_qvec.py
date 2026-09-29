#!/usr/bin/env python3
"""Phase 11 R8: build the question vectors (docs/phase11-results.md, "R8 pre-registration").

For every chunk of one domain the 4B writes a few questions a user might ask when
they want what the chunk says but do not know its names; each question is embedded
on its own (document side) and stored as a vector pointing at its chunk. The vectors
go into a COPY of the source index - the source is only ever opened read-only.

  .venv/bin/python scripts/build_qvec.py --limit 20 --out /tmp/qx-smoke.db   # smoke run
  .venv/bin/python scripts/build_qvec.py                                     # emacs, 3 per chunk

Both stages are resumable: generation skips chunks already in --cache, embedding
skips questions already in --out. Generation needs the generator server; embedding
needs the embedder server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import store  # noqa: E402

# v3's term-card prompt (phase 11 exploratory), adapted from "one manual entry" to
# "one manual passage". {n} is the --per-chunk count spelled out.
PROMPT_TEMPLATE = (
    "You help people find things in the GNU Emacs manual. Given one passage from the "
    "manual, write {n} short questions a user might ask when they want what this "
    "passage describes, but do not know its names. Use everyday words, the way a "
    "person would describe the task or the problem. Do not use any command, function, "
    "variable or key name from the passage. One question per line, no numbering, "
    "nothing else."
)
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
SAVE_EVERY = 250
EMBED_BATCH = 64


def system_prompt(per_chunk: int) -> str:
    return PROMPT_TEMPLATE.format(n=NUMBER_WORDS.get(per_chunk, str(per_chunk)))


def parse_questions(text: str, n: int) -> list[str]:
    lines = (l.strip().lstrip(" -*0123456789.)\t").strip() for l in text.splitlines())
    return [l for l in lines if l][:n]


def todo_chunks(chunks: list[tuple[str, str]], cache: dict) -> list[tuple[str, str]]:
    """Chunks (chunk_id, text) that have no cache entry yet, in their given order."""
    return [c for c in chunks if c[0] not in cache]


def load_chunks(src: Path, domain: str, limit: int | None = None) -> list[tuple[str, str]]:
    db = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        rows = db.execute("SELECT chunk_id, text FROM chunks WHERE domain=? ORDER BY rowid",
                          (domain,)).fetchall()
    finally:
        db.close()
    return rows[:limit] if limit else rows


def save_cache(path: Path, cache: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cache, indent=1))
    tmp.replace(path)


def generate(chunks, cache: dict, cache_path: Path, gen, per_chunk: int,
             parallel: int = 4) -> int:
    """Ask `gen` for questions for every chunk not yet cached; returns how many."""
    todo = todo_chunks(chunks, cache)
    system = system_prompt(per_chunk)

    def one(chunk):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": chunk[1]}]
        return chunk[0], parse_questions(gen.chat(msgs, max_tokens=120, temperature=0.0),
                                         per_chunk)

    t0 = time.time()
    try:
        with ThreadPoolExecutor(parallel) as pool:
            for i in range(0, len(todo), SAVE_EVERY):
                for cid, qs in pool.map(one, todo[i:i + SAVE_EVERY]):
                    cache[cid] = qs
                save_cache(cache_path, cache)
                done = min(i + SAVE_EVERY, len(todo))
                print(f"  generated {done}/{len(todo)}  {(time.time() - t0) / done:.2f} s/chunk",
                      flush=True)
    finally:
        save_cache(cache_path, cache)
    return len(todo)


def prepare_out(src: Path, out: Path) -> None:
    """Copy src to out with the backup API (src read-only) unless out exists."""
    if src.resolve() == out.resolve():
        raise SystemExit("--out must not be --src: the source index is never written")
    if out.exists():
        return
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    d = sqlite3.connect(out)
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()


def embed(out: Path, cache: dict, emb, domain: str, per_chunk: int) -> int:
    """Embed cached questions not yet in `out`'s chunk_questions; returns how many."""
    db = store.connect(out)
    try:
        have = set()
        if store.has_qvec(db):
            have = set(db.execute("SELECT chunk_id, question FROM chunk_questions"))
        todo = [(cid, q) for cid, qs in cache.items() for q in qs if (cid, q) not in have]
        for i in range(0, len(todo), EMBED_BATCH):
            batch = todo[i:i + EMBED_BATCH]
            vecs = emb.embed_documents([q for _, q in batch])
            store.create_qvec(db, len(vecs[0]))
            store.add_questions(db, [(c, q, v) for (c, q), v in zip(batch, vecs)])
            print(f"  embedded {min(i + EMBED_BATCH, len(todo))}/{len(todo)}", flush=True)
        store.set_meta(db, qvec_domain=domain, qvec_per_chunk=per_chunk,
                       qvec_prompt_sha256=hashlib.sha256(
                           system_prompt(per_chunk).encode()).hexdigest())
    finally:
        db.close()
    return len(todo)


def build(src: Path, out: Path, domain: str, per_chunk: int, cache_path: Path,
          gen=None, emb=None, stage: str = "both", parallel: int = 4,
          limit: int | None = None) -> None:
    if src.resolve() == out.resolve():
        raise SystemExit("--out must not be --src: the source index is never written")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if stage in ("generate", "both"):
        generate(load_chunks(src, domain, limit), cache, cache_path, gen, per_chunk, parallel)
    if stage in ("embed", "both"):
        prepare_out(src, out)
        embed(out, cache, emb, domain, per_chunk)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/index/phase11.db")
    ap.add_argument("--out", default="data/index/phase11-qx.db")
    ap.add_argument("--domain", default="emacs")
    ap.add_argument("--per-chunk", type=int, default=3)
    ap.add_argument("--cache", default=None,
                    help="default: data/index/qvec-<domain>.json")
    ap.add_argument("--gen-url", default="http://127.0.0.1:8083")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N chunks of the domain (smoke runs)")
    ap.add_argument("--stage", choices=("generate", "embed", "both"), default="both")
    args = ap.parse_args()
    cache = ROOT / (args.cache or f"data/index/qvec-{args.domain}.json")

    gen = emb = None
    if args.stage in ("generate", "both"):
        from smm.generate import Generator
        gen = Generator(args.gen_url)
        if not gen.health():
            print(f"no generator at {args.gen_url}", file=sys.stderr)
            return 2
    if args.stage in ("embed", "both"):
        from smm.embed import Embedder
        emb = Embedder()
        if not emb.health():
            print("embedder not running: ./scripts/servers.sh start embedder", file=sys.stderr)
            return 2
    build(ROOT / args.src, ROOT / args.out, args.domain, args.per_chunk, cache,
          gen=gen, emb=emb, stage=args.stage, parallel=args.parallel,
          limit=args.limit or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
