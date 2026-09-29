"""Integration test for phase 11 R8's store functions and build step, against the
REAL sqlite_vec (tsk_20260929_qvec). Run from the worktree root:

  /home/stupakzm/projects/slm-memory-management/.venv/bin/python tests/integration_qvec_store.py

Temp dbs only, dim 4, hand-made vectors; no server, no data/.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import store  # noqa: E402

_spec = importlib.util.spec_from_file_location("build_qvec", ROOT / "scripts" / "build_qvec.py")
build_qvec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_qvec)

DIM = 4


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def unit(i: int, eps: float = 0.0) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    v[(i + 1) % DIM] = eps
    return v


def chunk(cid: str, domain: str, ord_: int = 0) -> dict:
    return {"chunk_id": cid, "doc_id": cid.split(":")[0], "ord": ord_, "char_start": 0,
            "char_end": 5, "domain": domain, "text": f"text of {cid}"}


def make_src(path: Path) -> None:
    db = store.connect(path, DIM)
    store.add(db, [(chunk("e1:0", "emacs"), unit(0)), (chunk("e2:0", "emacs"), unit(1)),
                   (chunk("l1:0", "linux"), unit(2))])
    db.close()


def test_search_questions_maps_to_chunks_deduped():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.db"
        make_src(p)
        db = store.connect(p)
        check(not store.has_qvec(db), "fresh index has no qvec")
        store.create_qvec(db, DIM)
        store.create_qvec(db, DIM)  # idempotent
        check(store.has_qvec(db), "has_qvec after create")
        store.add_questions(db, [
            ("e1:0", "close question", unit(3, 0.1)),
            ("e2:0", "second chunk question", unit(3, 0.5)),
            ("e1:0", "far question of first chunk", unit(3, 0.9)),
        ])
        hits = store.search_questions(db, unit(3), k=3)
        check([h["chunk_id"] for h in hits] == ["e1:0", "e2:0"], hits)
        check(hits[0]["matched_question"] == "close question", hits[0])
        check(hits[0]["distance"] < hits[1]["distance"] and 0 < hits[0]["score"] <= 1, hits)
        need = {"chunk_id", "doc_id", "text", "prefix", "sec_id", "tag", "ord", "domain",
                "distance", "score"}
        check(need <= set(hits[0]), f"same keys as search(): {sorted(hits[0])}")
        check(len(store.search_questions(db, unit(3), k=1)) == 1, "k counts question hits")
        db.close()


def test_search_questions_honours_domain():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.db"
        make_src(p)
        db = store.connect(p)
        store.create_qvec(db, DIM)
        # the linux question is nearest, the emacs ones farther
        store.add_questions(db, [("l1:0", "linux q", unit(3, 0.0)),
                                 ("e1:0", "emacs q", unit(3, 0.4)),
                                 ("e2:0", "emacs q2", unit(3, 0.8))])
        check([h["chunk_id"] for h in store.search_questions(db, unit(3), k=2)]
              == ["l1:0", "e1:0"], "no domain: nearest first")
        got = store.search_questions(db, unit(3), k=2, domain="emacs")
        check([h["chunk_id"] for h in got] == ["e1:0", "e2:0"], got)
        check(all(h["domain"] == "emacs" for h in got), got)
        check(store.search_questions(db, unit(3), k=2, domain="none") == [], "empty domain")
        db.close()


def test_build_leaves_source_db_untouched():
    class _Gen:
        def chat(self, messages, max_tokens=400, temperature=0.0, grammar=None):
            return "a question?\nanother question?\nthird question?"

    class _Emb:
        def embed_documents(self, texts):
            return [unit(3, 0.1 * (i + 1)) for i, _ in enumerate(texts)]

    def counts(p):
        db = store.connect(p)
        try:
            return {t: db.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                    for t in ("chunks", "vec_chunks")}
        finally:
            db.close()

    with tempfile.TemporaryDirectory() as td:
        src, out, cache = Path(td) / "src.db", Path(td) / "out.db", Path(td) / "c.json"
        make_src(src)
        before = hashlib.sha256(src.read_bytes()).hexdigest()
        src_counts = counts(src)
        before2 = hashlib.sha256(src.read_bytes()).hexdigest()  # counts() must not write either
        build_qvec.build(src, out, "emacs", 3, cache, gen=_Gen(), emb=_Emb(), parallel=2)
        check(hashlib.sha256(src.read_bytes()).hexdigest() == before == before2,
              "the source db file changed")
        db = store.connect(out)
        n = db.execute("SELECT count(*) FROM chunk_questions").fetchone()[0]
        check(n == 6 and store.has_qvec(db), f"2 emacs chunks x 3 questions, got {n}")
        check(store.get_meta(db)["qvec_domain"] == "emacs", store.get_meta(db))
        db.close()
        check(counts(out) == src_counts, (counts(out), src_counts))
        # a second run is a no-op on the embed side
        build_qvec.build(src, out, "emacs", 3, cache, gen=_Gen(), emb=_Emb())
        db = store.connect(out)
        check(db.execute("SELECT count(*) FROM chunk_questions").fetchone()[0] == 6, "re-embedded")
        db.close()
        try:
            build_qvec.build(src, src, "emacs", 3, cache, gen=_Gen(), emb=_Emb())
        except SystemExit:
            pass
        else:
            raise AssertionError("--out == --src must be refused")


def test_fold_replaces_only_domain_vectors():
    class _Emb:
        def embed_documents(self, texts):
            return [[float(len(t) % 7), 1.0, 0.0, 0.5] for t in texts]

    def dump(p):
        db = store.connect(p)
        try:
            return (db.execute("SELECT * FROM chunks ORDER BY rowid").fetchall(),
                    dict(db.execute("SELECT rowid, embedding FROM vec_chunks").fetchall()),
                    store.has_qvec(db),
                    db.execute("SELECT count(*) FROM sqlite_master WHERE name='chunk_questions'")
                    .fetchone()[0], store.get_meta(db))
        finally:
            db.close()

    with tempfile.TemporaryDirectory() as td:
        src, out = Path(td) / "src.db", Path(td) / "out.db"
        make_src(src)
        before = hashlib.sha256(src.read_bytes()).hexdigest()
        build_qvec.prepare_out(src, out)
        n = build_qvec.fold(out, {"e1:0": ["q one?", "q two?"], "l1:0": ["ignored?"]},
                            _Emb(), "emacs")
        check(n == 1, f"only e1:0 is folded, got {n}")
        check(hashlib.sha256(src.read_bytes()).hexdigest() == before, "source changed")
        s_chunks, s_vecs, *_ = dump(src)
        o_chunks, o_vecs, has_q, has_cq, meta = dump(out)
        check(o_chunks == s_chunks, "chunks table must be identical")
        want = store.pack(_Emb().embed_documents(
            [build_qvec.fold_text("", "text of e1:0", ["q one?", "q two?"])])[0])
        check(o_vecs[1] == want, "folded vector is the embedding of fold_text")
        check(o_vecs[2] == s_vecs[2] and o_vecs[3] == s_vecs[3],
              "unfolded and other-domain vectors byte-identical")
        check(o_vecs[1] != s_vecs[1], "folded vector changed")
        check(not has_q and has_cq == 0, "no qvec / chunk_questions table")
        check(meta["qvec_fold"] == "1" and meta["qvec_domain"] == "emacs", meta)
        check(hashlib.sha256(build_qvec.system_prompt(3).encode()).hexdigest()
              == meta["qvec_prompt_sha256"], "prompt hash")


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
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
