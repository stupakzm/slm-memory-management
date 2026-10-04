"""Tests for `--normalize local` (phase 15 R12, docs/phase15-results.md "R12
pre-registration"): out-of-vocabulary question words are repaired only toward
words found in the extracts the reader is about to read, and only the reader
sees the repaired question.

Hermetic by design (blk_test_env_constraints): stdlib only, no .venv, no data/,
no models/, no servers. `sqlite_vec` is stubbed in sys.modules before
scripts/eval_answers.py and scripts/ask.py are imported (as in
tests/test_eval_answers.py); those modules' Generator/Embedder/Reranker/
Retriever/store are replaced by stubs, and their ROOT is pointed at a temp dir
holding a tiny sqlite `chunks` table, an eval file and a retrieval cache.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sqlite3
import sys
import tempfile
import types
from pathlib import Path

if "sqlite_vec" not in sys.modules:
    stub = types.ModuleType("sqlite_vec")
    stub.load = lambda *a, **kw: None  # never called: no real index is opened here
    sys.modules["sqlite_vec"] = stub

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import normalize  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


eval_answers = _load("eval_answers", "scripts/eval_answers.py")
ask = _load("ask", "scripts/ask.py")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def _make_db(path: Path, texts: list) -> None:
    """A real sqlite index reduced to what build_vocab reads: `chunks`."""
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE chunks (prefix TEXT, text TEXT)")
    for t in texts:
        conn.execute("INSERT INTO chunks VALUES (?, ?)", ("", t))
    conn.commit()
    conn.close()


# The corpus vocabulary used below: every ordinary question word is in it
# (count >= 2), the typo `windw` and the tool names `elpy` / `nmap` are not.
CORPUS = ["how to resize the window manager", "resize the window manager now",
          "install the package manager", "install the package now"]


def _hit(chunk_id: str, doc_id: str, prefix: str, text: str, score: float = 0.9) -> dict:
    return {"chunk_id": chunk_id, "doc_id": doc_id, "prefix": prefix, "text": text,
            "score": score, "rerank_score": score}


# --------------------------------------------------------------------------
# normalize_local
# --------------------------------------------------------------------------

def test_local_corrects_typo_to_extract_word():
    vocab = {"resize": 2, "window": 2, "manager": 2}
    extracts = ["wm.1\nSection: wm\nA window can be resized by the manager."]
    out, edits = normalize.normalize_local("how to resize the windw", extracts, vocab)
    check(out == "how to resize the window", f"typo not repaired: {out!r}")
    check(edits == [{"from": "windw", "to": "window", "distance": 1}], edits)
    # the module-level entry point is the same call
    out2, edits2 = normalize.normalize_query(
        "how to resize the windw", "local", vocab, extracts)
    check((out2, edits2) == (out, edits), (out2, edits2))
    # a word that is only in the extract's doc id and prefix still counts: that
    # is part of what the reader is shown
    out3, _ = normalize.normalize_local("a windw", ["window.1\npfx "], vocab)
    check(out3 == "a window", out3)
    # the extract string is exactly what build_prompt shows, minus its label
    h = _hit("c", "ls.1", "[ls] ", "list files")
    check(normalize.hit_extract(h) == "ls.1\n[ls] list files", normalize.hit_extract(h))


def test_local_ignores_words_absent_from_extracts():
    # corpus vocabulary holds `mmap`; plain spell correction would rewrite the
    # uninstalled tool name `nmap` into it (the R4a failure). Local repair only
    # looks at the extracts, which hold nothing near it.
    vocab = {"mmap": 5, "install": 3}
    q = "how do I install nmap"
    check(normalize.normalize(q, vocab)[0] == "how do I install mmap",
          "precondition: spell mode renames nmap -> mmap")
    out, edits = normalize.normalize_local(q, ["ls.1\nlist directory contents"], vocab)
    check(out == q and edits == [], f"nmap must be left alone: {out!r} {edits}")
    # the corpus vocabulary does not feed the candidates, even for a close word
    out, edits = normalize.normalize_local("a windw", ["ls.1\nlist files"], {"window": 9})
    check(out == "a windw" and edits == [], (out, edits))
    # a question word that is itself in the extracts is evidence, not a typo
    # (it is out of the corpus vocabulary here, count < 2, but the reader sees it)
    out, edits = normalize.normalize_local(
        "what is elpy", ["elpa.1\nelpy and elpa"], {"what": 3})
    check(out == "what is elpy" and edits == [], (out, edits))
    # no extracts at all: nothing to repair toward
    check(normalize.normalize_local("a windw", [], {"x": 1}) == ("a windw", []),
          "empty extracts must leave the question alone")
    check(normalize.normalize_local("", ["x"], {}) == ("", []), "empty question")


def test_local_leaves_vocab_words_alone():
    vocab = {"window": 3, "manager": 3, "tar": 9}
    extracts = ["wm.1\nwindows managers tab tars"]
    # in-vocabulary words are never touched, even with a 1-edit neighbour
    # (windows, managers) in the extracts
    q = "window manager"
    check(normalize.normalize_local(q, extracts, vocab) == (q, []), "vocab words changed")
    # 3-letter words are never eligible (R4a's rule), in or out of vocabulary
    q = "tab tsr"
    check(normalize.normalize_local(q, extracts, vocab) == (q, []), "short words changed")
    # case-insensitive vocabulary membership
    q = "Window MANAGER"
    check(normalize.normalize_local(q, extracts, vocab) == (q, []), "cased vocab words changed")


def test_local_distance_bands_and_ties():
    d = normalize.osa_distance
    # 4-7 letters: <= 1 edit. `windw` -> window is 1, `wndw` is 2, so unrepaired.
    ex = ["x.1\nwindow changes"]
    check(d("wndw", "window") == 2 and d("chngess", "changes") == 2, "test setup")
    check(normalize.normalize_local("wndw", ex, {})[0] == "wndw", "4-letter, 2 edits")
    check(normalize.normalize_local("chngess", ex, {})[0] == "chngess", "7-letter, 2 edits")
    check(normalize.normalize_local("windw", ex, {})[0] == "window", "5-letter, 1 edit")
    # a transposition is one edit (optimal string alignment)
    check(d("wnidow", "window") == 1, "test setup")
    check(normalize.normalize_local("wnidow", ex, {})[0] == "window", "transposition")
    # 8+ letters: <= 2 edits
    ex8 = ["x.1\ndirectories"]
    check(d("dirctoris", "directories") == 2, "test setup")
    out, edits = normalize.normalize_local("dirctoris", ex8, {})
    check(out == "directories" and edits[0]["distance"] == 2, (out, edits))

    # tie-break 1: smaller distance beats a more frequent farther word
    ex = ["x.1\nsorting " + "sortings " * 9]
    check(d("sortinng", "sorting") == 1 and d("sortinng", "sortings") == 2, "test setup")
    check(normalize.normalize_local("sortinng", ex, {})[0] == "sorting", "distance first")

    # tie-break 2: a keyboard-adjacent substitution beats a more frequent
    # non-adjacent one at the same distance (bind -> bins: d/s adjacent)
    check(normalize._is_kbd_substitution("bind", "bins")
          and not normalize._is_kbd_substitution("bind", "bing"), "test setup")
    ex = ["x.1\nbing bing bing bins"]
    check(normalize.normalize_local("bind", ex, {})[0] == "bins", "keyboard-adjacent first")

    # tie-break 3: higher count IN THE EXTRACTS (not in the corpus vocabulary)
    ex = ["x.1\nbink bing bing"]
    out, edits = normalize.normalize_local("bind", ex, {"bink": 99})
    check(out == "bing", f"extract frequency decides: {out!r}")
    check(edits == [{"from": "bind", "to": "bing", "distance": 1}], edits)

    # tie-break 4: the word itself, deterministically
    ex = ["x.1\nbink bing"]
    check(normalize.normalize_local("bind", ex, {})[0] == "bing", "alphabetical last")


def test_local_preserves_case_and_untouchables():
    vocab: dict = {}
    ex = ["x.1\nwindow"]
    cases = {
        "windw": "window", "Windw": "Window", "WINDW": "WINDOW",
        "(windw)?": "(window)?", "see windw, now": "see window, now",
        "two  spaces windw": "two  spaces window",     # whitespace is byte-exact
    }
    for q, want in cases.items():
        got = normalize.normalize_local(q, ex, vocab)[0]
        check(got == want, f"{q!r} -> {got!r}, want {want!r}")
    # flags, paths, KEY=VALUE, versions, anything with a digit: exact
    for q in ("--windw", "windw=1", "/etc/windw", "windw2", "v1.2.3", "windw.",
              "~/windw", "win_dw", "-windw", "windw-"):
        check(normalize.normalize_local(q, ex, vocab) == (q, []), f"touched {q!r}")
    # the edits list is in question order
    out, edits = normalize.normalize_local("windw and Windw", ex, vocab)
    check([e["from"] for e in edits] == ["windw", "Windw"], edits)
    check([e["to"] for e in edits] == ["window", "Window"], edits)


# --------------------------------------------------------------------------
# scripts/eval_answers.py --normalize local
# --------------------------------------------------------------------------

class _StubGen:
    """Stands in for smm.generate.Generator: records what the reader is asked."""
    calls: list = []

    def __init__(self, *a, **kw):
        pass

    def health(self):
        return True

    def answer(self, question, hits, cite_grammar=False, mode="cite"):
        _StubGen.calls.append((question, [h["doc_id"] for h in hits]))
        return "A window is resized by the manager [1]."


class _StubEmb:
    def health(self):
        return True


class _StubRetriever:
    asked: list = []

    def __init__(self, *a, **kw):
        pass

    def retrieve(self, question, k=5):
        _StubRetriever.asked.append(question)
        return [_hit("c1", "wm.1", "Section: wm\n", "A window is resized.", 0.9)]


class _StubStore:
    @staticmethod
    def connect(path):
        return object()


def _eval_fixture(tmp: Path) -> None:
    (tmp / "data" / "index").mkdir(parents=True)
    (tmp / "data" / "eval" / "results").mkdir(parents=True)
    _make_db(tmp / "data" / "index" / "t.db", CORPUS)
    rows = [
        {"qid": "q1", "kind": "answerable", "tags": [], "answer_contains": ["window"],
         "question": "how to resize the windw manager"},
        {"qid": "q2", "kind": "answerable", "tags": [], "answer_contains": ["window"],
         "question": "how to resize the window manager"},
        {"qid": "q3", "kind": "unanswerable", "tags": ["tool-not-installed"],
         "answer_contains": [], "question": "how to resize the windw manager"},
    ]
    (tmp / "data" / "eval" / "questions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    good = [_hit("c1", "wm.1", "Section: wm\n", "A window is resized by the manager.", 0.9)]
    weak = [_hit("c9", "zz.1", "Section: zz\n", "A window is somewhere here.", 0.2)]
    (tmp / "data" / "eval" / "results" / "t-retrieved.json").write_text(
        json.dumps({"q1": good, "q2": good, "q3": weak}))


def _run_eval(tmp: Path, argv: list) -> int:
    saved = (eval_answers.ROOT, eval_answers.Generator, eval_answers.Embedder,
             eval_answers.Reranker, eval_answers.Retriever, eval_answers.store, sys.argv)
    eval_answers.ROOT = tmp
    eval_answers.Generator = _StubGen
    eval_answers.Embedder = eval_answers.Reranker = _StubEmb
    eval_answers.Retriever, eval_answers.store = _StubRetriever, _StubStore
    sys.argv = ["eval_answers.py", "--db", "data/index/t.db", "--name", "t",
                "--no-aliases"] + argv
    try:
        with _quiet(), contextlib.redirect_stderr(io.StringIO()):
            return eval_answers.main()
    finally:
        (eval_answers.ROOT, eval_answers.Generator, eval_answers.Embedder,
         eval_answers.Reranker, eval_answers.Retriever, eval_answers.store,
         sys.argv) = saved


def test_eval_answers_local_repairs_reader_only():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _eval_fixture(tmp)
        cache_path = tmp / "data" / "eval" / "results" / "t-retrieved.json"
        cache_before = cache_path.read_bytes()
        out_path = tmp / "data" / "eval" / "results" / "t-answers.json"

        # --- generate stage
        _StubGen.calls = []
        rc = _run_eval(tmp, ["--stage", "generate", "--gate", "0.65", "--grammar",
                             "--normalize", "local"])
        check(rc == 0, f"rc {rc}")
        rep = json.loads(out_path.read_text())
        by = {r["qid"]: r for r in rep["results"]}
        # the reader got the repaired question for the row the gate lets through ...
        check(_StubGen.calls[0] == ("how to resize the window manager", ["wm.1"]),
              _StubGen.calls)
        # ... unchanged text for a row with nothing to repair ...
        check(_StubGen.calls[1][0] == "how to resize the window manager", _StubGen.calls)
        # ... and a gated row (q3, top score 0.2) never reaches the reader at all
        check(len(_StubGen.calls) == 2, f"gated row called the generator: {_StubGen.calls}")
        check(by["q3"]["gated"] is True, by["q3"])
        # records carry the raw question, the question the reader read, and the edits
        r1 = by["q1"]
        check(r1["question"] == "how to resize the windw manager", r1["question"])
        check(r1["question_normalized"] == "how to resize the window manager", r1)
        check(r1["normalize_edits"] == [{"from": "windw", "to": "window", "distance": 1}],
              r1["normalize_edits"])
        check(by["q2"]["question_normalized"] == by["q2"]["question"]
              and by["q2"]["normalize_edits"] == [], by["q2"])
        check(by["q3"]["question_normalized"] == by["q3"]["question"]
              and by["q3"]["normalize_edits"] == [], by["q3"])
        check(rep["config"]["normalize"] == "local", rep["config"])
        check(cache_path.read_bytes() == cache_before, "generate stage rewrote the cache")

        # --- --normalize off: the reader keeps the typo, records gain nothing
        _StubGen.calls = []
        rc = _run_eval(tmp, ["--stage", "generate", "--gate", "0.65"])
        check(rc == 0, f"rc {rc}")
        off = json.loads(out_path.read_text())
        check(_StubGen.calls[0][0] == "how to resize the windw manager", _StubGen.calls)
        check("normalize" not in off["config"], off["config"])
        check(all("question_normalized" not in r for r in off["results"]), "off added fields")

        # --- the retrieve stage is identical to --normalize off and sees the raw question
        res = tmp / "data" / "eval" / "results"
        outs = {}
        for mode in ("off", "local"):
            _StubRetriever.asked = []
            rc = _run_eval(tmp, ["--stage", "retrieve", "--name", f"r-{mode}",
                                 "--normalize", mode, "--rerank"])
            check(rc == 0, f"retrieve {mode}: rc {rc}")
            outs[mode] = (res / f"r-{mode}-retrieved.json").read_bytes()
            check(_StubRetriever.asked[0] == "how to resize the windw manager",
                  f"{mode}: retrieval did not see the raw question: {_StubRetriever.asked}")
        check(outs["off"] == outs["local"], "retrieve caches differ between off and local")

        # --- incompatible flags exit 2 before anything starts
        _StubGen.calls = []
        for flag in ("--cascade", "--llm-correct"):
            rc = _run_eval(tmp, ["--stage", "generate", "--normalize", "local", flag])
            check(rc == 2, f"{flag}: rc {rc}")
        check(_StubGen.calls == [], "generator used despite a rejected combination")


# --------------------------------------------------------------------------
# normalize.py CLI: --mode local pre-pass
# --------------------------------------------------------------------------

def _run_cli(argv: list):
    saved = sys.argv
    sys.argv = ["normalize.py"] + argv
    try:
        with _quiet(), contextlib.redirect_stderr(io.StringIO()):
            try:
                return normalize.main()
            except SystemExit as e:
                return e.code
    finally:
        sys.argv = saved


def test_prepass_local_counts_and_qids():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _make_db(tmp / "t.db", CORPUS)
        pool = [
            {"qid": "a1", "question": "how to resize the window manager"},          # clean
            {"qid": "a2", "question": "how to resize the elpy manager"},            # clean, name
            {"qid": "a1.y1", "question": "how to resize the windw manager",
             "variant_of": "a1", "variant_kind": "typo1",
             "edits": [{"word_index": 4, "before": "window", "after": "windw"}]},
            {"qid": "a1.y3", "question": "how to resize the windw manager",
             "variant_of": "a1", "variant_kind": "typo3",
             "edits": [{"word_index": 4, "before": "window", "after": "windw"}]},
            {"qid": "a1.yg", "question": "how to resize the windw manager",
             "variant_of": "a1", "variant_kind": "typo1",
             "edits": [{"word_index": 4, "before": "window", "after": "windw"}]},
        ]
        (tmp / "pool.jsonl").write_text("\n".join(json.dumps(r) for r in pool) + "\n")
        good = [_hit("c1", "wm.1", "Section: wm\n", "A window is resized by the manager.", 0.9)]
        weak = [_hit("c9", "zz.1", "Section: zz\n", "A window is here.", 0.2)]
        retrieved = {
            "a1": good, "a2": good, "a1.y1": good,
            # dict entry shape: the gate reads gate_hits, the reader reads hits
            "a1.y3": {"hits": good, "gate_hits": [dict(good[0], rerank_score=0.7)]},
            "a1.yg": {"hits": good, "gate_hits": weak},
        }
        (tmp / "ret.json").write_text(json.dumps(retrieved))

        base = ["--db", str(tmp / "t.db"), "--pool", str(tmp / "pool.jsonl"),
                "--vocab-cache", str(tmp / "v.json")]
        rc = _run_cli(base + ["--out", str(tmp / "local.json"), "--mode", "local",
                              "--retrieved", str(tmp / "ret.json"), "--gate", "0.65",
                              "--qids-out", str(tmp / "qids.json")])
        check(rc == 0, f"rc {rc}")
        rep = json.loads((tmp / "local.json").read_text())
        check(rep["n_questions"] == 5, rep["n_questions"])
        check(rep["kinds"] == {"clean": {"n": 2, "changed": 0, "gated": 0},
                               "typo1": {"n": 2, "changed": 1, "gated": 1},
                               "typo3": {"n": 1, "changed": 1, "gated": 0}}, rep["kinds"])
        check(rep["clean_changed"] == [], rep["clean_changed"])
        # changed qids, in pool order (a1.y1 before a1.y3), gated and clean rows absent
        check(json.loads((tmp / "qids.json").read_text()) == ["a1.y1", "a1.y3"],
              (tmp / "qids.json").read_text())
        by = {r["qid"]: r for r in rep["results"]}
        check(by["a1.yg"]["gated"] is True and by["a1.yg"]["changed"] is False
              and by["a1.yg"]["question_normalized"] == by["a1.yg"]["question"], by["a1.yg"])
        check(by["a1.y1"]["question_normalized"] == "how to resize the window manager",
              by["a1.y1"])
        check(by["a2"]["question_normalized"] == by["a2"]["question"], by["a2"])
        check({"mean", "p95"} <= set(rep["timing_ms"]), rep["timing_ms"])
        check(rep["typo_edit_recovery"]["typo1"]["reverted"] == 1, rep["typo_edit_recovery"])

        # --mode local needs --retrieved; --retrieved needs --mode local (argparse: exit 2)
        check(_run_cli(base + ["--out", str(tmp / "x.json"), "--mode", "local"]) == 2,
              "local without --retrieved must exit 2")
        check(_run_cli(base + ["--out", str(tmp / "x.json"),
                               "--retrieved", str(tmp / "ret.json")]) == 2,
              "--retrieved without --mode local must exit 2")

        # default --mode spell: today's report shape, no local-only keys
        rc = _run_cli(base + ["--out", str(tmp / "spell.json")])
        check(rc == 0, f"rc {rc}")
        sp = json.loads((tmp / "spell.json").read_text())
        check(all("gated" not in b for b in sp["kinds"].values()), sp["kinds"])
        check({"mode", "retrieved", "gate"}.isdisjoint(sp), sorted(sp))
        check(all("gated" not in r for r in sp["results"]), "spell results gained 'gated'")
        explicit = _run_cli(base + ["--out", str(tmp / "spell2.json"), "--mode", "spell"])
        check(explicit == 0, explicit)
        a, b = (json.loads((tmp / n).read_text()) for n in ("spell.json", "spell2.json"))
        for r in (a, b):
            r.pop("timing_ms")
        check(a == b, "--mode spell differs from the default")


# --------------------------------------------------------------------------
# scripts/ask.py --normalize local
# --------------------------------------------------------------------------

_ASK: dict = {}


class _AskGen:
    def __init__(self, *a, **kw):
        _ASK["generator"] = _ASK.get("generator", 0) + 1

    def health(self):
        return True

    def answer(self, question, hits, cite_grammar=True):
        _ASK.setdefault("answered", []).append(question)
        return "stub answer [1]"


class _AskEmb:
    def __init__(self, *a, **kw):
        _ASK["embedder"] = _ASK.get("embedder", 0) + 1

    def health(self):
        return True


class _AskRetriever:
    score = 0.9

    def __init__(self, *a, **kw):
        pass

    def retrieve(self, question, k=5):
        _ASK.setdefault("retrieved", []).append(question)
        return [_hit("c1", "wm.1", "Section: wm\n", "A window is resized.", _AskRetriever.score)]


def _run_ask(argv: list):
    _ASK.clear()
    saved = (ask.Generator, ask.Embedder, ask.Reranker, ask.Retriever, ask.store, sys.argv)
    ask.Generator, ask.Embedder, ask.Reranker = _AskGen, _AskEmb, _AskEmb
    ask.Retriever, ask.store = _AskRetriever, _StubStore
    sys.argv = ["ask.py"] + argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            rc = ask.main()
    finally:
        (ask.Generator, ask.Embedder, ask.Reranker, ask.Retriever, ask.store,
         sys.argv) = saved
    return rc, buf.getvalue()


def test_ask_local_rejects_incompatible_flags():
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "t.db"
        _make_db(db, CORPUS)
        base = ["--db", str(db), "--vocab-cache", str(Path(td) / "v.json"),
                "--normalize", "local"]
        for flag in (["--cascade"], ["--llm-correct"], ["--act"]):
            rc, out = _run_ask(base + flag + ["how to resize the windw manager"])
            check(rc == 2, f"{flag}: rc {rc}")
            # exits before any server is started or contacted
            check(not _ASK, f"{flag}: started something: {_ASK}")

        # compatible: retrieval and the gate see the raw question, the reader the repair
        rc, out = _run_ask(base + ["how to resize the windw manager"])
        check(rc == 0, f"rc {rc}")
        check(_ASK["retrieved"] == ["how to resize the windw manager"], _ASK["retrieved"])
        check(_ASK["answered"] == ["how to resize the window manager"], _ASK["answered"])
        check('(read as: "how to resize the window manager")' in out, out)

        # nothing to repair: no "(read as" line
        rc, out = _run_ask(base + ["how to resize the window manager"])
        check(rc == 0 and "(read as" not in out, out)

        # gated: refused, the generator is never constructed, the reader is never asked
        _AskRetriever.score = 0.2
        try:
            rc, out = _run_ask(base + ["how to resize the windw manager"])
        finally:
            _AskRetriever.score = 0.9
        check(rc == 0 and "generator" not in _ASK and "(read as" not in out, (_ASK, out))

        # default off: the question goes through untouched
        rc, out = _run_ask(["--db", str(db), "how to resize the windw manager"])
        check(_ASK["answered"] == ["how to resize the windw manager"], _ASK)
        check("(read as" not in out, out)


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
