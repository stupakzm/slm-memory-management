"""Tests for scripts/build_raft_data.py (RAFT-style data for the 4B reader).

Hermetic (blk_test_env_constraints): /usr/bin/python3, no pytest, no server, no models,
no data/. The planning core takes injectable search_fn / embed_fn, so no sqlite_vec is
needed (it is stubbed only because importing smm pulls the store in); the one database
here is a plain sqlite file holding a `chunks` table.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import random
import sqlite3
import sys
import tempfile
import types
from pathlib import Path

if "sqlite_vec" not in sys.modules:
    stub = types.ModuleType("sqlite_vec")
    stub.load = lambda *a, **kw: None
    sys.modules["sqlite_vec"] = stub

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import generate, grammar  # noqa: E402

_spec = importlib.util.spec_from_file_location("build_raft_data", ROOT / "scripts" / "build_raft_data.py")
brd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(brd)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india",
         "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo",
         "sierra", "tango", "uniform", "victor", "whiskey", "xray", "yankee", "zulu"]


def make_db(path: Path, per_domain: int = 150, tiny: int = 0) -> None:
    """Two domains of `per_domain` chunks in 6-chunk docs; `tiny` extra chunks of a
    third domain ('lone') that has too few chunks for four distractors."""
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE chunks(rowid INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE NOT NULL, "
               "doc_id TEXT NOT NULL, domain TEXT NOT NULL, prefix TEXT NOT NULL DEFAULT '', "
               "text TEXT NOT NULL)")
    rows = []
    for dom in ("emacs", "linux"):
        for i in range(per_domain):
            rows.append((f"{dom}{i:03d}", f"doc{dom}{i // 6}", dom, f"[{dom}] ",
                         f"Chunk {dom}{i:03d} explains the --flag-{i} switch and C-x {i}."))
    for i in range(tiny):
        rows.append((f"lone{i}", "doclone", "lone", "", f"lonely chunk {i}"))
    db.executemany("INSERT INTO chunks(chunk_id, doc_id, domain, prefix, text) VALUES(?,?,?,?,?)", rows)
    db.commit()
    db.close()


def cache_for(chunks, per_chunk: int = 2) -> dict:
    out = {}
    for c in chunks:
        i = int(c["chunk_id"][-3:]) if c["chunk_id"][-3:].isdigit() else 0
        out[c["chunk_id"]] = [f"how can a person {WORDS[i % 26]} the {WORDS[(i + j + 1) % 26]} "
                              f"thing number {i} {c['domain']} {j}" for j in range(per_chunk)]
    return out


class Fake:
    """search_fn / embed_fn pair: the 'vector' is the question itself; the gold chunk is
    always the best hit, the rest follow in a stable hashed order, so a test can see
    whether the planner drops the gold chunk itself from the distractors."""

    def __init__(self, chunks, gold_of):
        self.chunks = chunks
        self.gold_of = gold_of          # question -> gold chunk_id
        self.embeds = 0
        self.searches = 0

    def embed(self, q):
        self.embeds += 1
        return q

    def search(self, vec, k, domain):
        self.searches += 1
        pool = [c for c in self.chunks if c["domain"] == domain]
        gold = self.gold_of.get(vec)
        pool.sort(key=lambda c: (c["chunk_id"] != gold,
                                 hashlib.sha256((vec + c["chunk_id"]).encode()).hexdigest()))
        return [dict(c, distance=0.1) for c in pool[:k]]


def world(per_domain: int = 150, tiny: int = 0, n: int = 100, seed: int = 15,
          gold_removed: float = 0.25, typo_share: float = 0.30, excluded=(), eval_qs=()):
    tmp = Path(tempfile.mkdtemp())
    make_db(tmp / "i.db", per_domain, tiny)
    db = sqlite3.connect(tmp / "i.db")
    chunks = brd.load_chunks(db)
    db.close()
    cache = cache_for(chunks)
    gold_of = {q: cid for cid, qs in cache.items() for q in qs}
    pairs = brd.eligible_pairs(chunks, [cache], set(excluded), brd.leak_index(eval_qs), 0.8)
    decs = brd.choose(pairs, n, seed, gold_removed, typo_share)
    return tmp, chunks, cache, Fake(chunks, gold_of), decs


def build_plan(**kw):
    tmp, chunks, cache, fake, decs = world(**kw)
    plan_path = tmp / "plan.jsonl"
    brd.run_plan(decs, fake.search, fake.embed, plan_path)
    return tmp, fake, decs, brd.read_jsonl(plan_path)


def gold_row(gold_pos=2, typo=False, question="how do I list files"):
    ex = [{"chunk_id": f"c{i}", "doc_id": f"d{i}", "prefix": "", "text": f"text {i}"} for i in range(5)]
    ex[gold_pos] = {"chunk_id": "gold", "doc_id": "dg", "prefix": "[p] ",
                    "text": "Use --all to list hidden files, or -a."}
    return {"id": "r1", "question": question, "question_clean": question, "typo": typo,
            "gold_chunk": "gold", "gold_removed": False, "extracts": ex, "gold_pos": gold_pos}


# ----------------------------------------------------------------------- tests

def test_excluded_docs_come_from_eval_gold_and_variants():
    rows = [
        {"qid": "a1", "question": "q", "doc": "tar.1", "gold_sec_ids": ["tar.1#OPTIONS/x", "grep.1#SEC"]},
        {"qid": "a1.y1", "question": "q2", "variant_of": "a1"},
        {"qid": "a2.s", "question": "q3", "variant_of": "a2", "doc": "ls.1", "gold_sec_ids": []},
        {"qid": "a2", "question": "q4", "doc": "ls.1", "gold_sec_ids": ["emacs.emacs#visiting"]},
        {"qid": "a3.p", "question": "q5", "paraphrase_of": "a3"},
        {"qid": "a3", "question": "q6", "doc": "sed.1"},
        {"qid": "n1", "request": "no doc at all"},
    ]
    got = brd.excluded_docs(rows)
    check(got == {"tar.1", "grep.1", "ls.1", "emacs.emacs", "sed.1"}, got)
    # a variant file alone, without its base, still contributes its own gold
    check(brd.excluded_docs(rows[2:3]) == {"ls.1"}, "variant alone")
    # the base's gold reaches a variant that carries none of its own
    check("grep.1" in brd.excluded_docs([rows[1], rows[0]]), "variant listed before its base")
    # and the exclusion is by whole doc: no chunk of an excluded doc survives selection
    tmp, chunks, cache, fake, decs = world(per_domain=60, n=500, excluded={"docemacs0", "doclinux1"})
    check(decs, "something planned")
    check(all(d["chunk"]["doc_id"] not in {"docemacs0", "doclinux1"} for d in decs), "excluded doc leaked")
    check(any(d["chunk"]["doc_id"] == "docemacs1" for d in decs), "other docs still used")


def test_questions_too_similar_to_eval_are_dropped():
    ev = ["How do I swap two words around the cursor"]
    idx = brd.leak_index(ev)
    check(brd.leaks("how do i swap two words around the cursor?", idx, 0.8), "verbatim up to case/punct")
    check(brd.leaks("How do I swap two words around the cursor quickly", idx, 0.8), "8/9 = 0.89")
    check(not brd.leaks("swap words", idx, 0.8), "unrelated enough")
    check(not brd.leaks("How do I swap two words around the cursor quickly please now", idx, 0.8), "8/11")
    check(brd.leaks("swap words", idx, 0.2), "threshold is a parameter")
    chunk = {"chunk_id": "c1", "doc_id": "d", "domain": "emacs", "prefix": "", "text": "t"}
    cache = {"c1": ["how do i swap two words around the cursor", "what is the frobnicator for"]}
    pairs = brd.eligible_pairs([chunk], [cache], set(), idx, 0.8)
    check([q for _, q in pairs] == ["what is the frobnicator for"], pairs)


def test_plan_is_deterministic_for_a_seed():
    _, _, d1, p1 = build_plan(n=60, seed=15)
    _, _, d2, p2 = build_plan(n=60, seed=15)
    _, _, d3, p3 = build_plan(n=60, seed=16)
    check(len(p1) == 60, len(p1))
    check(p1 == p2, "same seed, same plan")
    check(json.dumps(p1) == json.dumps(p2), "same bytes")
    check([r["id"] for r in p1] != [r["id"] for r in p3], "another seed picks other pairs")
    check(len({r["id"] for r in p1}) == 60, "ids unique")


def test_gold_removed_share_and_refusal_target():
    _, _, _, plan = build_plan(n=400)
    removed = [r for r in plan if r["gold_removed"]]
    share = len(removed) / len(plan)
    check(0.18 < share < 0.32, share)
    for r in removed:
        check(r["gold_pos"] is None and len(r["extracts"]) == 5, r["id"])
        check(r["gold_chunk"] not in [e["chunk_id"] for e in r["extracts"]], "gold not removed")
    for share_arg, want in ((0.0, 0), (1.0, 40)):
        _, _, _, p = build_plan(n=40, gold_removed=share_arg)
        check(sum(r["gold_removed"] for r in p) == want, (share_arg, p[0]["gold_removed"]))
    rows, counts = brd.assemble(plan, {r["id"]: "Use --flag-1 [1]." for r in plan})
    refused = [x for x in rows if x["gold_removed"]]
    check(len(refused) == len(removed) == counts["refusals"], counts)
    check(all(x["messages"][-1] == {"role": "assistant", "content": grammar.REFUSAL} for x in refused),
          "target of a gold-removed row is the refusal")
    check(grammar.REFUSAL == "I don't know.", grammar.REFUSAL)


def test_context_has_five_extracts_gold_position_shuffled():
    _, _, _, plan = build_plan(n=200)
    kept = [r for r in plan if not r["gold_removed"]]
    check(all(len(r["extracts"]) == 5 for r in plan), "five extracts")
    check(all(r["extracts"][r["gold_pos"]]["chunk_id"] == r["gold_chunk"] for r in kept), "gold at gold_pos")
    check({r["gold_pos"] for r in kept} == {0, 1, 2, 3, 4}, {r["gold_pos"] for r in kept})
    counts = [sum(r["gold_pos"] == p for r in kept) for p in range(5)]
    check(min(counts) > len(kept) / 5 * 0.5, counts)
    check(all(len({e["chunk_id"] for e in r["extracts"]}) == 5 for r in plan), "no repeated extract")
    check(all(set(e) == {"chunk_id", "doc_id", "prefix", "text"} for r in plan for e in r["extracts"]),
          "extract fields")


def test_distractors_never_include_the_gold_chunk():
    tmp, fake, decs, plan = build_plan(n=120)
    for r in plan:
        others = [e["chunk_id"] for i, e in enumerate(r["extracts"]) if i != r["gold_pos"]]
        check(r["gold_chunk"] not in others, r["id"])
    # the fake search really did return the gold chunk first, so the check above bites
    r = plan[0]
    hits = fake.search(r["question_clean"], 20, "emacs" if r["gold_chunk"].startswith("emacs") else "linux")
    check(hits[0]["chunk_id"] == r["gold_chunk"], "gold was the top hit")
    # fewer than 4 distractors -> the pair is dropped, and not searched again on resume
    tmp, chunks, cache, fake2, decs2 = world(per_domain=30, tiny=3, n=400)
    lone = [d for d in decs2 if d["chunk"]["domain"] == "lone"]
    check(lone, "some pairs from the tiny domain")
    plan_path = tmp / "plan.jsonl"
    written, dropped = brd.run_plan(decs2, fake2.search, fake2.embed, plan_path)
    check(dropped >= len(lone) and written + dropped == len(decs2), (written, dropped))
    check(not any(r["gold_chunk"].startswith("lone") for r in brd.read_jsonl(plan_path)), "dropped")
    before = fake2.searches
    brd.run_plan(decs2, fake2.search, fake2.embed, plan_path)
    check(fake2.searches == before, "nothing searched twice")


def test_typo_share_and_typo_changes_one_word():
    _, _, _, plan = build_plan(n=400, typo_share=0.30)
    share = sum(r["typo"] for r in plan) / len(plan)
    check(0.2 < share < 0.4, share)
    for r in plan:
        check((r["question"] != r["question_clean"]) == r["typo"], r["id"])
    qs = ["How do I show hidden files in a directory listing?",
          "when something matches I also want the lines around it",
          "which command visits a file purely for viewing"]
    for q in qs:
        for s in range(50):
            t = brd.add_typo(q, random.Random(s))
            a, b = q.split(), t.split()
            check(len(a) == len(b), (q, t))
            diff = [(x, y) for x, y in zip(a, b) if x != y]
            check(len(diff) == 1, (q, t))
            x, y = diff[0]
            check(x[0] == y[0] and len(x.strip("?.")) >= 4 and abs(len(x) - len(y)) <= 1, (x, y))
            check(sorted(x) == sorted(y) or len(x) != len(y), (x, y))
    # flags, key chords, commands and short words are never touched
    q = "use --exclude-from or M-x find-file via -X on `grep` and C-x C-f"
    check(all(brd.add_typo(q, random.Random(s)) == q for s in range(30)), "protected tokens")
    check(brd.add_typo("how do I", random.Random(1)) == "how do I", "no 4-letter word: unchanged")
    kinds = {"swap": 0, "drop": 0, "double": 0}
    for s in range(200):
        w = brd._typo_word("example", random.Random(s))
        kinds["swap" if len(w) == 7 else "drop" if len(w) == 6 else "double"] += 1
        check(w != "example" and w[0] == "e", w)
    check(all(v > 20 for v in kinds.values()), kinds)


def test_teacher_prompt_shows_only_the_gold_chunk():
    row = gold_row(gold_pos=3, typo=True, question="how do I lsit files")
    row["question_clean"] = "how do I list files"
    msgs = brd.teacher_messages(row)
    check(msgs[0] == {"role": "system", "content": generate.SYSTEM_V2}, msgs[0])
    user = msgs[1]["content"]
    check(user.startswith("Documentation extracts:\n\n[1] dg\n[p] Use --all"), user)
    check("[2]" not in user and "text 0" not in user and "text 4" not in user, "distractor leaked")
    check(user.endswith("Question: how do I list files"), user)
    check("lsit" not in user, "the teacher reads the clean question")
    seen = {}

    def chat(messages, max_tokens, gram):
        seen.update(messages=messages, max_tokens=max_tokens, grammar=gram)
        return "Use --all [1]."

    tmp = Path(tempfile.mkdtemp())
    n = brd.run_teach([row, dict(row, id="r2", gold_removed=True)], chat, tmp / "t.jsonl")
    check(n == 1, "a gold-removed row needs no teacher call")
    check(seen["max_tokens"] == 200 and seen["grammar"] == grammar.cited_answer(1), seen)
    check(seen["messages"] == msgs, "same messages as teacher_messages")
    check(brd.read_jsonl(tmp / "t.jsonl") == [{"id": "r1", "answer": "Use --all [1]."}], "teach.jsonl")


def test_teacher_answer_dropped_when_identifier_not_in_chunk():
    gold = gold_row()["extracts"][2]
    check(brd.judge_answer("Use --all [1].", gold) == "ok", "identifier in chunk")
    check(brd.judge_answer("Use --everything [1].", gold) == "ungrounded", "invented flag")
    check(brd.judge_answer("Use --all or --almost-all [1].", gold) == "ungrounded", "one of two invented")
    check(brd.judge_answer("Use -a [1].", gold) == "ok", "short flag in chunk")
    check(brd.judge_answer("Use -Z [1].", gold) == "ungrounded", "short flag not in chunk")
    check(brd.judge_answer("Press C-x C-f [1].", gold) == "ungrounded", "chord not in chunk")
    check(brd.judge_answer("Hidden files are listed [1].", gold) == "ok", "no identifier at all")
    check(brd.judge_answer("Use --all.", gold) == "no_citation", "no citation")
    check(brd.judge_answer("I don't know.", gold) == "no_citation", "refusal has none")
    check(brd.judge_answer("Use --all [2].", gold) == "no_citation", "extract the teacher never saw")
    rows, counts = brd.assemble([gold_row()], {"r1": "Use --everything [1]."})
    check(rows == [] and counts["dropped_ungrounded"] == 1 and counts["kept"] == 0, counts)


def test_citation_remapped_to_shuffled_position():
    for pos in range(5):
        row = gold_row(gold_pos=pos)
        rows, _ = brd.assemble([row], {"r1": "Use --all [1]. It also works as -a [1]."})
        out = rows[0]["messages"][-1]["content"]
        check(out == f"Use --all [{pos + 1}]. It also works as -a [{pos + 1}].", out)
        check(("[1]" in out) == (pos == 0), out)
        user = rows[0]["messages"][1]["content"]
        check(f"[{pos + 1}] dg\n[p] Use --all" in user, "the cited number is the gold chunk's own")
    check(brd.remap_citations("a [1] b [1]", 4) == "a [5] b [5]", "single pass")


def test_assemble_writes_chat_messages_in_reader_format():
    tmp = Path(tempfile.mkdtemp())
    typo = gold_row(gold_pos=1, typo=True, question="how do I lsit files")
    typo["question_clean"] = "how do I list files"
    plain = dict(gold_row(gold_pos=4), id="r2")
    removed = dict(gold_row(), id="r3", gold_removed=True, gold_pos=None)
    brd.append_jsonl(tmp / "plan.jsonl", [typo, plain, removed])
    brd.append_jsonl(tmp / "teach.jsonl", [{"id": "r1", "answer": "Use --all [1]."},
                                           {"id": "r2", "answer": "Use -a [1]."}])
    with contextlib.redirect_stdout(io.StringIO()):
        brd.run_assemble(tmp / "plan.jsonl", tmp / "teach.jsonl", tmp / "train.jsonl")
    rows = brd.read_jsonl(tmp / "train.jsonl")
    check([r["id"] for r in rows] == ["r1", "r2", "r3"], rows)
    check(all(set(r) == {"id", "messages", "gold_removed", "typo"} for r in rows), rows[0].keys())
    for r, src in zip(rows, (typo, plain, removed)):
        want = generate.build_prompt(src["question"], src["extracts"], system=generate.SYSTEM_V2,
                                     header="Documentation extracts")
        check(r["messages"][:2] == want, "prompt is the reader's own run-time prompt")
        check([m["role"] for m in r["messages"]] == ["system", "user", "assistant"], r["messages"])
    check("how do I lsit files" in rows[0]["messages"][1]["content"], "the student reads the typo")
    check("list files" not in rows[0]["messages"][1]["content"].split("Question:")[1], "not the clean one")
    check(rows[0]["typo"] is True and rows[1]["typo"] is False, "typo flag")
    check(rows[0]["messages"][2]["content"] == "Use --all [2].", rows[0]["messages"][2])
    check(rows[2]["gold_removed"] is True and rows[2]["messages"][2]["content"] == "I don't know.", rows[2])
    # a plan row nobody taught yet stops the stage rather than writing a short set
    brd.append_jsonl(tmp / "plan2.jsonl", [typo])
    try:
        brd.run_assemble(tmp / "plan2.jsonl", tmp / "none.jsonl", tmp / "train2.jsonl")
    except SystemExit:
        pass
    else:
        raise AssertionError("assemble must refuse a plan row without a teacher answer")
    check(not (tmp / "train2.jsonl").exists(), "nothing written")


def test_stages_resume_without_redoing_work():
    tmp, chunks, cache, fake, decs = world(n=40)
    plan_path = tmp / "plan.jsonl"
    brd.run_plan(decs, fake.search, fake.embed, plan_path)
    full = plan_path.read_text()
    total = len(full.splitlines())
    check(total == 40 and fake.searches == 40, (total, fake.searches))
    fake.searches = 0
    brd.run_plan(decs, fake.search, fake.embed, plan_path)
    check(fake.searches == 0 and plan_path.read_text() == full, "a finished plan is skipped")
    plan_path.write_text("".join(full.splitlines(True)[:15]))
    brd.run_plan(decs, fake.search, fake.embed, plan_path)
    check(fake.searches == 25, fake.searches)
    check(plan_path.read_text() == full, "resumed plan equals the uninterrupted one")

    plan = brd.read_jsonl(plan_path)
    asked = []

    def chat(messages, max_tokens, gram):
        asked.append(messages[1]["content"])
        return "Chunk text [1]."

    teach_path = tmp / "teach.jsonl"
    n = brd.run_teach(plan, chat, teach_path, parallel=2)
    need = sum(not r["gold_removed"] for r in plan)
    check(n == need == len(asked), (n, need, len(asked)))
    done = teach_path.read_text()
    teach_path.write_text("".join(done.splitlines(True)[:10]))
    asked.clear()
    brd.run_teach(plan, chat, teach_path, parallel=2)
    check(len(asked) == need - 10, len(asked))
    check(sorted(teach_path.read_text().splitlines()) == sorted(done.splitlines()), "same teach rows")
    asked.clear()
    check(brd.run_teach(plan, chat, teach_path) == 0 and not asked, "finished teach is skipped")

    out = tmp / "train.jsonl"
    with contextlib.redirect_stdout(io.StringIO()):
        check(brd.run_assemble(plan_path, teach_path, out) is not None, "first assemble runs")
    out.write_text("sentinel\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        check(brd.run_assemble(plan_path, teach_path, out) is None, "existing output skipped")
    check(out.read_text() == "sentinel\n", "not rewritten")


def test_counts_line_reports_kept_and_dropped():
    rows = [
        dict(gold_row(), id="ok1", typo=True),
        dict(gold_row(), id="ok2"),
        dict(gold_row(), id="bad"),
        dict(gold_row(), id="nocite"),
        dict(gold_row(), id="ref", gold_removed=True, gold_pos=None, typo=True),
    ]
    teach = {"ok1": "Use --all [1].", "ok2": "Use -a [1].", "bad": "Use --nope [1].",
             "nocite": "Use --all."}
    out, counts = brd.assemble(rows, teach)
    check(brd.counts_line(counts) == "plan 5 kept 3 dropped_ungrounded 1 dropped_no_citation 1 "
                                     "refusals 1 typos 2", brd.counts_line(counts))
    check(len(out) == 3, len(out))
    tmp = Path(tempfile.mkdtemp())
    brd.append_jsonl(tmp / "plan.jsonl", rows)
    brd.append_jsonl(tmp / "teach.jsonl", [{"id": k, "answer": v} for k, v in teach.items()])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        brd.run_assemble(tmp / "plan.jsonl", tmp / "teach.jsonl", tmp / "train.jsonl")
    check(buf.getvalue().strip() == "plan 5 kept 3 dropped_ungrounded 1 dropped_no_citation 1 "
                                    "refusals 1 typos 2", buf.getvalue())


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
