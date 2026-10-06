#!/usr/bin/env python3
"""Build a RAFT-style fine-tuning set for the 4B reader, in three resumable stages.

Reading has been the bottleneck since phase 8: prompting went as far as it can. The
standard fix is to train the reader on contexts like the ones it meets at run time -
one gold chunk among dense-search distractors, and a share of contexts with the gold
chunk removed, whose target is the refusal.

  build_raft_data.py plan     --work DIR --questions qvec-emacs.json --questions qvec-linux.json \
                              --eval data/eval/questions.jsonl ...   (needs the embedder server)
  build_raft_data.py teach    --work DIR --gen-url URL               (needs the 30B server)
  build_raft_data.py assemble --work DIR                             (no server)

plan   picks (chunk, question) pairs, builds each context (gold + 4 dense distractors with
       the gold at a random position, or 5 distractors with the gold removed), adds a typo
       to a share of the questions, and writes plan.jsonl.
teach  asks a stronger model for a short cited answer from the gold chunk ALONE (the clean
       question, the cited_answer(1) grammar) and writes teach.jsonl.
assemble writes the training examples (chat messages in the reader's own run-time format):
       the refusal for removed-gold rows; for the rest the teacher's answer with its [1]
       rewritten to the gold chunk's final position, kept only when every identifier it
       names occurs in that chunk.

Evaluation never leaks in: every chunk of every doc that is a gold doc of any eval row
(for variants, their base row's too) is excluded - the whole doc, so no eval gold page is
trained on - and a question whose token Jaccard against any eval question reaches
--leakage-jaccard is dropped (the R13 check of scripts/train_embedder.py, restated).

Every stage is idempotent: an existing output is skipped, a partial JSONL is appended to
by skipping the ids already present. Stdlib only; the servers are used through smm.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import generate, grammar  # noqa: E402

HEADER = "Documentation extracts"
N_EXTRACTS = 5
N_GOLD_DISTRACTORS = N_EXTRACTS - 1
SEARCH_K = 20
TEACH_BATCH = 64
_TOKEN = re.compile(r"[a-z0-9]+")
_CITATION = re.compile(r"\[[1-9]\]")
_PUNCT = ".,;:?!\"'()"

_gr = None


def grounding():
    """scripts/grounding_report.py, loaded on first use (tight_idents)."""
    global _gr
    if _gr is None:
        spec = importlib.util.spec_from_file_location(
            "grounding_report", ROOT / "scripts" / "grounding_report.py")
        _gr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_gr)
    return _gr


# ---------------------------------------------------------------- eval exclusion

def read_eval_rows(path: Path) -> list[dict]:
    text = Path(path).read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(l) for l in text.splitlines() if l.strip()]
    return [d for d in data if isinstance(d, dict)]


def _own_docs(row: dict) -> set:
    docs = {row["doc"]} if row.get("doc") else set()
    for sec in row.get("gold_sec_ids") or []:
        docs.add(sec.split("#", 1)[0])
    return docs


def excluded_docs(eval_rows: list[dict]) -> set:
    """doc_ids of every eval gold page: a row's `doc` and the doc part of each of its
    gold_sec_ids, plus the same for the base row of a variant or paraphrase."""
    by_qid = {r["qid"]: r for r in eval_rows if r.get("qid")}
    out = set()
    for row in eval_rows:
        seen = set()
        while row is not None and id(row) not in seen:
            seen.add(id(row))
            out |= _own_docs(row)
            row = by_qid.get(row.get("variant_of") or row.get("paraphrase_of"))
    return out


# ------------------------------------------------------------------- leakage

def tokens(s: str) -> frozenset:
    return frozenset(_TOKEN.findall(s.lower()))


def leak_index(eval_questions) -> dict:
    """token -> the eval questions' token sets containing it."""
    postings: dict[str, list] = {}
    for t in {tokens(q) for q in eval_questions}:
        for w in t:
            postings.setdefault(w, []).append(t)
    return postings


def leaks(question: str, postings: dict, threshold: float) -> bool:
    """True when some eval question's token Jaccard against `question` is >= threshold."""
    q = tokens(question)
    seen = set()
    for w in q:
        for t in postings.get(w, ()):
            if id(t) in seen:
                continue
            seen.add(id(t))
            if len(q & t) / len(q | t) >= threshold:
                return True
    return False


# ------------------------------------------------------------------ selection

def load_chunks(db) -> list[dict]:
    cur = db.execute("SELECT chunk_id, doc_id, domain, prefix, text FROM chunks ORDER BY rowid")
    return [{"chunk_id": r[0], "doc_id": r[1], "domain": r[2], "prefix": r[3], "text": r[4]}
            for r in cur.fetchall()]


def pair_id(chunk_id: str, question: str) -> str:
    return "raft-" + hashlib.sha256(f"{chunk_id}\n{question}".encode()).hexdigest()[:12]


def quotas(counts: dict, n: int) -> dict:
    """n split across the keys of `counts` in proportion to their values (largest remainder)."""
    total = sum(counts.values())
    if not total:
        return {k: 0 for k in counts}
    exact = {k: n * v / total for k, v in counts.items()}
    out = {k: int(x) for k, x in exact.items()}
    for k in sorted(counts, key=lambda k: (-(exact[k] - out[k]), k))[:n - sum(out.values())]:
        out[k] += 1
    return out


def eligible_pairs(chunks, caches: list[dict], excluded: set, postings: dict,
                   threshold: float) -> list[tuple]:
    """(chunk, question) for every cached question of a chunk outside the excluded docs
    that does not leak an eval question; chunk order, then question order, no repeats."""
    cache = {}
    for c in caches:
        for cid, qs in c.items():
            cache.setdefault(cid, []).extend(qs)
    out, seen = [], set()
    for ch in chunks:
        if ch["doc_id"] in excluded:
            continue
        for q in cache.get(ch["chunk_id"], ()):
            pid = pair_id(ch["chunk_id"], q)
            if q.strip() and pid not in seen and not leaks(q, postings, threshold):
                seen.add(pid)
                out.append((ch, q))
    return out


def choose(pairs, n: int, seed: int, gold_removed: float, typo_share: float) -> list[dict]:
    """Sample n pairs, balanced across domains in proportion to each domain's chunk count,
    and draw every random decision of the plan up front from random.Random(seed): this
    is what lets plan resume without changing a single decision."""
    rng = random.Random(seed)
    by_dom: dict[str, list] = {}
    for p in pairs:
        by_dom.setdefault(p[0]["domain"], []).append(p)
    nchunks = {d: len({p[0]["chunk_id"] for p in ps}) for d, ps in by_dom.items()}
    quota = quotas(nchunks, n)
    picked = []
    for d in sorted(by_dom):
        picked += rng.sample(by_dom[d], min(quota[d], len(by_dom[d])))
    rng.shuffle(picked)
    decs = []
    for ch, q in picked:
        decs.append({
            "id": pair_id(ch["chunk_id"], q), "chunk": ch, "question": q,
            "removed": rng.random() < gold_removed, "typo": rng.random() < typo_share,
            "pos": rng.randrange(N_EXTRACTS), "tseed": rng.getrandbits(32),
        })
    return decs


# ----------------------------------------------------------------------- typos

def _typo_word(word: str, rng: random.Random) -> str:
    """One edit of a word, never of its first letter: swap two adjacent different
    letters, drop one letter, or double one letter."""
    ops = ["drop", "double"]
    swaps = [i for i in range(1, len(word) - 1) if word[i] != word[i + 1]]
    if swaps:
        ops.append("swap")
    op = rng.choice(ops)
    if op == "swap":
        i = rng.choice(swaps)
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    i = rng.randrange(1, len(word))
    return word[:i] + word[i + 1:] if op == "drop" else word[:i + 1] + word[i] + word[i + 1:]


def add_typo(question: str, rng: random.Random) -> str:
    """The question with one typo in one plain word of 4+ letters; a token that looks like
    a flag, key chord or command (anything but letters between punctuation: a '-', '_',
    '/', digit, backtick ...) is never touched. Unchanged when no word qualifies."""
    cands = []
    for m in re.finditer(r"\S+", question):
        core = m.group().strip(_PUNCT)
        if len(core) >= 4 and core.isascii() and core.isalpha():
            cands.append((m.start() + m.group().find(core), core))
    if not cands:
        return question
    start, word = rng.choice(cands)
    return question[:start] + _typo_word(word, rng) + question[start + len(word):]


# ------------------------------------------------------------------------ plan

def _extract(c: dict) -> dict:
    return {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "prefix": c["prefix"],
            "text": c["text"]}


def plan_row(dec: dict, search_fn, embed_fn) -> dict | None:
    """The plan row for one decision, or None when dense search finds too few distractors
    (4, or 5 when the gold chunk is removed)."""
    gold = dec["chunk"]
    hits = search_fn(embed_fn(dec["question"]), SEARCH_K, gold["domain"])
    distractors = [_extract(h) for h in hits if h["chunk_id"] != gold["chunk_id"]]
    need = N_EXTRACTS if dec["removed"] else N_GOLD_DISTRACTORS
    if len(distractors) < need:
        return None
    if dec["removed"]:
        extracts, gold_pos = distractors[:N_EXTRACTS], None
    else:
        extracts = distractors[:N_GOLD_DISTRACTORS]
        extracts.insert(dec["pos"], _extract(gold))
        gold_pos = dec["pos"]
    asked = dec["question"]
    if dec["typo"]:
        asked = add_typo(asked, random.Random(dec["tseed"]))
    return {"id": dec["id"], "question": asked, "question_clean": dec["question"],
            "typo": asked != dec["question"], "gold_chunk": gold["chunk_id"],
            "gold_removed": dec["removed"], "extracts": extracts, "gold_pos": gold_pos}


def read_jsonl(path: Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def append_jsonl(path: Path, rows) -> None:
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_plan(decs: list[dict], search_fn, embed_fn, plan_path: Path) -> tuple[int, int]:
    """Append a plan row for every decision not already in plan_path (or recorded as
    dropped in plan_path.dropped); returns (written, dropped) for this run."""
    plan_path = Path(plan_path)
    dropped_path = plan_path.with_name(plan_path.name + ".dropped")
    have = {r["id"] for r in read_jsonl(plan_path)}
    if dropped_path.exists():
        have |= set(dropped_path.read_text().split())
    written = dropped = 0
    for dec in decs:
        if dec["id"] in have:
            continue
        row = plan_row(dec, search_fn, embed_fn)
        if row is None:
            with open(dropped_path, "a") as f:
                f.write(dec["id"] + "\n")
            dropped += 1
            continue
        append_jsonl(plan_path, [row])
        written += 1
    return written, dropped


# ----------------------------------------------------------------------- teach

def teacher_messages(row: dict) -> list[dict]:
    """The teacher sees the clean question and ONLY the gold chunk, as extract [1]."""
    gold = next(e for e in row["extracts"] if e["chunk_id"] == row["gold_chunk"])
    return generate.build_prompt(row["question_clean"], [gold], system=generate.SYSTEM_V2,
                                 header=HEADER)


def run_teach(plan_rows: list[dict], chat_fn, teach_path: Path, parallel: int = 4,
              max_tokens: int = 200) -> int:
    """Ask for an answer to every gold-present plan row not yet in teach_path; returns
    how many. chat_fn(messages, max_tokens, grammar) -> text."""
    have = {r["id"] for r in read_jsonl(teach_path)}
    todo = [r for r in plan_rows if not r["gold_removed"] and r["id"] not in have]
    g = grammar.cited_answer(1)

    def one(row):
        return {"id": row["id"], "answer": chat_fn(teacher_messages(row), max_tokens, g)}

    with ThreadPoolExecutor(parallel) as pool:
        for i in range(0, len(todo), TEACH_BATCH):
            append_jsonl(teach_path, pool.map(one, todo[i:i + TEACH_BATCH]))
            print(f"  taught {min(i + TEACH_BATCH, len(todo))}/{len(todo)}", flush=True)
    return len(todo)


# -------------------------------------------------------------------- assemble

def remap_citations(answer: str, gold_pos: int) -> str:
    """The teacher saw one extract, so every citation is [1]: point it at [gold_pos+1]."""
    return re.sub(r"\[1\]", f"[{gold_pos + 1}]", answer)


def judge_answer(answer: str, gold: dict) -> str:
    """'ok', 'no_citation' (no [1], or a citation to an extract the teacher never saw) or
    'ungrounded' (an identifier of the answer that is not in the gold chunk)."""
    cites = _CITATION.findall(answer)
    if "[1]" not in cites or any(c != "[1]" for c in cites):
        return "no_citation"
    haystack = gold["prefix"] + gold["text"]
    if any(i not in haystack for i in grounding().tight_idents(answer)):
        return "ungrounded"
    return "ok"


def example(row: dict, target: str) -> dict:
    msgs = generate.build_prompt(row["question"], row["extracts"], system=generate.SYSTEM_V2,
                                 header=HEADER)
    msgs.append({"role": "assistant", "content": target})
    return {"id": row["id"], "messages": msgs, "gold_removed": row["gold_removed"],
            "typo": row["typo"]}


def assemble(plan_rows: list[dict], teach: dict) -> tuple[list[dict], dict]:
    """(examples, counts). A row without a teacher answer raises: assemble never writes a
    set that silently lacks the rows teach has not reached."""
    out = []
    counts = {"plan": len(plan_rows), "kept": 0, "dropped_ungrounded": 0,
              "dropped_no_citation": 0, "refusals": 0, "typos": 0}
    for row in plan_rows:
        if row["gold_removed"]:
            target = grammar.REFUSAL
            counts["refusals"] += 1
        else:
            if row["id"] not in teach:
                raise SystemExit(f"no teacher answer for {row['id']}: run teach first")
            gold = row["extracts"][row["gold_pos"]]
            verdict = judge_answer(teach[row["id"]], gold)
            if verdict != "ok":
                counts["dropped_" + verdict] += 1
                continue
            target = remap_citations(teach[row["id"]], row["gold_pos"])
        out.append(example(row, target))
        counts["kept"] += 1
        counts["typos"] += bool(row["typo"])
    return out, counts


def counts_line(c: dict) -> str:
    return (f"plan {c['plan']} kept {c['kept']} dropped_ungrounded {c['dropped_ungrounded']} "
            f"dropped_no_citation {c['dropped_no_citation']} refusals {c['refusals']} "
            f"typos {c['typos']}")


def run_assemble(plan_path: Path, teach_path: Path, out_path: Path) -> dict | None:
    out_path = Path(out_path)
    if out_path.exists():
        print(f"{out_path} exists, skipping")
        return None
    teach = {r["id"]: r["answer"] for r in read_jsonl(teach_path)}
    rows, counts = assemble(read_jsonl(plan_path), teach)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(out_path)
    print(counts_line(counts))
    return counts


# ------------------------------------------------------------------------- CLI

def cmd_plan(args) -> None:
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    plan_path = work / "plan.jsonl"
    meta = {k: getattr(args, k) for k in ("db", "questions", "eval", "leakage_jaccard", "n",
                                          "seed", "gold_removed", "typo_share")}
    meta_path = work / "plan.meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise SystemExit(f"{meta_path} differs from these arguments: resuming would mix plans")
    meta_path.write_text(json.dumps(meta, indent=1))

    from smm import store
    from smm.embed import Embedder
    db = store.connect(Path(args.db))
    chunks = load_chunks(db)
    eval_rows = [r for p in args.eval for r in read_eval_rows(Path(p))]
    eval_qs = [r.get("question") or r.get("request") for r in eval_rows]
    caches = [json.loads(Path(p).read_text()) for p in args.questions]
    pairs = eligible_pairs(chunks, caches, excluded_docs(eval_rows),
                           leak_index([q for q in eval_qs if q]), args.leakage_jaccard)
    decs = choose(pairs, args.n, args.seed, args.gold_removed, args.typo_share)
    print(f"{len(pairs)} eligible pairs, {len(decs)} planned", flush=True)
    emb = Embedder(args.embed_url)
    written, dropped = run_plan(decs, lambda v, k, d: store.search(db, v, k, d),
                                emb.embed_query, plan_path)
    print(f"plan: wrote {written}, dropped {dropped} (too few distractors)")


def cmd_teach(args) -> None:
    work = Path(args.work)
    plan = read_jsonl(Path(args.plan or work / "plan.jsonl"))
    gen = generate.Generator(args.gen_url)
    n = run_teach(plan, lambda m, mt, g: gen.chat(m, max_tokens=mt, temperature=0.0, grammar=g),
                  Path(args.teach or work / "teach.jsonl"), args.parallel, args.max_tokens)
    print(f"teach: asked {n}")


def cmd_assemble(args) -> None:
    work = Path(args.work)
    run_assemble(Path(args.plan or work / "plan.jsonl"), Path(args.teach or work / "teach.jsonl"),
                 Path(args.out or work / "train.jsonl"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--work", required=True)
    p.add_argument("--db", default="data/index/phase11.db")
    p.add_argument("--questions", action="append", required=True,
                   help="question cache {chunk_id: [questions]} (repeatable)")
    p.add_argument("--eval", action="append", default=[], help="eval questions file (repeatable)")
    p.add_argument("--leakage-jaccard", type=float, default=0.8)
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--seed", type=int, default=15)
    p.add_argument("--gold-removed", type=float, default=0.25)
    p.add_argument("--typo-share", type=float, default=0.30)
    p.add_argument("--embed-url", default="http://127.0.0.1:8081")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("teach")
    p.add_argument("--work", required=True)
    p.add_argument("--plan")
    p.add_argument("--teach", help="output (default WORK/teach.jsonl)")
    p.add_argument("--gen-url", required=True, help="the 30B llama-server")
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--parallel", type=int, default=4)
    p.set_defaults(fn=cmd_teach)
    p = sub.add_parser("assemble")
    p.add_argument("--work", required=True)
    p.add_argument("--plan")
    p.add_argument("--teach")
    p.add_argument("--out", help="output (default WORK/train.jsonl)")
    p.set_defaults(fn=cmd_assemble)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
