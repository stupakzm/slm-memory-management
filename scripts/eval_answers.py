#!/usr/bin/env python3
"""Score generated answers: correctness, and the abstention behaviour that is the
whole point of the project.

Phase 1 had no gate and no grammar - abstention was requested in the prompt and
enforced nowhere - and the result was the project's worst number: when retrieval
found nothing useful, the model invented an answer 56.5% of the time, sometimes
with a fabricated citation attached. Phase 2 adds two mechanisms and this script
turns each on independently, because a bundle that improves the headline tells you
nothing about which half did it:

  --gate T     refuse before generation when the top-1 score is below T; the model
               is never invoked, so it cannot speculate
  --grammar    constrained decoding: every claim carries an in-range citation

Two stages, because 6 GB of VRAM does not hold the embedder, the reranker and the
4B generator at once.

  .venv/bin/python scripts/eval_answers.py --stage retrieve --mode dense --rerank --name phase2-full
  ./scripts/servers.sh stop && ./scripts/servers.sh start generator
  .venv/bin/python scripts/eval_answers.py --stage generate --name phase2-full --gate 0.65 --grammar
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import gold, grammar, lexical, store  # noqa: E402
from smm import normalize as qnorm  # noqa: E402
from smm.cascade import ABSTAIN_RE, abstained, run_cascade  # noqa: E402
from smm.embed import Embedder  # noqa: E402
from smm.generate import Generator  # noqa: E402
from smm.rerank import Reranker  # noqa: E402
from smm.retrieve import Retriever, cap_per_doc, expand, gate_score  # noqa: E402

# ABSTAIN_RE/abstained moved to smm.cascade (phase 11 R4d build): the cascade
# is the thing that has to decide, tier by tier, whether the reader refused,
# so this is where the rule now lives. Re-exported under the SAME names so
# `eval_answers.abstained`/`eval_answers.ABSTAIN_RE` keep working unchanged -
# see tests/test_scoring.py, which asserts the exact all-sentences behaviour
# and passes with no edits.


def evidence_in(hits: list, toks: list, token_aliases) -> tuple:
    """(aliased, strict): `aliased` matches how `correct` is scored (via
    gold.is_correct, so an alias found in a hit counts); `strict` is the
    original substring-only check, kept so the two can be compared. With
    --no-aliases the caller passes an effectively empty token_aliases, so
    aliased == strict automatically."""
    aliased = any(gold.is_correct(h["text"], toks, token_aliases) for h in hits)
    strict = any(all(t in h["text"] for t in toks) for h in hits)
    return aliased, strict


def cache_entry(hits: list, gate_hits: list, rewrites: int, corrected: str | None = None) -> object:
    """Shape of one cached-retrieval entry. `rewrites == 0` and no `corrected`
    text returns `hits` unchanged - today's exact format, so `--rewrites 0`
    (with --llm-correct off) reproduces existing runs byte-for-byte.
    Otherwise returns a dict carrying the (possibly fused) `hits`, the
    original question's own `gate_hits` (see blk_fusion_gate_semantic_slip:
    the gate must never read the fused list's own top-1), and - only under
    --llm-correct - `question_corrected`, so the generate stage can reuse the
    SAME corrected text `correct_text` cached it with, rather than asking the
    model again."""
    if rewrites == 0 and corrected is None:
        return hits
    entry = {"hits": hits, "gate_hits": gate_hits}
    if corrected is not None:
        entry["question_corrected"] = corrected
    return entry


def unpack_entry(entry: object) -> tuple:
    """Inverse of `cache_entry`'s hits/gate_hits half. A plain list (legacy
    cache, or --rewrites 0 with --llm-correct off) gives `(entry, entry)`;
    the dict form gives both lists back out. See `correct_text` for the
    (optional) cached corrected question, which this does not return -
    every existing 2-tuple call site stays exactly as it was."""
    if isinstance(entry, dict):
        return entry["hits"], entry["gate_hits"]
    return entry, entry


def correct_text(entry: object) -> str | None:
    """The cached --llm-correct question text for one retrieval entry, or
    None if --llm-correct was off for this run (legacy/plain-list entries,
    and dict entries with no question_corrected key)."""
    return entry.get("question_corrected") if isinstance(entry, dict) else None


def select_hits(hits: list, read_k: int, cap: int) -> list:
    """Phase 7 (tsk_20260926_0967344e): cut what the MODEL reads, separately
    from what the gate reads. `gate_hits`/`gate_score` are untouched by this -
    see blk_fusion_gate_semantic_slip. Defaults (read_k=0, cap=0) return
    `hits` unchanged, reproducing today's behaviour exactly.

    `read_k<=0` means "every cached hit"; when capping is also requested
    (cap > 0) that means capping over the whole list, not over zero items -
    `cap_per_doc`'s own `k` is the read-k or the full length, never a bare
    0 that would silently empty the result."""
    if cap > 0:
        return cap_per_doc(hits, read_k if read_k > 0 else len(hits), cap)
    if read_k > 0:
        return hits[:read_k]
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/index/phase2.db")
    ap.add_argument("--eval", default="data/eval/questions.jsonl")
    ap.add_argument("--name", default="answers")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mode", choices=("dense", "bm25", "hybrid"), default="dense")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--domain", default=None,
                    help="restrict retrieval to one namespace; omit to let every "
                         "domain in the index compete, which is what phase 5 measures")
    ap.add_argument("--candidates", type=int, default=50)
    ap.add_argument("--expand", type=int, default=0,
                    help="structured index: widen each hit by N neighbouring chunks "
                         "within its own section before the model reads it")
    ap.add_argument("--gate", type=float, default=0.0,
                    help="refuse before generation below this top-1 score (0 = no gate)")
    ap.add_argument("--grammar", action="store_true", help="GBNF-enforced citations")
    ap.add_argument("--stage", choices=("retrieve", "generate", "both"), default="both")
    ap.add_argument("--rewrites", type=int, default=0,
                    help="fuse each question with N model rewrites, each retrieved and "
                         "reranked separately; 0 reproduces existing runs exactly")
    ap.add_argument("--gen-url", default="http://127.0.0.1:8080",
                    help="generator llama-server, used for --rewrites > 0 (stage "
                         "retrieve) and for answer generation itself (stage generate)")
    ap.add_argument("--no-aliases", action="store_true",
                     help="restore strict scoring: no gold-token aliases (tsk_20260926_a51d0707)")
    ap.add_argument("--cache", default=None,
                    help="retrieval cache name, read and written as NAME-retrieved.json "
                         "(default: --name); set this to share one retrieval run (e.g. "
                         "made with a large -k) across several generation arms "
                         "(tsk_20260926_0967344e)")
    ap.add_argument("--read-k", type=int, default=0,
                    help="how many cached hits the MODEL reads, cutting the cache down "
                         "further at generate time (0 = every cached hit, today's "
                         "behaviour); never affects gate_hits (tsk_20260926_0967344e)")
    ap.add_argument("--cap-per-doc", type=int, default=0,
                    help="cap how many of the read hits may come from one doc_id, so "
                         "other documents' evidence is not crowded out (0 = off; "
                         "tsk_20260926_0967344e)")
    ap.add_argument("--answer-mode", choices=("cite", "quote"), default="cite",
                    help="phase 10: 'quote' requires each claim to open with an exact "
                         "quotation from the extract it cites, verified after generation "
                         "(grammar.verify_quotes); a failed quote converts the record to "
                         "the refusal before scoring. 'cite' (default) reproduces existing "
                         "runs exactly.")
    ap.add_argument("--qids", default=None,
                    help="path to a JSON list of qids; restricts rows to those qids, kept "
                         "in the eval set's own order, applied after --limit")
    ap.add_argument("--normalize", choices=("off", "spell"), default="off",
                    help="phase 11 R4a: 'spell' corrects the question against the index's "
                         "own vocabulary (smm.normalize) before retrieval AND generation; "
                         "'off' (default) reproduces every existing run byte-for-byte")
    ap.add_argument("--vocab-cache", default=None,
                    help="override path for --normalize spell's vocab cache (default: "
                         "<db>.vocab.json next to --db)")
    ap.add_argument("--rewrite-style", choices=("man", "docs"), default="man",
                    help="phase 11 R4b: 'docs' asks the rewriter for documentation's "
                         "own terminology (manual pages and the GNU Emacs manuals) "
                         "instead of everyday words; only meaningful with --rewrites > "
                         "0. 'man' (default) reproduces every existing run exactly.")
    ap.add_argument("--llm-correct", action="store_true",
                    help="phase 11 R4a': ask the 4B generator itself to fix spelling "
                         "in the question (Generator.correct) before retrieval AND "
                         "generation, reusing the SAME corrected text at both stages; "
                         "composes with --normalize (normalize runs first). Default off "
                         "reproduces every existing run byte-for-byte.")
    ap.add_argument("--cascade", action="store_true",
                    help="phase 11 R4d: rewrite only after the plain search refuses "
                         "(smm.cascade), replacing --rewrites' fusion path (--rewrites "
                         "is ignored). Requires --stage both: the tier depends on the "
                         "reader's own answer, so retrieval and generation run inline, "
                         "row by row, with all three servers resident - there is no "
                         "retrieval cache under --cascade. Records gain cascade_tier "
                         "(0/1/2), cascade_seconds (wall time beyond tier 0) and "
                         "cascade_rewrites ([] at tier 0). Default off reproduces "
                         "every existing run byte-for-byte.")
    args = ap.parse_args()
    args.cache = args.cache or args.name

    if args.cascade and args.stage != "both":
        print("--cascade requires --stage both (the tier depends on the reader's "
              "own answer)", file=sys.stderr)
        return 2

    qid_aliases = {} if args.no_aliases else gold.load_aliases(
        ROOT / "data" / "eval" / "gold_aliases.json")

    cache = ROOT / "data" / "eval" / "results" / f"{args.cache}-retrieved.json"
    rows = [json.loads(l) for l in (ROOT / args.eval).open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    if args.qids:
        qid_set = set(json.loads(Path(args.qids).read_text()))
        rows = [r for r in rows if r["qid"] in qid_set]

    # Normalised once, here, so retrieval (including rewrites/fusion) and
    # generation read the SAME corrected text in every stage of this run -
    # each of --stage retrieve/generate/both re-derives it identically
    # since both build_vocab (cached next to --db) and normalize() are pure
    # functions of --db's own content. `off` leaves `normalized` empty and
    # every downstream use falls back to `row["question"]` unchanged.
    normalized: dict = {}
    if args.normalize == "spell":
        vocab = qnorm.build_vocab(ROOT / args.db, cache_path=args.vocab_cache)
        for row in rows:
            normalized[row["qid"]] = qnorm.normalize_query(row["question"], "spell", vocab)

    def query_text(row: dict) -> str:
        return normalized[row["qid"]][0] if args.normalize == "spell" else row["question"]

    if args.cascade:
        return run_cascade_eval(args, rows, qid_aliases, normalized, query_text)

    # --- stage 1: retrieval only (embedder + reranker resident) ---
    if args.stage in ("retrieve", "both"):
        emb = None
        if args.mode in ("dense", "hybrid"):
            emb = Embedder()
            if not emb.health():
                print("embedder not running: ./scripts/servers.sh start embedder", file=sys.stderr)
                return 2
        rr = None
        if args.rerank:
            rr = Reranker()
            if not rr.health():
                print("reranker not running: ./scripts/servers.sh start reranker", file=sys.stderr)
                return 2
        rewrite_gen = None
        if args.rewrites or args.llm_correct:
            rewrite_gen = Generator(args.gen_url)
            if not rewrite_gen.health():
                print(f"no generator at {args.gen_url}: "
                      f"./scripts/servers.sh start generator", file=sys.stderr)
                return 2
        db = store.connect(ROOT / args.db)
        if args.mode in ("bm25", "hybrid") and not lexical.has_index(db):
            print(f"{args.db} has no chunks_fts", file=sys.stderr)
            return 2
        r = Retriever(db, embedder=emb, reranker=rr, mode=args.mode,
                      candidates=args.candidates, domain=args.domain)
        retrieved, t0 = {}, time.time()
        for i, row in enumerate(rows, 1):
            qtext = query_text(row)
            # --llm-correct composes with --normalize: normalize (above) runs
            # first, then the model correction, on the already-normalized text
            # (blk_normalize_spell_api's wiring pattern, extended). Cached here
            # so the generate stage reuses this SAME corrected text rather than
            # asking the model again (phase 11 R4a').
            corrected = rewrite_gen.correct(qtext) if args.llm_correct else None
            if corrected is not None:
                qtext = corrected
            if args.rewrites:
                rewrites = rewrite_gen.rewrites(qtext, n=args.rewrites, style=args.rewrite_style)
                hits, _winner, _variants, gate_hits = r.retrieve_fused(
                    qtext, rewrites=rewrites, k=args.k)
            else:
                hits = r.retrieve(qtext, k=args.k)
                gate_hits = hits
            # Expansion changes what the model reads, not how anything ranked, so it
            # belongs here rather than inside the retriever - and `evidence_retrieved`
            # below then means what it says: the answer was in front of the model.
            # It applies only to the fused hits the generator will read; gate_hits
            # is never shown to the model, so it is never expanded.
            hits = expand(db, hits, span=args.expand) if args.expand else hits
            retrieved[row["qid"]] = cache_entry(hits, gate_hits, args.rewrites, corrected=corrected)
            if sys.stdout.isatty():
                print(f"\r  retrieve {i}/{len(rows)}  {(time.time()-t0)/i:.2f}s/q", end="", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(retrieved))
        print(f"\n  cached retrieval for {len(retrieved)} questions in {time.time()-t0:.0f}s "
              f"(mode={args.mode}, rerank={args.rerank})")
        if args.stage == "retrieve":
            print(f"  -> {cache}\n  now: ./scripts/servers.sh stop && "
                  f"./scripts/servers.sh start generator")
            return 0

    # --- stage 2: generation only (generator resident) ---
    gen = Generator(args.gen_url)
    if not gen.health():
        print("generator not running: ./scripts/servers.sh start generator", file=sys.stderr)
        return 2
    if not cache.exists():
        print(f"no cached retrieval at {cache}; run --stage retrieve first", file=sys.stderr)
        return 2
    retrieved = json.loads(cache.read_text())

    results, t0 = [], time.time()
    for i, row in enumerate(rows, 1):
        entry = retrieved[row["qid"]]
        hits, gate_hits = unpack_entry(entry)
        # The gate is architectural: below threshold the model is never invoked, so
        # there is no opportunity to speculate. That is the whole of Finding 04.
        # It reads gate_hits exactly as before select_hits ever runs - never the
        # cut-down list the model actually reads (blk_fusion_gate_semantic_slip).
        score = gate_score(gate_hits)
        gated = bool(args.gate) and score < args.gate
        hits = select_hits(hits, args.read_k, args.cap_per_doc)
        # --llm-correct: reuse the SAME corrected text the retrieve stage cached
        # (correct_text), never re-ask the model here - see the cache-time
        # comment above. Falls back to query_text(row) (normalize, or the raw
        # question) when --llm-correct is off, or for a legacy cache.
        question_corrected = correct_text(entry)
        qtext = question_corrected if question_corrected is not None else query_text(row)
        if gated:
            text = grammar.REFUSAL
        elif args.answer_mode == "quote":
            text = gen.answer(qtext, hits, mode="quote")
        else:
            text = gen.answer(qtext, hits, cite_grammar=args.grammar)

        # Quote mode's verification has to run BEFORE correct/evidence scoring:
        # a claim whose opening quote isn't actually in the extract it cites is
        # converted to the refusal here, so `correct` below scores the refusal
        # (i.e. False) rather than the ungrounded text (phase 10;
        # blk_phase9_parametric_knowledge_failure).
        quote_info, answer_raw = None, None
        if not gated and args.answer_mode == "quote":
            quote_info = grammar.verify_quotes(text, hits)
            if quote_info["quote_failed"]:
                answer_raw = text
                text = grammar.REFUSAL

        rec = {
            "qid": row["qid"], "kind": row["kind"], "tags": row["tags"],
            "variant_kind": row.get("variant_kind"),
            "question": row["question"], "answer": text,
            "abstained": gated or abstained(text),
            "gated": gated,
            "retrieved_docs": [h["doc_id"] for h in hits],
            "top_score": score,
        }
        if answer_raw is not None:
            rec["answer_raw"] = answer_raw
        if quote_info is not None:
            rec.update(quote_info)
        if args.normalize == "spell":
            norm_text, norm_edits = normalized[row["qid"]]
            rec["question_normalized"] = norm_text
            rec["normalize_edits"] = norm_edits
        if args.llm_correct:
            rec["question_corrected"] = qtext
        if row["kind"] == "answerable":
            toks = row["answer_contains"]
            # tsk_20260927_typos: a typo/paraphrase variant (row["qid"] like
            # "a01.y1") has no alias entry of its own; gold.aliases_for
            # falls back to its base's (row["variant_of"]/"paraphrase_of"),
            # a no-op for every qid that already has its own entry.
            row_aliases = gold.aliases_for(
                qid_aliases, row["qid"], row.get("variant_of") or row.get("paraphrase_of"))
            rec["correct"] = gold.is_correct(text, toks, row_aliases)
            rec["correct_strict"] = all(t in text for t in toks)
            aliased_ev, strict_ev = evidence_in(hits, toks, row_aliases)
            rec["evidence_retrieved"] = aliased_ev
            rec["evidence_retrieved_strict"] = strict_ev
            if not gated:
                rec.update(grammar.verify_citations(text, hits, toks))
        results.append(rec)
        if sys.stdout.isatty():
            print(f"\r  {i}/{len(rows)}  {(time.time()-t0)/i:.1f}s/q", end="", flush=True)
    print()

    return write_report(args, results, t0)


def write_report(args, results: list, t0: float) -> int:
    """Score `results` (one record per row - see the per-row loops above and
    `run_cascade_eval` below, both of which build the SAME record shape) and
    write `<name>-answers.json`. Factored out so --cascade's inline loop
    (which has no retrieval cache and cannot share the two-stage code above)
    still ends in exactly this one scoring/report path - not a second,
    possibly-drifting copy of it."""
    ans = [r for r in results if r["kind"] == "answerable"]
    una = [r for r in results if r["kind"] == "unanswerable"]

    correct = sum(1 for r in ans if r["correct"])
    with_ev = [r for r in ans if r["evidence_retrieved"]]
    no_ev = [r for r in ans if not r["evidence_retrieved"]]
    correct_given_ev = sum(1 for r in with_ev if r["correct"])
    wrong_abstain = sum(1 for r in with_ev if r["abstained"])
    right_abstain = sum(1 for r in una if r["abstained"])
    spoke_blind = sum(1 for r in no_ev if not r["abstained"])
    unsupported = (spoke_blind + (len(una) - right_abstain)) / max(len(results), 1)

    print(f"\n=== {args.name} ===  {len(results)} questions, {time.time()-t0:.0f}s"
          f"  (gate={args.gate or 'off'}, grammar={args.grammar})\n")
    print(f"answerable   n={len(ans)}")
    print(f"  answer contains gold token   {correct/len(ans):6.1%}  ({correct}/{len(ans)})")
    print(f"  evidence was retrieved       {len(with_ev)/len(ans):6.1%}")
    print(f"  correct WHEN evidence there  {correct_given_ev/max(len(with_ev),1):6.1%}"
          f"   <- generation quality, retrieval factored out")
    print(f"  false abstention (had ev.)   {wrong_abstain/max(len(with_ev),1):6.1%}  <- coverage lost")
    print(f"\nunanswerable n={len(una)}")
    print(f"  correctly abstained          {right_abstain/max(len(una),1):6.1%}"
          f"  ({right_abstain}/{len(una)})   <- THE core requirement")

    print(f"\nspeaking without evidence")
    print(f"  answered with no evidence    {spoke_blind/max(len(no_ev),1):6.1%}"
          f"  ({spoke_blind}/{len(no_ev)})   <- phase 1's worst number")
    print(f"  unsupported, all questions   {unsupported:6.1%}"
          f"  ({spoke_blind + len(una) - right_abstain}/{len(results)})")
    n_gated = sum(1 for r in results if r["gated"])
    if args.gate:
        print(f"  gate fired                   {n_gated/len(results):6.1%}  ({n_gated}/{len(results)})")

    # Only real answers can carry a citation: a refusal is uncited by construction,
    # and counting refusals as uncited would flatter or damn the grammar at random.
    answered = [r for r in ans if not r["gated"] and not r["abstained"]]
    if answered and "n_citations" in answered[0]:
        uncited = sum(1 for r in answered if r.get("uncited"))
        oor = sum(1 for r in answered if r.get("out_of_range", 0))
        supported = sum(1 for r in answered if r.get("cite_supported"))
        print(f"\ncitations    n={len(answered)} answers the model actually wrote")
        print(f"  no citation at all           {uncited/len(answered):6.1%}")
        print(f"  citation out of range        {oor/len(answered):6.1%}")
        print(f"  cited extract supports it    {supported/max(len(answered),1):6.1%}"
              f"   <- the check the grammar exists to enable")

    by_reason = defaultdict(list)
    for r in una:
        for t in r["tags"]:
            if t in ("tool-not-installed", "out-of-corpus", "requires-execution"):
                by_reason[t].append(r)
    print("\n  abstention by reason         n    rate")
    for t, g in sorted(by_reason.items()):
        print(f"    {t:26} {len(g):3}  {sum(1 for x in g if x['abstained'])/len(g):6.1%}")

    outdir = ROOT / "data" / "eval" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{args.name}-answers.json"
    config = {"mode": args.mode, "rerank": args.rerank, "gate": args.gate,
              "grammar": args.grammar, "candidates": args.candidates,
              "expand": args.expand, "db": args.db, "domain": args.domain,
              "rewrites": args.rewrites, "aliases": not args.no_aliases,
              "cache": args.cache, "read_k": args.read_k,
              "cap_per_doc": args.cap_per_doc, "answer_mode": args.answer_mode,
              "qids": args.qids, "abstain_rule": "all-sentences",
              "evidence_rule": "aliased"}
    # Only added under --normalize spell, so --normalize off's output stays
    # byte-identical to every run made before this flag existed.
    if args.normalize == "spell":
        config["normalize"] = args.normalize
    # Same convention for R4b/R4a': only added when non-default, so every
    # existing run's config (and defaults off) stays byte-for-byte unchanged.
    if args.rewrite_style != "man":
        config["rewrite_style"] = args.rewrite_style
    if args.llm_correct:
        config["llm_correct"] = True
    if args.cascade:
        config["cascade"] = True
    out.write_text(json.dumps({
        "name": args.name, "k": args.k, "n": len(results),
        "config": config,
        "answer_accuracy": correct / max(len(ans), 1),
        "accuracy_given_evidence": correct_given_ev / max(len(with_ev), 1),
        "false_abstention_with_evidence": wrong_abstain / max(len(with_ev), 1),
        "abstention_recall": right_abstain / max(len(una), 1),
        "spoke_without_evidence": spoke_blind / max(len(no_ev), 1),
        "unsupported_rate": unsupported,
        "results": results,
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


def run_cascade_eval(args, rows: list, qid_aliases: dict, normalized: dict,
                      query_text) -> int:
    """--cascade: retrieval and generation run inline, row by row, with the
    embedder, reranker AND generator all resident together - there is no
    retrieval cache (unlike --rewrites, the tier a row needs depends on the
    reader's own answer, so the two stages above cannot be separated in
    time). `smm.cascade.run_cascade` runs the tier 0/1/2 loop itself; this
    function only wires it to the eval's usual per-row record and ends in
    the SAME `write_report` the two-stage path does, so both produce the
    same output shape plus, here, the three cascade_* fields.

    Reranking is always on here, independent of `--rerank` (which the
    two-stage path uses to ablate the cross-encoder): the R4d pre-
    registration's tier 0 IS "R3's configuration" - dense, reranked,
    gated - and every tier's gate/widening depends on a real reranker score
    (docs/phase11-results.md, "R4d pre-registration").
    """
    emb = Embedder()
    if not emb.health():
        print("embedder not running: ./scripts/servers.sh start embedder", file=sys.stderr)
        return 2
    rr = Reranker()
    if not rr.health():
        print("reranker not running: ./scripts/servers.sh start reranker", file=sys.stderr)
        return 2
    gen = Generator(args.gen_url)
    if not gen.health():
        print(f"no generator at {args.gen_url}: "
              f"./scripts/servers.sh start generator", file=sys.stderr)
        return 2
    db = store.connect(ROOT / args.db)
    if args.mode in ("bm25", "hybrid") and not lexical.has_index(db):
        print(f"{args.db} has no chunks_fts", file=sys.stderr)
        return 2
    r = Retriever(db, embedder=emb, reranker=rr, mode=args.mode,
                  candidates=args.candidates, domain=args.domain)

    def read_view(hits: list) -> list:
        h = expand(db, hits, span=args.expand) if args.expand else hits
        return select_hits(h, args.read_k, args.cap_per_doc)

    results, t0 = [], time.time()
    for i, row in enumerate(rows, 1):
        qtext = query_text(row)
        # Same wiring as the two-stage path (--llm-correct composes with
        # --normalize, which query_text() already applied): the cascade
        # itself gets no special treatment, only the resulting text.
        corrected = gen.correct(qtext) if args.llm_correct else None
        if corrected is not None:
            qtext = corrected

        def answer_fn(q: str, raw_hits: list) -> str:
            ph = read_view(raw_hits)
            if args.answer_mode == "quote":
                return gen.answer(q, ph, mode="quote")
            return gen.answer(q, ph, cite_grammar=args.grammar)

        cres = run_cascade(r, gen, qtext, answer_fn, args.gate, k=args.k)
        hits = read_view(cres["hits"])
        gated = cres["gated"]
        score = cres["score"]
        text = cres["answer"]

        # Quote mode's verification has to run BEFORE correct/evidence scoring -
        # see the identical comment in the two-stage loop above.
        quote_info, answer_raw = None, None
        if not gated and args.answer_mode == "quote":
            quote_info = grammar.verify_quotes(text, hits)
            if quote_info["quote_failed"]:
                answer_raw = text
                text = grammar.REFUSAL

        rec = {
            "qid": row["qid"], "kind": row["kind"], "tags": row["tags"],
            "variant_kind": row.get("variant_kind"),
            "question": row["question"], "answer": text,
            "abstained": gated or abstained(text),
            "gated": gated,
            "retrieved_docs": [h["doc_id"] for h in hits],
            "top_score": score,
            "cascade_tier": cres["tier"],
            "cascade_seconds": cres["seconds"],
            "cascade_rewrites": cres["rewrites"],
        }
        if answer_raw is not None:
            rec["answer_raw"] = answer_raw
        if quote_info is not None:
            rec.update(quote_info)
        if args.normalize == "spell":
            norm_text, norm_edits = normalized[row["qid"]]
            rec["question_normalized"] = norm_text
            rec["normalize_edits"] = norm_edits
        if args.llm_correct:
            rec["question_corrected"] = qtext
        if row["kind"] == "answerable":
            toks = row["answer_contains"]
            row_aliases = gold.aliases_for(
                qid_aliases, row["qid"], row.get("variant_of") or row.get("paraphrase_of"))
            rec["correct"] = gold.is_correct(text, toks, row_aliases)
            rec["correct_strict"] = all(t in text for t in toks)
            aliased_ev, strict_ev = evidence_in(hits, toks, row_aliases)
            rec["evidence_retrieved"] = aliased_ev
            rec["evidence_retrieved_strict"] = strict_ev
            if not gated:
                rec.update(grammar.verify_citations(text, hits, toks))
        results.append(rec)
        if sys.stdout.isatty():
            print(f"\r  {i}/{len(rows)}  {(time.time()-t0)/i:.1f}s/q", end="", flush=True)
    print()

    return write_report(args, results, t0)


if __name__ == "__main__":
    raise SystemExit(main())
