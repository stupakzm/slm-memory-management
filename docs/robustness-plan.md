# Robustness plan: the same question, asked the way people actually ask it

Written 2026-09-27, before any of it was run.

## The goal, in the user's words

"Fast and accurate for any human request." Here that means: a function named in other
words, with synonyms, tersely, or with typos ("mostake" for "mistake") must still land
on the right page and get the right answer. It does **not** widen scope. A question the
docs cannot answer is still refused; that stays the project's core promise.

## Where things stand

- Robustness has never been measured at a useful size. The 166-question set has 20 typo
  and 20 terse variants. Phase 1 measured dense recall on typo variants at 60.0% → 53.3%
  (n=15), and nothing since.
- The shipped 4B, given the right page, answers correctly 84% of the time on man pages.
  Retrieval gains alone have never moved correctness (phases 6-8). So the question for
  every variation kind is *where* it breaks: at retrieval (wrong page), or at reading.
- First Emacs eval (2026-09-27): 60/60 evidence retrieved, 35/60 strict correct, about 49-51/60
  once a documented key binding counts as naming its command. That scoring gap is fixed first,
  because every later number depends on it.

## Parts, in order

Each code change runs as an orchestrated task (worktree, code-verified acceptance, registry)
and is merged when it passes. Each measurement states its decision rule before the run.

### R1. Fix Emacs scoring: key ⇄ command aliases
Derive, never assert: for each Emacs question, find the ‘KEY’ (‘command’) pairs in its gold
section and add the key as an alias of the command token (and the reverse for key questions)
in `data/eval/gold_aliases.json`. Rescore the two phase 11 arms offline (no GPU).
*Done when:* the aliases are derivable by a tracked script, re-derivation is byte-identical,
and the rescored arms are committed.

### R2. A variation pool: many wordings per answer
Every clean question (72 man-page + 44 Emacs unique answerable, plus the unanswerable ones,
because a typo must not break abstention either) gets six variants:

| kind | how it is made | example |
|---|---|---|
| `typo1` | generated: one realistic typo (keyboard-adjacent swap, drop, double, transpose) | "mostake" |
| `typo3` | generated: two to three typos | |
| `synonym` | written: different vocabulary for the same need | "wipe" for "delete" |
| `casual` | written: wordy, conversational | "ugh how do i ..." |
| `terse` | written: keywords only | "tar exclude list file" |
| `no-name` | written: the need, without naming the tool | "archive but skip some files" |

The typos come from a seeded, deterministic generator (`scripts/make_variants.py`) that never
touches the answer token. The written variants are authored against the gold section and run
through `resolve_gold.py` (tokens inherited, leak check on). Target is about 1,000 rows, in
`data/eval/variations.jsonl`.
- R2a: the typo generator plus tests.
- R2b: the written variants, man pages.
- R2c: the written variants, Emacs.

### R3. Baseline robustness, per kind
Shipped config on `phase11.db` with no domain filter (the real product). For each variant
kind: evidence@5, correct (aliased), abstention on unanswerable variants, and the loss versus
each variant's own clean question, paired per base. This produces the map of what breaks, and
where. Phase 11 write-up part 1.

### R4. Query-side normalisation that the 4B budget can afford
Pre-registered after R3, with the decision rule set from R3's gaps. Three arms plus a control:
- **a. vocabulary spell-correct.** A vocabulary of the corpus's own words and identifiers
  (commands, flags, page names). Out-of-vocabulary query words are snapped to the nearest
  in-vocabulary word within edit distance 2, weighting keyboard-adjacent substitutions.
  No model, microseconds per query.
- **b. 4B rewrite.** "Fix spelling, restate as a precise documentation query", fused with the
  original via the existing `--rewrites` path. About 1 s.
- **c. a + b.**
Each arm is measured on the full pool **and** on the clean questions. A normaliser that
helps typos but costs clean answers does not ship.

### R5. Speed
Rerank 20 candidates versus 50, on the pool. Report p50 and p95 latency per question,
end to end. Ship the cheapest setting that loses nothing measurable.

### R6. Grounding
Quote mode (`--answer-mode quote`, already built) versus cite on the full pool with the 4B.
It targets the reader side of the gap that R3 exposes.

### R7. Write-up
`docs/phase11-results.md`, a README status row, and memory. Null results stated as null.

## Costs (see memory: compute budget)

The pool is about 1,000 questions. One arm takes about 17 min of retrieval plus about 12 min
of generation, and the arms share one cached retrieval where only generation changes. R3-R6
come to about 3-4 h of GPU in total. None of it needs an index rebuild.

## Authority

The user authorised running this part by part, including commits and merges, without asking
(2026-09-27). Not covered: pushing to the remote, and ORCH's mandatory checkpoints (those
still stop and ask).

## Status 2026-09-29: plan complete

Every item below is done or closed. Results and verdicts are in `docs/phase11-results.md` ("Phase 11
closing"). Shipped: `asq --rewrites 0`. Not shipped, with their measured reasons: R8 (question
vectors; +12 net, no side cost, misses significance by one row and latency by 0.28 s), R6 (quote;
−258), R5b (adaptive depth; −4), R4a″ (English-only spelling; renames 4/160). Found: deterministic
eval serving (setting C). Closed without a run: the term-menu arm (dominated on latency by R8).
Natural next arm: a cheaper R8 (smaller question route, or replace rather than add candidates) on a
pool with more target rows, generated under setting C.

## Next session: task list (written 2026-09-29, after R4a-R5, R4d and the term-menu check)

Where things stand: every R4/R5 arm is run and written in `docs/phase11-results.md` (summary table
"Phase 11 R4 summary"). None ships. The term-menu go/no-go passed on its held-out half. Ranked by
value for cost:

1. **Switch `asq` to `--rewrites 0`.** *Done 2026-09-29* (user said yes; tsk_20260929_rewrites0,
   branch `orch/tsk_20260929_rewrites0`, awaiting merge). `--rewrites` now defaults to 0 in every
   mode, with a stubbed test (`tests/test_ask_defaults.py`). Measured in R4b: the old default lost
   47/600 answers against plain retrieval and was about 1 s slower. Correction: no registry check
   pinned that line (the cascade checks cover `tests/test_cascade.py` only), so nothing was
   superseded.
2. **Document expansion: plain-English questions as extra index vectors.** The term-menu check showed
   4B-written user-style questions bridge the vocabulary gap (held-out 13/22 = 59 % vs 45 % for
   manual-text cards). Simplified into the main index: per chunk, 2-3 generated "how would a user ask
   for this" questions, stored as **extra vectors pointing at that chunk**. The reranker still scores
   the real chunk text against the original question, and the gate is unchanged. No query-time
   generation, no rewrite shown to anything. Emacs first: ~19k chunks × 0.31 s ≈ 1.6 h generation +
   ~30 min embedding, additive, no full rebuild. Pre-register (target: no-name + synonym; guards:
   clean unharmed, abstention, latency), then build via orch-task.
   *In progress 2026-09-29:* pre-registered as **R8** (`fc6ae0a`, docs/phase11-results.md; 3
   questions per chunk, M = 30 question vectors widen the pool). Built as tsk_20260929_qvec
   (branch `orch/tsk_20260929_qvec`, awaiting merge; `scripts/build_qvec.py`,
   `--question-vectors`). Generation is running into `data/index/qvec-emacs.json` (0.49 s/chunk,
   about 2.5 h). Then: embed into `phase11-qx.db`, pool retrieval, and re-generation only where
   the reader's input changed.
3. **Typo correction that can't rename things.** R4a/R4a′ recovered ~28 typo answers but renamed
   names. New rule: correct a word only into an **ordinary English word** (an English wordlist), never
   into a technical term: `mostake→mistake` yes, `nmap→mmap` and `elpy→elpa` no. CPU only.
   Pre-register with R4a's four rules.
4. **Deterministic eval serving.** 1/40 answers reworded on identical input, and 9 tier-0
   answer/refuse flips in R4d. Run the eval generator with one slot (`--parallel 1`) so paired tests
   lose that noise floor. Check it with a repeat run.
5. **R6 quote mode**: pre-registered (`234b173`), not run. Generation only, from R3's cache,
   ~25 min. The natural guard for anything in item 2 that brings new pages to the reader.
6. **Adaptive rerank depth for speed**: 20 candidates first, 50 only when the top score is weak.
   R5: 20 is 58 % faster and loses answers only where it is unsure (no-name).
7. **Term menu as an arm** (only if item 2 underdelivers): menu from both card types, the 4B picks
   ≤ 3 with grammar-restricted choices, widen the pool, rerank against the original.
8. **R7 write-up**: README status row and phase 11 closing section.

Do not: rewrite every question with the 4B (R4b), use a stricter gate on retries (R4d scores don't
separate), use a bigger reranker, or use a bigger reader (the 30B answers from its own knowledge).

Artifacts from 2026-09-28/29 (gitignored, local): `data/index/phase11-scratch/`. The pool file is
`cat data/eval/{questions,emacs_questions,variations_typo,variations_man,variations_emacs}.jsonl`.
- `cards_plain.json`: 7,065 Emacs term cards with 2 plain-English questions each (36 min GPU)
- `termmenu_split.json`: frozen dev/test split of the 62 Emacs bases (seed 20260928); test half
  used once, by v3
- `termmenu_*.py`: the go/no-go scripts
- `r4b_compare.py`: the paired arm-vs-control table used for every R4/R5 result
- `r4b-qids.json`, `r4d-qids.json`, `r4a2-prepass.json`: the arms' row lists and pre-pass
