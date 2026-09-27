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
