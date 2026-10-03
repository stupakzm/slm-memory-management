# Phase 14: a floor quota, tuned on a dev half and judged on a held-out half (R11)

## Why

Phase 13 found:
- **Domain known (oracle):** +40 net answers on the pool.
- **Hard router (R9):** loses it, −31 on Emacs.
- **Equal 25 + 25 quota (R10):** +17 on Emacs, but the man pages pay −3 and abstention −3 (both
  missing their margins).

R10 gives the smaller domain half the pool, even for a question that is plainly about a man page. A
*floor* gives each domain a guaranteed minimum and lets the rest of the pool be filled by relevance:
- **F = 0** is today's open search.
- **F = 25** (two domains, 50 candidates) is R10.

Somewhere between those may keep most of R10's Emacs gain at little man-page cost. But F is a knob, and
choosing it on the rows it is judged on would overfit. So it is chosen on one half of the pool and
judged, once, on the other.

## The floor pool (definition)

For n candidates, D domains and floor F (with D·F ≤ n):
1. Each domain contributes its own top F chunks, from a domain-filtered dense search.
2. The remaining n − D·F slots go to the best remaining chunks by dense distance, across all domains,
   from the open search.
3. Duplicates are dropped, and a slot that a duplicate would have taken is filled in turn from the
   remainder.

The pool of n is then reranked to k = 5 as usual, with the gate unchanged.

## Split

The pool's 160 base questions are shuffled with `random.Random(14)`. The first 80 bases are **dev**;
the other 80 are **test**. Every variant or paraphrase follows its base, found by following
`variant_of`/`paraphrase_of` to the root, so no wording of a test question is seen in dev.

| | Emacs answerable | man answerable | unanswerable |
|---|---|---|---|
| dev | 153 | 257 | 166 |
| test | 171 | 302 | 111 |

## How the sweep is run (and checked before it counts)

A cross-encoder scores each question–passage pair independently of the other candidates in its pool.
So for each question, the union of the open top 50 and each domain's top 25 is reranked **once**. Every
floor's top 5 is then read from those cached scores, with no further reranking. This makes the sweep
one retrieval pass instead of six. The premise is checked before any result is read:

- **Validation:** the simulated F = 0 must match `p13-ctl`'s ordered top 5 on ≥ 95 % of dev rows, and
  the simulated F = 25 must match `p13-quota`'s on ≥ 95 %. Today's retrieval alone reorders about 1
  row in 50. **If either check fails, the sweep stops and is reported as broken. Nothing is selected.**

## Selecting F (dev only)

Grid: F ∈ {0, 5, 10, 15, 20, 25}. Every arm is generated under setting C and scored paired against
F = 0 on dev. The chosen F is the one with the largest dev answerable net (both domains), among those
with dev man net ≥ 0 and dev abstention net ≥ 0. A tie goes to the smaller F. If no F > 0 qualifies,
the phase ends there: there is no test run, and the result is null.

## The decision (test half, run once, chosen F against F = 0)

The chosen F ships as `asq`'s default (a branch for the user to merge, never merged unattended) only if
all four hold on the test half:
1. **Emacs answerable (171):** net ≥ +3 and p < 0.05.
2. **Man answerable (302):** net ≥ −1.
3. **Abstention (111):** net ≥ −1.
4. **Live cost:** the live `--route floor` retrieval path, timed on its own run over the test half,
   averages ≤ 4.08 s/q (phase 13's ceiling: oracle 3.58 + 0.5).

Both test arms (F = 0 and chosen F) come from the same sweep pass, generated under setting C. Rule 4 is
then measured on a separate live run of the chosen F, which also checks that the live path reproduces
the simulated top 5 (reported).

## Known limits, stated now

- The test half is still the same kind of pool (written variants of manual-derived bases). Phase 12's
  realistic set is reported for the chosen F as a second, harder held-out check, but it does not
  decide.
- Half the rows means less power. A real but small effect can fail rule 1 here.

## Validation result (2026-10-03): the sweep is broken, and nothing is selected

The dev sweep (576 rows, floors 0 to 25, one rerank per question over the candidate union) ran. Then the
pre-registered check:

| simulated arm vs real run | ordered top 5 | same top-5 set | same gate | required |
|---|---|---|---|---|
| F = 0 vs `p13-ctl` | 94.1 % | 97.2 % | 99.8 % | ≥ 95 %: **fail** (barely) |
| F = 25 vs `p13-quota` | **56.9 %** | 76.7 % | 99.5 % | ≥ 95 %: **fail** |

As pre-registered, the sweep stops. **No floor is selected, the test half is not touched, and R11 has no
result.**

**Why: reranker scores depend on batch composition.** For chunks present in both runs, the score
difference is:
- **F = 0:** median **0**, p90 0, 1.3 % of pairs over 0.01. The sweep's union starts with the same open
  top 50 in the same order, so the batches of 16 mostly line up.
- **F = 25:** median 5e-4, p90 **0.019**, max 0.119, **16.8 %** of pairs over 0.01. The same 50 chunks
  arrive in a different order inside a larger union, so each pair shares its batch with different
  documents.

So `Reranker.rerank` (batches of 16 per request, `src/smm/rerank.py`) does not score a
question–passage pair independently of its batch-mates on this server. Differences up to about 0.02
reorder near-tied top-5 candidates.

**What this means beyond phase 14:**
- **Any two runs whose candidate pools differ carry this as noise:** phase 13's oracle, R9 and R10
  against the open control, and phase 11's R8. It is noise, not bias: it does not favour either arm.
  (Identical runs share their batches, so their 1-in-50 reorders are a separate server noise.) Those
  comparisons stay paired and their verdicts stand. Margins of a few rows sit inside this noise.
- **A rerank-score cache only works if each pair's score is batch-independent.** Two ways to get that:
  - score pairs one per request (batch = 1, slower; cost unmeasured)
  - fix the order and partners deterministically (e.g. sort candidates by chunk_id before batching), in
    both the live path and the sweep.

  Either changes the live system's scores, so it is a new, pre-registered change with its own
  control.

## Reranker batch-independence: pre-registration (2026-10-03, before any code or run)

**Goal:** a question–passage pair's rerank score must not depend on the other candidates. This is
needed for score caching and sweeps, and it removes a noise source from every paired comparison.

**Suspected causes.** Two mechanisms can couple a pair's score to its batch-mates:
- the client sends 16 documents per request (`Reranker.rerank(batch=16)`)
- the eval's "full" reranker server runs `--parallel 4` (`servers.sh start reranker`), so the server
  evaluates several documents side by side

`asq`'s serve profile runs the reranker with `--parallel 1`.

**Arms**, each on the same 50 dev questions (`random.Random(18)` over the phase 14 dev half):
- **A (today's eval):** server `--parallel 4`, client batch 16.
- **B:** server `--parallel 4`, client batch 1.
- **C:** server `--parallel 1`, client batch 16.
- **D:** server `--parallel 1`, client batch 1.

**Determinism test** (`scripts/rerank_check.py`, tracked). For each question, the same candidate pairs
are scored in two contexts:
- (i) the open top 50 in its own order
- (ii) the union of open top 50 and both domains' top 25, shuffled with a fixed seed

The statistic is max |score(i) − score(ii)| over the shared pairs. An arm is **batch-independent** if
that max is ≤ 1e-4 over all 50 questions.

**Cost:** mean seconds to rerank context (i) per question, per arm.

**Choice:** the fastest batch-independent arm. If none is independent, stop and report.

**Adoption** (a separate full-pool run, only if an arm is chosen). The chosen arm's settings become the
eval default, and the live default if it differs from what `asq` already uses, only if a full-pool run
under them, paired against `p13-ctl`, shows:
1. answerable net ≥ −3
2. abstention net ≥ −1
3. retrieval s/q ≤ 3.25 + 1.0

Otherwise the noise is documented and the defaults stay.
