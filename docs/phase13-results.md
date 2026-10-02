# Phase 13: domain routing (R9)

## Why

Phase 12's re-screen (docs/phase12-results.md) found that telling retrieval the domain was the largest
effect seen so far. On 80 realistic Emacs questions, `--domain emacs` alone gained +7 net answers (9
gained, 2 lost, p = 0.065) and +13 evidence over open-domain retrieval. Question vectors added +1 and
+0 on top. But on that set, `--domain` is an oracle: every row is an Emacs question. A real asker's
domain has to be inferred, and every man-page question routed to Emacs by mistake is a likely loss.
This phase measures how much of that gain a router recovers on the full pool, which mixes both
domains, and what it costs the man-page rows.

## Arms (all 1,160 pool rows, dense, reranked, gate 0.65, k 5, generation under setting C)

- **control:** `p11-pool-r3d`, R3's open-domain configuration (no domain filter), already generated
  under setting C.
- **oracle (ceiling):** `--route oracle`. Each row is retrieved within its own domain: `emacs` for Emacs
  rows, `linux` for man-page rows (`domain: null` in the eval files). This is what an asker gets by
  passing `--domain` themselves, or what a caller that knows its context (e.g. a shell inside Emacs)
  could pass. It is not a router and is never a default.
- **R9 dense-vote (the candidate):** `--route dense-vote`. One query embedding, an open dense search,
  and a majority vote over the domains of its top 5 chunks. Then the 50 candidates are searched within
  the winning domain and reranked as usual. A tie goes to the domain of the top-ranked chunk. There are
  no tunable knobs: 5 is the reader's own k, and the vote is a plain majority. The extra cost is one
  vector search and no extra reranking.

## Decision rule (R9 ships as `asq`'s default `--route dense-vote` only if all four hold)

Paired per row against the control, with aliased correctness from stored answer text (today's
`gold_aliases.json`):
1. **Recovers Emacs answers:** Emacs answerable rows (324): net ≥ +5 **and** exact two-sided McNemar
   (sign test on discordant pairs) p < 0.05.
2. **Man pages unharmed:** man-page answerable rows (559): net ≥ −2.
3. **No new invention:** over all 277 unanswerable rows, abstention net ≥ −1 (it must not fall by 2 or
   more).
4. **Affordable:** mean retrieval wall time per question, from the retrieve stage's own log line, rises
   by ≤ 0.5 s against the oracle run on the same day (the oracle does the same single rerank, so the
   difference is the router's own cost).

The oracle arm is held to rules 1 to 3 as well, and reported. If R9 fails rule 1 and the oracle passes
it, the result is "routing helps, this router doesn't". The next step is then a stronger router
(reranked vote, or a model classifier) under its own pre-registration. If the oracle fails rule 1 too,
domain routing is not worth pursuing on this pool, and phase 12's +7 was specific to the realistic set.

## Control drift check (before the arms run)

The control's retrieval was cached in phase 11 under older code (the reranker overflow fix landed
since). Open-domain retrieval is re-run today on 50 pool rows (seed 13). If more than 2 of the 50 differ
in top-5 chunk ids or in the gate decision, the control is regenerated in full (`p13-ctl`, open domain,
setting C) and used in place of `p11-pool-r3d`.

## Diagnostics (reported, not rules)

- Router accuracy: R9's routed domain against the row's true domain, by domain and by kind (clean,
  typo, synonym, casual, terse, no-name, paraphrase).
- R9's recovery fraction: R9 net ÷ oracle net on the Emacs answerable rows.
- Losses, by qid, for every man-page row R9 routed to Emacs.

## Known limits, stated now

- The pool's Emacs variants were written against the gold sections and often name Emacs. That makes
  routing easier here than for realistic askers, so a pass is an upper estimate for unnamed questions.
- One pre-registered router is tested. If it fails, a second router is a new experiment, not a retune.

## Control drift check (run 2026-10-02, before any arm)

The pool file is the five eval files concatenated in phase 11's order (1,160 rows; sha256 prefix
`a5da3af88c1c6d94`). The sample is `random.Random(13).sample(pool qids, 50)`. Open-domain retrieval was
run today and compared with `p11-pool-retrieved.json`: **3 of 50 rows differ** in their ordered top-5
chunk ids (a30, a29.y3, eu05.y3), with no gate decision changed. That exceeds the limit of 2, so **the control is regenerated in
full as `p13-ctl`**, as pre-registered.

The same 50 rows were then retrieved a second time today: 1 of 50 differs (a30, a near-tie reorder), and
the largest top-1 score difference is 0.00027. Retrieval therefore has a small noise floor of its own,
about 2 % of rows reordering between identical runs, separate from the generator's (setting C:
0/40). A paired net within ±2 on 500+ rows is inside that floor.
