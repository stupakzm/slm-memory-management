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

## Results (run 2026-10-02/03)

All three arms on the 1,160-row pool: full profile, generation under setting C on :8090 (an unrelated
process holds :8080). Scored with `scripts/screen_report.py --group domain --group-default linux` over
the five pool files, with today's aliases. The regenerated control `p13-ctl` scores 540/883 answerable,
the same total as `p11-pool-r3d`.

| arm | Emacs answerable (324) | man answerable (559) | abstention (277) | evidence (883) | retrieval s/q |
|---|---|---|---|---|---|
| control `p13-ctl` (open) | 202 | 338 | 254 | 711 | 3.25 |
| **oracle** `p13-oracle` | 233: **+31** (37/6), p 1.6e-6 | 347: +9 (16/7), p 0.09 | 256: +2 | 754 | 3.58 |
| **R9** `p13-r9` (dense-vote) | 171: **−31** (8/39), p 5.5e-6 | 345: +7 (16/9), p 0.23 | 252: −2 | 661 | 3.91 |

(gained/lost in brackets; s/q is the retrieve stage's own wall time ÷ 1,160.)

**R9 against the rule:**
1. Emacs net ≥ +5 with p < 0.05: **fail** (−31, significantly *worse*).
2. Man net ≥ −2: pass (+7).
3. Abstention net ≥ −1: **fail** (−2).
4. Router cost ≤ +0.5 s/q against the oracle: pass (+0.33).

**R9 does not ship.** **The oracle passes rules 1 to 3** (+31, p 1.6e-6; +9; +2). As pre-registered, the
verdict is "routing helps, this router doesn't". R9's recovery fraction is −31/+31 = **−1.0**: it
loses on the Emacs rows exactly what the oracle gains.

**Why the router fails (diagnostics):**
- **It is biased toward man pages.** 75 % of the index is man pages, so their chunks win most open top-5
  votes. R9 routed Emacs rows correctly only 281/432 times (65 %): clean 82 %, typo1 85 %, synonym 56 %,
  casual 55 %, no-name 36 %. Man-page rows were 714/728 (98 %).
- **A hard route is unforgiving.** A misrouted Emacs row loses every Emacs passage, while the open control
  kept some. So each routing error is a likely loss (39 lost against 8 gained). The 14 man rows routed
  to Emacs (6 answerable) cost little by comparison.
- A vote over the *reranked* open top-5 would route better, but not well enough to fix this: majority
  318/432 Emacs, top-1 342/432, linux 701 and 675 of 728 (computed from `p13-ctl`'s cached retrieval).
  It would still send about 90 Emacs rows to the wrong domain.

**What the oracle says.** Knowing the domain is worth **+40 net answers on 883** (+4.5 points) and +43
evidence, at +0.33 s/q. It also helps the man pages (+9), because Emacs chunks no longer crowd their
candidate pool. It is several times larger than phase 11's best pool arm (R8: +12 net answers), and it replicates
phase 12's +7 on the realistic set at pool scale.

**Next (not yet pre-registered):**
- **R10, per-domain candidate quotas, no routing decision.** Search each domain separately (e.g. 25 + 25
  candidates) and let the reranker choose among both. The oracle's gain may come mostly from giving each
  domain's passages a fair share of the 50 rerank slots. Under a fixed open pool, a 75 % man-page index
  crowds Emacs out. A quota cannot misroute, so it cannot reproduce R9's −31.
- **Usable today:** `asq --domain emacs` *is* the oracle. When the asker knows the domain, saying so is
  worth this much. A caller that knows its context can pass it (Emacs exports `INSIDE_EMACS` to its
  shells).

Frozen outputs (committed): `p13-{ctl,oracle,r9}-answers.json` (sha256 `cd04ee56…`, `667cb943…`,
`c77a8d09…`) and `p13-r9-routes.json` (`08f28984…`). Noise: setting C generation 0/40. Retrieval
reorders about 1/50 rows between identical runs (drift check above), well inside these margins.

## R10 pre-registration: per-domain candidate quotas (2026-10-03, before any code or run)

**Hypothesis.** The oracle's +40 comes mostly from giving each domain's passages a fair share of the 50
rerank slots. The man pages are 75 % of the index, and an open dense pool crowds Emacs out. If so, the
fix needs no routing decision. Search every domain separately, give each an equal share of the
candidates, and let the reranker choose among all of them. A quota cannot misroute, so it cannot
reproduce R9's −31.

**Arm `p13-quota`:** `--route quota`. For the index's domains (`store.domains`, sorted by name), each
gets `n // D` candidates, with the remainder going one each to the first domains in that order. For 50
candidates and 2 domains that is 25 + 25. Each comes from its own domain-filtered dense search, and the
union of 50 is reranked as usual to k = 5. The gate is unchanged (0.65 on the reranked top-1). No knob
is tuned: an equal split is the only split that does not presume which domain a question belongs to.

**Rule: R9's four, unchanged, paired against `p13-ctl`.** R10 ships as `asq`'s default
(`--route quota`) only if all of these hold:
1. Emacs answerable (324): net ≥ +5 and p < 0.05.
2. Man answerable (559): net ≥ −2.
3. Abstention (277): net ≥ −1.
4. Retrieval s/q ≤ the oracle's 3.58 + 0.5 = 4.08.

Shipping means a code task that changes the default, left on a branch for the user to merge. It is
never merged unattended.

**Reported, not rules:**
- Recovery fraction against the oracle: (R10 net) ÷ (oracle net), per domain.
- The realistic Emacs set (100 rows, phase 12 labels): `p13-quota-real`, paired against open
  `p12-ctl` and the domain-known `p12b-ctl`. This shows whether the effect holds where askers do not
  name Emacs.
- Unanswerable rows: the reranker now sees 25 candidates from a domain the question may have nothing to
  do with. If abstention falls, this is where it shows.

## R10 results (run 2026-10-03, overnight, unattended)

Code: `orch/tsk_20261003_quota` (c4dd9fd). The run used that branch, which is not yet merged. Same pool,
same control `p13-ctl`, same scorer, generation under setting C on :8090.

| | Emacs answerable (324) | man answerable (559) | abstention (277) | evidence | retrieval s/q |
|---|---|---|---|---|---|
| control `p13-ctl` | 202 | 338 | 254 | 711 | 3.25 |
| **R10 quota** `p13-quota` | 219: **+17** (26/9), p 0.006 | 335: **−3** (28/31), p 0.80 | 251: **−3** | 727 | 3.56 |
| oracle (for reference) | 233: +31 | 347: +9 | 256: +2 | 754 | 3.58 |

**Against the rule:**
1. Emacs: pass (+17, p 0.006).
2. Man ≥ −2: **fail** (−3).
3. Abstention ≥ −1: **fail** (−3).
4. ≤ 4.08 s/q: pass (3.56).

**R10 does not ship.** The flag stays opt-in (`--route quota`), and `asq`'s default is unchanged.

**What it shows:**
- **A quota recovers about half of the oracle's Emacs gain (+17 of +31) without a routing decision, and
  without R9's collapse.** It confirms the crowding hypothesis in part: Emacs passages do gain from a
  guaranteed share of the rerank slots.
- **The man pages pay a little for it.** They drop from most of 50 candidates to 25. The −3 is 31 lost
  against 28 gained (p 0.8), so it is churn rather than a clear harm, but it misses a margin set in
  advance, and the rule stands.
- **The abstention changes are mostly that same churn, not invasion.** Of the 9 unanswerable rows whose
  abstention changed (6 newly answered, 3 newly abstained), 8 had a man-page-only top 5. One, `u07.k`
  ("gdb set breakpoint"), got an all-Emacs top 5: the Emacs manual's own GDB interface. That one is the
  pre-registered risk, and on this pool it happened once.
- **Realistic set (reported only):** `p13-quota-real` scores 30/80 against open `p12-ctl` at 28, a net of
  +2 (4/2), evidence 52 against 49. Domain-known `p12b-ctl` scores 35 (+7). Where askers do not name
  Emacs, the quota recovers little. So the pool's +17 leans on Emacs rows whose wording already helps.

**Where this leaves domain routing.** Domain knowledge is worth +40 on the pool. A hard router loses it
(R9: −31), and an equal quota recovers half on Emacs at a small man-page cost (R10). An unequal quota
(e.g. a floor for the smaller domain) adds a tuning knob, and tuning it on this pool would overfit. If
tried, it needs a fresh pre-registration and ideally a held-out set. Explicit `--domain` remains the
only way to the full gain today.

Frozen outputs (committed): `p13-quota-answers.json` (sha256 `610f18cb…`) and `p13-quota-real-answers.json`
(`ae66438e…`).
