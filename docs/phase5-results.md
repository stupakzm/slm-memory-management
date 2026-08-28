# Phase 5 — user-fed knowledge, and corpus identity

> Gate: adding a second unrelated domain does not regress the Linux eval numbers.

**Passed exactly.** With namespacing, an index carrying 38,938 chunks of user-fed
material — 41% of its contents — returns bit-identical Linux numbers to the
single-domain baseline: recall@1/3/5/10/20 of 44.1 / 62.2 / 70.9 / 80.3 / 88.2 and
MRR 0.557, every figure unchanged.

The more useful result is that the premise behind the gate turned out to be wrong in
one direction and right in another, and neither could have been guessed from the
sizes involved.

## What was built

| | |
|---|---|
| Ingest | file, directory or URL → documents in the same shape a man page has |
| Chunking | unchanged — the flat phase 1/2 chunker, because a user document is already a document |
| Namespace | a `domain` column on `chunks`; retrieval scopes to one, or lets all compete |
| Index | one index, not one per domain |

Making user material look like a man page — an id, a summary, an ordered list of
sections — is what let phases 1 through 4 be reused whole. The reranker, the
abstention gate, the citation grammar and the tool layer all work on ingested text
without knowing it is not a manual page.

**One index, not one per domain, is a deliberate choice**: with separate indexes the
dilution this phase exists to measure cannot happen, so it could not be measured
either. The namespace is a filter over a shared index, which means "what does
namespacing buy" is answerable rather than assumed.

`src/smm/ingest.py` is honest about its weak point. Man pages gave up their structure
for free because groff guarantees body text is indented; a pasted note guarantees
nothing. The splitter tries Markdown headings, falls back to packing paragraphs into
~2,000-character sections, and falls back again to one section. Tier two is a stable
unit, not authored structure — and phase 3 measured what unreliable boundaries cost.
User material is retrieved on the same footing as man pages but it is not cut as
well, and no amount of pipeline reuse changes that.

## Two domains, chosen to make the test able to fail

| domain | source | documents | chunks | share of index |
|---|---|---|---|---|
| `linux` | man pages 1/5/7/8 | 4,147 | 56,567 | 59.2% |
| `literature` | 39 public-domain books from Project Gutenberg | 39 | 32,708 | 34.2% |
| `userdocs` | package READMEs + GNOME end-user help, all `/C/` | 1,602 | 6,230 | 6.5% |

`literature` is the briefing's own example of an unrelated domain, and it is large —
27 MB, a third of the index. `userdocs` was added afterwards because the literature
result made it clear the specified experiment **could not fail**: novels and manual
pages are too far apart to compete. `userdocs` is documentation for the *same
software the man pages describe*, written for end users, and that is where confusion
is actually plausible.

## Dilution tracks proximity, not volume

Across 166 Linux questions and 8,300 candidate slots:

| domain | share of index | share of candidate pool | share of top-5 | questions it reached |
|---|---|---|---|---|
| `linux` | 59.2% | 95.7% | 92.8% | 166/166 |
| `literature` | **34.2%** | **0.0%** | **0.0%** | **0/166** |
| `userdocs` | 6.5% | 4.3% | **7.2%** | 90/166 |

A third of the index is invisible. Not rare — **zero**, in every one of 8,300 slots.
Unscoped retrieval over the two-domain index returned numbers identical to the
scoped run, digit for digit, including the top-1 score distributions the abstention
gate reads. The embedder separates Linux documentation from English novels without
being told the boundary exists.

Meanwhile `userdocs`, at a fifth of literature's size, takes a *larger* share of the
top 5 than of the index and reaches 90 of 166 questions. Its cost is small but real
and consistent — −0.8% at every k, MRR −0.006, exactly one question broken at k=5:

| | recall@5 | recall@20 | MRR@10 |
|---|---|---|---|
| baseline, one domain | 70.9% | 88.2% | 0.557 |
| + literature, unscoped | 70.9% | 88.2% | 0.557 |
| + literature + userdocs, unscoped | 70.1% | 87.4% | 0.551 |
| + both, **scoped to `linux`** | **70.9%** | **88.2%** | **0.557** |

**So the briefing's claim — that vector search degrades measurably when unrelated
domains share an index — is not what happens.** Unrelated domains do not compete at
all. *Adjacent* domains do, at a rate set by how close they are rather than how much
of the index they occupy. Namespacing is worth having, but the case for it is not the
one that motivated it.

## The domain that needs protecting is the new one

Namespacing is naturally argued for as protecting the established corpus from
newcomers. On this eval it is the other way round.

Sixteen literature questions, verified against the fetched texts the way
`resolve_gold.py` verifies phase 0 — two candidates were dropped because those
editions phrase the line differently:

| | recall@1 | recall@5 | MRR@10 | gold doc in top 5 |
|---|---|---|---|---|
| scoped to `literature` | 62.5% | 68.8% | 0.662 | 100% |
| unscoped | 56.2% | 62.5% | 0.594 | 100% |

Linux questions lose nothing to literature; literature questions lose to Linux. The
mechanism is plausible — man pages contain ordinary English prose in DESCRIPTION,
so they can half-match a question about a novel, while nothing in a novel half-matches
a flag lookup — but it is one question out of sixteen and the honest reading is
directional only. It is enough to say the asymmetry runs the opposite way to the
intuition, and not enough to say how large it is.

That the new domain is retrievable at all matters for the gate: namespacing that
buried `literature` would have passed the "no regression" test trivially by making
the ingested material unreachable.

## The gate threshold does transfer across domains

Phase 4 found that the abstention threshold swept on documentation *questions* was
badly wrong for tool *requests* — 0.65 against 0.30, an order of magnitude apart in
the score's own terms. The obvious worry for phase 5 is that a newly ingested domain
inherits a threshold fitted to man pages.

Measured, it mostly does not matter:

| answerable questions | n | p10 | p50 | p90 | refused at 0.65 |
|---|---|---|---|---|---|
| `linux` | 127 | 0.802 | 0.991 | 0.999 | 6 (4.7%) |
| `literature` | 16 | 0.688 | 0.958 | 0.998 | 1 (6.2%) |

The ingested domain scores a little lower and loses a little more coverage, but the
distributions overlap heavily and the threshold is serviceable on both. So phase 4's
lesson wants narrowing: a tuned threshold is a property of the **task** — what a
question looks like and what counts as an answer — much more than of the **corpus**
it is asked against. Changing from questions to imperatives moved the right threshold
by a factor of two; changing from manual pages to novels barely moved it.

## A-MEM-style linking: not built, on purpose

The briefing lists it as belonging here "if anywhere". It is not here.

The project's founding argument is that sub-7B models cannot build a knowledge graph
and fail hard rather than gracefully, and that man pages are tractable *because*
`SEE ALSO` is a curated edge list authored by humans. User-fed material arrives with
no such edge list. Building one with the 4B model is precisely the load-bearing
failure this design was chosen to avoid — and phase 3 already showed that a structure
which looks right can quietly make the abstention gate worse.

If it is attempted later, it needs a metric it can fail against first. There is no
question in any eval set here whose answer depends on a link between two user notes,
so a linking layer could be added today and nothing would say whether it helped.

## Corpus identity

Phase 4 found that re-extracting picked up 33 `cmake-*` pages installed since August,
and separately that a one-line extractor change altered 1,084 documents. Neither
surfaced as an error. An index quietly built from a different corpus than the eval it
is scored against invalidates a comparison without ever failing.

`src/smm/fingerprint.py` gives a corpus a digest over three things that can each move
a number: per-document content, the extractor's own source, and the document set. The
digest is written into index metadata at build time and checked at eval time, so a
mismatch prints a warning instead of producing a quietly incomparable number.

```
corpus     b11bbf0145f18aed  4147 docs, 42105 sections, 44.51 MB, extractor 52432dc2baa117b0
```

```bash
.venv/bin/python scripts/corpus_fingerprint.py --write   # record
.venv/bin/python scripts/corpus_fingerprint.py --check   # has the machine moved?
```

`--check` exits non-zero and names what was added, removed or changed. Indexes built
before fingerprinting say so rather than claiming a match — an unverifiable
provenance is a weaker claim than a matching one, not an equal one.

It deliberately does not cover chunking or embedding: those belong to an index, and
two indexes over one corpus should agree on corpus identity.

## Two bugs the ingest path found

**Colliding document ids.** `slug()` truncates at 48 characters, and four of 1,602
GNOME help pages collided —
`help-evolution-mail-account-manage-microsoft-exc` three ways. The collision
surfaced only as a `UNIQUE constraint failed` deep in the vector store, which is a
bad way to learn about it. `ingest.dedupe_ids` now disambiguates with a short digest
of the *source*, so an id does not depend on the order a batch was read in.

**`vec0` does not honour `INSERT OR REPLACE`.** It raises a uniqueness violation
instead of replacing, so re-inserting an existing rowid failed. The vector is now
deleted before insertion, which is what makes an interrupted ingest safe to re-run —
and both ingests in this phase were interrupted and resumed.

## Reproducing

```bash
.venv/bin/python scripts/fetch_domain.py --domain literature      # ~27 MB, public domain
./scripts/collect_userdocs.sh                                    # the adjacent domain
cp data/index/phase4.db data/index/phase5.db
./scripts/servers.sh start embedder
.venv/bin/python scripts/ingest.py --db data/index/phase5.db --domain literature \
    data/corpus/domains/literature/
.venv/bin/python scripts/ingest.py --db data/index/phase5.db --domain userdocs \
    data/corpus/domains/userdocs/

./scripts/servers.sh start reranker
.venv/bin/python scripts/eval_retrieval.py --db data/index/phase5.db --domain linux \
    --mode dense --rerank --name phase5-3domain-scoped
.venv/bin/python scripts/eval_retrieval.py --db data/index/phase5.db \
    --mode dense --rerank --name phase5-3domain-unscoped
.venv/bin/python scripts/compare_runs.py phase4-retrieval phase5-3domain-scoped
```

| run | index | scope |
|---|---|---|
| `phase5-scoped` / `phase5-unscoped` | linux + literature | identical to each other |
| `phase5-3domain-scoped` | all three | identical to the phase 4 baseline |
| `phase5-3domain-unscoped` | all three | −0.8% at every k |
| `phase5-literature` / `-unscoped` | all three, literature questions | 68.8% → 62.5% |
