# Phase 3 — structure-aware chunking. Reverted.

> Gate: keep it only if it beats phase 2 on the eval set. Revert without sentiment
> if it does not. — the briefing

It does not. Three indexes, about two and a half hours of GPU time, and the answer
is **+8 questions fixed, −8 broken, net zero.** The default pipeline stays flat.

The code stays too, behind flags, because the result is worth being able to
reproduce and because one of the four findings below is a coupling nobody would
have predicted from the outside.

## What was built and why

Phase 1 split its retrieval misses in two and phase 2 closed one half. Of phase 2's
36 remaining misses at k=5, **16 retrieve the right page and rank the answer 6th to
20th, or miss it entirely** — a 1,000-char window slid blindly through
`sshd_config.5`'s 65 KB DESCRIPTION cuts between `PermitRootLogin` and its
description as readily as around it.

The fix needs no model, because groff already marked the boundaries. Every option,
config directive and mount flag is a tagged paragraph: a lone tag line at column 0
followed by an indented block. So the whole parser is one question asked of each
column-0 line — *is the next non-blank line indented?* — resting on the same groff
invariant the section parser already rests on.

Chunk size was held at 1,000 characters so that **the only variable is where the
boundaries fall**. Ceiling re-measured, not inherited: 100%, same as flat
(`scripts/chunk_stats.py`).

| index | chunker | prefix | chunks |
|---|---|---|---|
| phase 2 | flat 1,000-char window | doc identity | 52,401 |
| `phase3-entry` | every section entry-packed | doc identity | 74,197 |
| `phase3-path` | every section entry-packed | + heading trail | 74,197 |
| `phase3-mixed` | entry-packed where tagged, windowed elsewhere | + heading trail | 74,764 |

## Retrieval

| pipeline | r@1 | r@5 | r@10 | r@20 | MRR@10 |
|---|---|---|---|---|---|
| phase 2 flat, dense | 34.9% | 63.5% | 75.4% | 83.3% | 0.471 |
| `phase3-entry`, dense | 34.9% | 50.8% | 63.5% | 74.6% | 0.421 |
| `phase3-path`, dense | 33.3% | 60.3% | 70.6% | 81.0% | 0.436 |
| `phase3-mixed`, dense | 34.1% | 57.9% | 72.2% | 79.4% | 0.443 |
| **phase 2 flat, + rerank** | 43.7% | **71.4%** | 81.0% | **89.7%** | **0.553** |
| `phase3-entry`, + rerank | 43.7% | 69.0% | 77.8% | 84.9% | 0.543 |
| `phase3-path`, + rerank | 42.9% | **72.2%** | 83.3% | 87.3% | 0.553 |
| `phase3-mixed`, + rerank | 38.9% | 71.4% | 82.5% | 87.3% | 0.529 |

Structure-aware chunking is *much* worse under dense retrieval alone and roughly
level once the reranker runs. Most of the dense loss is ranking rather than pool
membership — candidate-pool recall at depth 50 is 90.5% flat against 87.3%
structured — which is exactly the kind of loss a cross-encoder repairs.

### Counted properly, it is a wash

The eval set carries variants and paraphrases, so a single underlying question can
appear four times. Counting surface question ids and counting distinct questions
give different stories, and only one of them is true:

| candidate vs phase 2 | surface | distinct |
|---|---|---|
| `phase3-entry` | +10 −13 = **−3** | +8 −11 = **−3** |
| `phase3-path` | +13 −12 = **+1** | +9 −9 = **0** |
| `phase3-mixed` | +11 −11 = **0** | +8 −8 = **0** |

Fixed: `a07 a11 a25 a27 a35 a37 a40 b12`. Broken: `a04 a15 a19 a20 a33 a34 a39 c01`.

**The fixed set is the target set.** Every one of those eight is a
right-document-wrong-chunk miss from phase 2's failure list — `find` deleting what
it finds, `ls` human-readable sizes, `sed` in-place editing, `mount` read-only,
`systemd` restart-on-crash. The mechanism does precisely what it was designed to do.
It also breaks eight others, and no amount of staring at the aggregate would have
told us that.

### The `concept` collapse is one question

The tag table looked alarming — `concept` −31.2%, five surface questions of 16 — and
it is **one underlying question counted four times**: `c01`, `c01.t`, `c01.z` and
`p09` are the base, terse, misspelled and paraphrased forms of *what is the
difference between `kill` and `kill -9`*. Phase 1's write-up warned about exactly
this arithmetic; it applies here too.

The one question is still worth understanding, because it produced the design's
second rule (below).

## Four findings

**1. The heading trail is worth more than the boundaries.** Putting
`OPTIONS / Local file selection` in the embedded prefix moved dense recall@5 from
50.8% to 60.3% — nine points, far more than the boundary change itself contributed.
Phase 1 established that naming the *document* in every chunk is worth 5.6 points;
this is the same argument one level down, and the extractor already had it parsed.
That result stands regardless of the revert, and is the piece most worth carrying
forward.

**2. Cut on structure only where the document has structure.** The first two indexes
entry-packed every section, which put `signal.7`'s table of signal numbers in a chunk
of its own. Correct as structure, useless as an embedding: a bare column of
`SIGHUP 1 1 1 1` means nothing without the paragraph above it naming the columns. A
tagged option is self-describing; a table is not. `phase3-mixed` entry-packs a
section only when at least half its text sits in tagged paragraphs — 11,441 sections
of the corpus, 27.6 MB — and windows the other 26,884 exactly as phase 1 did. That
recovered part of `c01` and cost a whole index build to learn.

**3. It is not chunk size.** The obvious explanation for the breakage — entries are
too small to embed well — is wrong, and measurably so. The gold chunks of the broken
questions have a median size of 932 characters under structured chunking against 954
under flat. They are the same size. What changed is which neighbours came along.

**4. The chunker and the abstention gate are coupled, in the wrong direction.** This
is the finding that decided the phase, and it was not on anyone's list.

Top-1 reranker score, by population:

| population | phase 2 flat, p50 | phase 3, p50 |
|---|---|---|
| answerable, evidence retrieved | 0.991 | 0.993 |
| answerable, evidence missed | 0.987 | 0.989 |
| **unanswerable** | **0.328** | **0.454** |

A tight, self-contained option entry looks *more* convincing to a cross-encoder than
a ragged 1,000-char window — including when the corpus cannot answer the question at
all. Cleaner chunks made the gate's job harder.

The consequence is decisive. Sweeping end-to-end outcomes at a budget of two points
of accuracy:

| | accuracy | abstention | unsupported |
|---|---|---|---|
| phase 2, ungated | 50.0% | 85.0% | 16.9% |
| phase 2, gate 0.6476 | 48.4% | **92.5%** | **15.1%** |
| phase 3, ungated | **52.4%** | 85.0% | 18.1% |
| phase 3, best gate in budget | *none* | — | — |
| phase 3, gate 0.6216 (equal accuracy) | 48.4% | 87.5% | 17.5% |

Phase 2 buys 7.5 points of abstention recall for 1.6 points of accuracy. **Phase 3
cannot buy any** — there is no threshold that improves abstention while staying
within two points of its own ungated accuracy. Held at equal accuracy, phase 2 gives
92.5% abstention and phase 3 gives 87.5%.

## The decision, and the case against it

Phase 3 has a real argument. **Ungated, it is the more accurate system**: 52.4%
against 50.0%, and it retrieves evidence for 92 questions against 90. If this project
optimised for answering, phase 3 would ship.

It optimises for not answering wrongly. The README's first sentence is that the
defining behaviour is saying *I don't know*, and on that axis phase 3 is worse by
5 points of abstention recall and 2.4 points of unsupported answers, with no gate
setting that recovers either. **Reverted.**

Section expansion — widening each hit to its neighbours within its own section
before the model reads it — is the one piece that helped on its own terms: +2
questions of accuracy and +2 questions of evidence in front of the model. It is
kept in the code (`--expand N`) and is inert on a flat index, which has no sections
to expand into.

## What this means for phase 4

Phase 2 said the gate catches unanswerable questions but not answerable ones whose
retrieval failed, because those two populations have the same reranker score.
Phase 3 says the chunker moves that score too, and moves it the wrong way.

Both point at the same gap: **there is no signal in this system that distinguishes
"the retriever found the answer" from "the retriever found something confident and
wrong."** The retriever's own confidence is not that signal and cannot be made into
one by cutting the corpus differently. A second, independent check — a verifier that
reads the answer against the cited extract, rather than a threshold on the score
that produced it — is the next thing worth building, and the GBNF citation from
phase 2 already makes it computable.

## Reproducing

```bash
.venv/bin/python scripts/chunk_stats.py            # ceiling and cost, no GPU

# the three indexes, ~41 min each
.venv/bin/python scripts/build_index.py --chunker structured --tagged-ratio 0 \
    --out data/index/phase3-entry.db
.venv/bin/python scripts/build_index.py --chunker structured --tagged-ratio 0 \
    --section-path --out data/index/phase3-path.db
.venv/bin/python scripts/build_index.py --chunker structured --section-path \
    --out data/index/phase3-mixed.db

.venv/bin/python scripts/eval_retrieval.py --db data/index/phase3-mixed.db \
    --mode dense --rerank --name phase3-mixed-rerank
.venv/bin/python scripts/compare_runs.py phase2-dense-rerank phase3-mixed-rerank
```

`--tagged-ratio 0` entry-packs every section, which is the rule the first two
indexes used. A later bookkeeping fix in the same function shifts 5 documents of
4,114 by one chunk each; none of them are in the eval set.

---

## Correction (phase 4)

Two defects found while validating the phase 4 tool eval set affect the numbers
above. Neither changes a conclusion; both are documented in
`docs/phase4-results.md`.

1. **`u22` was mislabelled `out-of-corpus`.** `pip.1`, `pip-install.1` and
   `pip3-install.1` are all indexed, and `pip-install.1#USAGE` reads
   `python -m pip install [options] <requirement specifier>`. The question is
   answerable, so abstention recall here was **understated by about two points**:
   phase 3's 85.0% becomes 87.2% and phase 2's 92.5% becomes 94.9%.
   The gap that decided the revert is unchanged.
2. **A corpus extraction bug** deleted tagged-paragraph tags that are
   cross-references (`pip3-install(1)`), across 1,084 of 4,158 pages. Rebuilding the
   index on the corrected corpus moved recall@5 by less than one question.
