# Phase 1 — the deliberately boring baseline

Flat fixed-size chunks, dense retrieval only, no tree, no graph, no reranker, no
abstention gate. Its job is to be the number every later phase must beat.

## Setup

| | |
|---|---|
| Corpus | 4,114 man pages, sections 1/5/7/8 |
| Chunking | 1,000 chars, 150 overlap, structure-blind, doc-identity prefix |
| Chunks | 52,401 |
| Embedder | Qwen3-Embedding-0.6B Q8_0, 1024-dim, last-token pooling |
| Store | sqlite-vec, single 291 MB file |
| Index build | 42 min, 21 chunks/s, GPU saturated at 89% |
| Query latency | 99 ms retrieval, 1.1 s generation |
| Eval | 166 questions, 126 answerable, 40 unanswerable |

**Chunking ceiling: 100%.** Every answerable question has its complete answer
inside at least one chunk, so nothing below is a chunking artefact.

## Retrieval

```
recall@1  34.9%   recall@3  55.6%   recall@5  63.5%   recall@10 75.4%   recall@20 83.3%
MRR@10    0.471   gold document in top 5: 67.5%
```

| tag | n | r@5 | r@20 |
|---|---|---|---|
| flag lookup | 81 | **56.8%** | 81.5% |
| exact-token | 39 | 69.2% | 82.1% |
| config (man5) | 27 | 77.8% | 85.2% |
| concept (man7) | 16 | 68.8% | 87.5% |

Flag lookup is the core use case and the **worst** category. Flag descriptions are
two lines inside option dumps that run to 240 KB.

## Query-noise robustness

| variant | base r@5 | variant r@5 | |
|---|---|---|---|
| terse | 60.0% | **86.7%** | *improves* |
| typo | 60.0% | 53.3% | mild loss |
| no-tool | 60.0% | **33.3%** | collapses |

Terse queries doing *better* is not a fluke: "tar exclude list of files" names the
tool and carries no filler to dilute the embedding. Dropping the tool name is what
hurts, and it is the case both BM25 and the `SEE ALSO` graph are positioned to fix.

Typo degradation is mild for dense retrieval — which matters for phase 2, because
BM25 would be devastated by the same queries. Dense is the typo-robust half of the
hybrid; the lexical half needs fuzzy or trigram matching, not just stemming.

## Where retrieval actually fails

Of 46 misses at k=5:

**59% went to the wrong document** — and plausibly so:

| question | wanted | got |
|---|---|---|
| search every file underneath a folder | `grep.1` | `find.1` |
| list files changed within the last week | `find.1` | `git-whatchanged.1` |
| sizes print as raw bytes | `ls.1` | `ffmpeg-utils.1` |
| make a service start at boot | `systemctl.1` | `daemon.7` |
| look for a string across a source tree | `grep.1` | `git-grep.1` |

**41% found the right document but the wrong chunk:**

| question | doc rank | answer rank |
|---|---|---|
| change permissions on a folder and everything inside | `chmod.1` @2 | @11 |
| tell ssh to use one particular private key | `ssh.1` @1 | @13 |
| sudo without typing a password | `sudoers.5` @1 | @12 |
| signals a program cannot intercept | `signal.7` @1 | @17 |
| mount so nothing can write to it | `mount.8` @1 | not in top 20 |

This split is the most useful thing phase 1 produced, because the two halves have
different fixes and both are now *measured* rather than assumed:

- wrong-document → reranker and BM25 (phase 2)
- right-document-wrong-chunk → **structure-aware chunking (phase 3)**, one chunk per
  option entry instead of a blind 1,000-char window

## Abstention: what a raw score gate buys

Sweeping a threshold on the top-1 dense score:

| threshold | abstention recall | coverage kept |
|---|---|---|
| 0.550 | 37.5% | 96.0% |
| **0.570** | **62.5%** | **90.5%** |
| 0.590 | 77.5% | 80.2% |
| 0.630 | 90.0% | 41.3% |

**Best gate holding ≥90% coverage: 62.5% abstention recall.** 37.5% of unanswerable
questions score above the answerable 10th percentile, so the two populations overlap
badly. This is the concrete reason Finding 04 wants the *reranker* score at the gate
rather than the raw dense score, and it is the number phase 2 has to beat.

## Generation and abstention (prompt-only)

Phase 1 asks for abstention in the system prompt and enforces nothing. The research
says that fails at 4B. It does — but not uniformly, and the shape matters.

```
answerable   n=126        answer contains gold token   47.6%
                          evidence was retrieved       63.5%
                          correct when evidence there  71.2%
unanswerable n= 40        correctly abstained          85.0%
```

Abstention by reason, on the unanswerable set:

| reason | n | abstained |
|---|---|---|
| tool-not-installed | 26 | **100.0%** |
| requires-execution | 8 | 75.0% |
| out-of-corpus | 6 | **33.3%** |

**Prompt-only abstention works perfectly where the retrieved extracts are visibly
about something else** (asked about docker, handed man pages for other tools, the
model says it doesn't know). **It fails where the model's own parametric knowledge
is strong** — every miss is a C library or pip question it simply knew the answer to.

And it fabricates citations while doing it:

> *What does malloc return when the allocation fails?* → **"NULL [3]"**
> *How do I install Python packages with pip?* → **"python -m pip install ... [3]"**

Extract [3] says nothing of the kind. Asking for citations in a prompt produces
citation-shaped text, not citations. This is the concrete argument for phase 2's
GBNF-enforced chunk-ID citation: an uncited claim has to be structurally impossible,
not discouraged.

### The number that actually matters

Splitting the answerable set by whether evidence was retrieved at all:

| | n | |
|---|---|---|
| evidence retrieved | 80 | |
|   → answered correctly | 57 | 71.2% |
|   → abstained anyway | 5 | 6.2% — genuine false abstention, and low |
| evidence **not** retrieved | 46 | |
|   → abstained, correct call | 20 | 43.5% |
|   → **answered anyway** | **26** | **56.5%** |

The headline 85% abstention recall flatters the system. When retrieval simply fails
to find the answer, the model invents one **56.5% of the time**. Counting both
sources, it speaks without evidence on **32 of 166 questions (19%)**.

Meanwhile genuine false abstention — refusing when the evidence was right there — is
only 6.2%. So the model is not too cautious. It is not cautious enough, and only in
the specific case where retrieval quietly returned nothing useful.

## Phase 1 baseline, for phase 2 to beat

| metric | phase 1 |
|---|---|
| retrieval recall@5 | 63.5% |
| MRR@10 | 0.471 |
| answer accuracy | 47.6% |
| accuracy given evidence | 71.2% |
| abstention recall (unanswerable) | 85.0% |
| unsupported answers (all questions) | 19.3% |
| abstention recall at ≥90% coverage, score gate | 62.5% |

## What phase 2 must do, in priority order

1. **Reranker over top-50 → keep 3–5.** Addresses both failure halves: near-miss
   documents and buried option chunks. Also supplies a sharper score for the gate.
2. **Score gate before generation.** The 56.5% invent-an-answer rate is the single
   biggest defect, and it happens exactly where a gate applies.
3. **BM25 fused with dense.** For the 39 `exact-token` questions and the `no-tool`
   collapse. Needs typo tolerance — dense handled misspellings fine, BM25 will not.
4. **GBNF-enforced citations.** Fabricated `[3]` markers prove prompting is not enough.

## Ablation: does the document-identity prefix help?

Every chunk in the main index is prefixed with `tar(1) - an archiving utility`, on
the theory that a window from the middle of a long page otherwise never says which
command it documents. That was an assumption, so it got its own 39-minute index
build and its own eval.

| metric | no prefix | prefix | delta |
|---|---|---|---|
| recall@1 | 33.3% | 34.9% | +1.6% |
| recall@3 | 50.0% | 55.6% | +5.6% |
| **recall@5** | **57.9%** | **63.5%** | **+5.6%** |
| recall@10 | 64.3% | 75.4% | +11.1% |
| recall@20 | 78.6% | 83.3% | +4.8% |
| MRR@10 | 0.432 | 0.471 | +0.039 |

At k=5: **14 questions fixed, 7 broken, net +7.** Not a uniform improvement, which
is why the comparison is reported per question rather than as a single number.

Per tag, in raw question counts rather than percentages — the distinction matters:

| tag | n | no prefix → prefix | |
|---|---|---|---|
| flag lookup | 81 | 40 → 46 | **+6** |
| exact-token | 39 | 23 → 27 | **+4** |
| variant-terse | 15 | 9 → 13 | **+4** |
| config | 27 | 20 → 21 | +1 |
| variant-typo | 15 | 8 → 8 | 0 |
| concept | 16 | 12 → 11 | −1 |
| variant-no-tool | 15 | 6 → 5 | −1 |

**Decision: keep the prefix.** It wins on the core use case (flag lookup, +6) and
most strongly on terse queries (+4 of 15), which is how these questions actually get
typed. The mechanism is straightforward: the prefix injects the tool name into every
chunk, so a query that names the tool matches far more of the page.

A caution about reading this table. As percentages, `concept` (−6.2%) and
`variant-no-tool` (−6.7%) look like a real trade-off, and there is a tidy story
available — the prefix should dilute chunks when the query deliberately avoids
naming the tool. **Both "regressions" are one question.** On subsets of 15 and 16
that is noise, and the tidy story is not supported. The effect may well be real; this
experiment cannot show it. If it matters later, it needs more `no-tool` questions,
not a firmer conclusion from these.

The score gate is a wash: 65.0% abstention recall at ≥90% coverage without the
prefix, 62.5% with it. Prefixing does not help abstention either way.

Both indexes are kept. `scripts/compare_runs.py` reports fixed/broken by qid, per-tag
deltas, and the gate operating point for any two runs.

---

## Correction (phase 4)

Two defects found while validating the phase 4 tool eval set affect the numbers
above. Neither changes a conclusion; both are documented in
`docs/phase4-results.md`.

1. **`u22` was mislabelled `out-of-corpus`.** `pip.1`, `pip-install.1` and
   `pip3-install.1` are all indexed, and `pip-install.1#USAGE` reads
   `python -m pip install [options] <requirement specifier>`. The question is
   answerable, so abstention recall here was **understated by about two points**:
   85.0% → 87.2% on the unanswerable set.
   The write-up above quotes u22's answer, *"python -m pip install ... [3]"*, as fabrication. The **citation** was fabricated; the **answer** is what `pip-install.1` says. The abstention-by-reason table's
   `out-of-corpus` row is n=5, not n=6.
2. **A corpus extraction bug** deleted tagged-paragraph tags that are
   cross-references (`pip3-install(1)`), across 1,084 of 4,158 pages. Rebuilding the
   index on the corrected corpus moved recall@5 by less than one question.
