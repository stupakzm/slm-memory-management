# Phase 2 — the precision layer

Phase 1 ended with four instructions, in priority order: rerank, gate, fuse BM25,
enforce citations. Three of them worked. One did not, and the most useful thing
phase 2 produced is the measurement of which.

| | |
|---|---|
| Index | phase 1 chunks and vectors, unchanged, plus an FTS5 table |
| Candidates | dense top-50 |
| Rerank | Qwen3-Reranker-0.6B Q8, cross-encoder over all 50, keep 5 |
| Gate | refuse below top-1 reranker score 0.65, model never invoked |
| Generation | Qwen3-4B Q4, GBNF-constrained citations |
| Latency | 89 ms retrieve + 3.1 s rerank + 1.1 s generate |

## Headline

| metric | phase 1 | phase 2 | |
|---|---|---|---|
| retrieval recall@5 | 63.5% | **71.4%** | +7.9 |
| MRR@10 | 0.471 | **0.553** | +0.082 |
| answer accuracy | 47.6% | 48.4% | +0.8 |
| accuracy given evidence | 71.2% | 65.6% | −5.6 (see below) |
| abstention recall (unanswerable) | 85.0% | **92.5%** | +7.5 |
| unsupported answers (all questions) | 19.3% | **15.1%** | −4.2 |
| abstention recall @ ≥90% coverage | 62.5% | **72.5%** | +10.0 |
| uncited claims | — | **0.0%** | structural |
| citations pointing at real evidence | — | 69.8% | new |

The briefing predicted "the largest single jump in the project" here. Retrieval
jumped. **Answer accuracy moved 0.8 points.** That gap is the phase 2 result, and
the rest of this document is about where the retrieval gain went.

`accuracy given evidence` falling is mostly a denominator effect, not a regression,
and it is worth taking apart because the rate is the one number here that looks like
a loss:

| | correct / with evidence | rate |
|---|---|---|
| phase 1 | 57 / 80 | 71.2% |
| phase 2, no gate | 61 / 90 | 67.8% |
| phase 2, gate 0.65 | 59 / 90 | 65.6% |

The reranker surfaces evidence for ten more questions and the model converts four of
them — 40%, well below the 71% it manages on the questions phase 1 already found. So
the count rises while the rate falls. The gate then costs two more, which is the
coverage it is knowingly bought with. Reported as a rate because that is how phase 1
reported it, and broken out because the comparison is not like for like.

## The reranker earns its place

Cross-encoder over the dense top-50, keeping 5:

| metric | dense | + rerank |
|---|---|---|
| recall@1 | 34.9% | **43.7%** |
| recall@5 | 63.5% | **71.4%** |
| recall@20 | 83.3% | **89.7%** |
| MRR@10 | 0.471 | **0.553** |
| gold doc in top 5 | 67.5% | 72.2% |

At k=5: **17 questions fixed, 7 broken, net +10.** In raw counts per tag, because
percentages on subsets of 15 are what the phase 1 write-up warned about:

| tag | n | dense → rerank | |
|---|---|---|---|
| flag lookup | 81 | 46 → 53 | **+7** |
| exact-token | 39 | 27 → 30 | **+3** |
| concept | 16 | 11 → 13 | +2 |
| variant-no-tool | 15 | 5 → 7 | +2 |
| variant-typo | 15 | 8 → 9 | +1 |
| config | 27 | 21 → 22 | +1 |
| variant-terse | 15 | 13 → 11 | −2 |

The +7 on flag lookup is the one that matters: it is the core use case, it was
phase 1's worst category, and 81 questions is enough to read. The rest are one or
two questions each and should be treated as noise until a larger set says otherwise.

Recall@20 reaching 89.7% is worth noting against the **candidate-pool ceiling of
90.5%** — the fraction of questions whose answer chunk is anywhere in the dense
top-50. The reranker is extracting essentially everything the pool contains. Further
retrieval gains have to come from a better pool, not better reranking.

The cost is real: 3.1 s per question for 50 pairs through a 0.6B cross-encoder on an
RTX 3060, against 89 ms for the dense search it reorders.

## BM25 does not earn its place

This is the phase 2 finding that contradicts the plan.

Phase 1 predicted BM25 would fix the `no-tool` collapse and the exact-token
questions. Measured, on the same 166 questions:

| | recall@5 | MRR@10 |
|---|---|---|
| dense | 63.5% | 0.471 |
| BM25 | 35.7% | 0.215 |
| RRF hybrid | 64.3% | 0.446 |
| dense + rerank | 71.4% | 0.553 |
| hybrid + rerank | 72.2% | 0.534 |

Hybrid is +1 question at k=5 and worse at every other depth. But k=5 is the wrong
place to look, because the reranker reorders whatever it is given — what fusion has
to justify is the **candidate pool**:

| pool depth | dense | BM25 | RRF | RRF 2:1 |
|---|---|---|---|---|
| @20 | 83.3% | 50.0% | 79.4% | 83.3% |
| @50 | **90.5%** | 59.5% | 88.9% | 91.3% |
| @100 | 92.9% | 68.3% | 92.1% | 92.9% |

Equal-weight fusion makes the pool *worse*. And decisively: of the **12 questions
dense misses at pool depth 50, BM25 finds 0.**

Two predictions failed and both are worth stating plainly.

**No-tool queries.** Phase 1 guessed BM25 would rescue queries that omit the tool
name. BM25 scores 6.7% on them — the worst of any category, against dense's 33.3%.
The guess had the mechanism backwards: dropping the tool name removes exactly the
high-IDF term BM25 depends on, while the embedding still has the paraphrase.

**Exact tokens.** The phase 0 eval set forbids a question containing its own answer
token — `resolve_gold.py` rejects those — so no query ever spells `--exclude-from`.
BM25's advertised advantage is the case this eval set excludes by construction, and
excludes deliberately, because that is how people actually type.

So the fair test is to ask directly, using the answer token *as* the query:

| query = the answer token, n=39 | recall@5 |
|---|---|
| BM25, dashes split | 69.2% |
| BM25, dashes kept as one atom | 82.1% |
| dense | **84.6%** |

The tokenizer matters and was itself ablated: keeping `--exclude-from` as a single
token recovers 13 points on token queries and costs nothing on natural ones (35.7% →
34.9%, one question). It still loses to dense. **On its own home ground, with the
tokenizer chosen to favour it, BM25 loses.** Qwen3-Embedding handles literal flag
strings better than the briefing assumed a dense model would.

**Decision: BM25 is not in the default pipeline.** The code, the FTS5 index and the
`--mode bm25 / hybrid` switches stay, so the result is reproducible and so a corpus
with different query habits can turn it back on. It is one `--mode` flag, not a
deletion. If a later phase adds a query log where users do type flag names verbatim,
this is the first thing to re-measure.

## The gate works — on the wrong population

The gate is the mechanism Finding 04 asks for: below threshold the generator is
never invoked, so speculation is impossible rather than discouraged.

The reranker's score is a far better gate signal than the dense score, exactly as
predicted. On phase 1's own metric — best abstention recall while keeping ≥90% of
answerable questions — it goes **62.5% → 72.5%**.

### Choosing the threshold on the right objective

Sweeping the retrieval score against a coverage target picks 0.8673. Sweeping
*end-to-end outcomes* picks 0.65, and it is a strictly better operating point:

| threshold chosen by | accuracy | abstention | unsupported |
|---|---|---|---|
| retrieval proxy, 0.8673 | 44.4% | 92.5% | 14.5% |
| end-to-end, 0.65 | **48.4%** | 92.5% | 15.1% |

Four points of accuracy for the same abstention recall. The method is cheap and
should be the default: the gate only decides *whether* to invoke the model, never
what it writes, so every threshold can be replayed exactly from one ungated
generation run. `sweep_gate.py --answers` does this at zero GPU cost, and the
replayed prediction for 0.65 matched the confirming run to the question.

### What the gate cannot do

Phase 1 named its worst defect precisely: when retrieval quietly returned nothing
useful, the model invented an answer 56.5% of the time — and concluded this "happens
exactly where a gate applies."

It does not. Top-1 reranker score, by population:

| population | n | p10 | p50 | p90 |
|---|---|---|---|---|
| answerable, evidence retrieved | 90 | 0.874 | 0.991 | 0.999 |
| answerable, **evidence missed** | 36 | 0.536 | **0.987** | 0.999 |
| unanswerable | 40 | 0.013 | 0.328 | 0.994 |

The middle row is the problem. A question whose retrieval failed looks, to the
reranker, almost exactly like one whose retrieval succeeded — median 0.987 against
0.991. At threshold 0.65 the gate removes **70% of unanswerable questions and 11% of
answerable-but-unretrieved ones**, at a cost of 3% of the questions it should keep.

The consequence is visible in the end-to-end numbers: abstention recall climbs
85.0% → 92.5%, while "answered with no evidence" barely moves, 61.1% → 61.1%. The
gate catches questions the corpus cannot answer. It does not catch questions the
corpus *can* answer that retrieval got wrong, because the reranker is confidently
wrong on a plausible chunk from a plausible wrong page.

That is a mechanism failure, not a threshold failure — no threshold separates two
distributions with the same median. Fixing it needs a signal that is not the
retriever's own confidence.

## GBNF citations: cheap, and honest about what they buy

Phase 1 asked for citations in the prompt and got citation-shaped text — `NULL [3]`
where extract [3] said nothing of the kind. Constrained decoding, with no gate, so
the grammar is measured alone:

| | prompt only | GBNF |
|---|---|---|
| answer accuracy | 50.0% | 50.0% |
| accuracy given evidence | 68.9% | 67.8% |
| **claims with no citation** | 5.4% | **0.0%** |
| citation out of range | 0.0% | 0.0% |
| **cited extract contains the answer** | 61.6% | **70.6%** |

Free on accuracy, and it removes uncited claims by construction rather than by
persuasion.

Be exact about the guarantee. A grammar cannot make a citation *correct* — nothing
stops the model producing a parametric answer and pointing at extract 2. What it
guarantees is that a pointer exists and parses, which is what makes the third row
computable at all. Phase 1 could not run that check: `[3]` was free text the model
could omit or malform. The grammar is not the guarantee; it is what makes the
guarantee checkable.

The 30% of citations that do *not* point at the answer is now a visible, measurable
defect rather than an invisible one.

## Phase 2 exit gate

> Abstention recall high on the unanswerable set without collapsing answer coverage
> on the answerable set.

**Passed.** 92.5% abstention (from 85.0%) with answer accuracy 48.4% (from 47.6%)
and unsupported answers down from 19.3% to 15.1%.

Two things it did not deliver, both now measured:

1. The predicted jump in answer accuracy. +7.9 points of recall@5 bought +0.8 points
   of accuracy, because the model converts newly-retrieved evidence at only 40%.
2. Any dent in speaking without evidence, which is a retrieval-precision problem the
   retriever's own confidence cannot detect.

## What phase 3 has to answer

Phase 1's failure split still stands, and phase 2 resolved the half it was aimed at:

- **wrong document** (59% of misses) → the reranker fixed a large part of this.
- **right document, wrong chunk** (41%) → untouched. This is phase 3's target, and
  it is now also the mechanism behind phase 2's two failures: a 1,000-char window
  cut blindly through a 240 KB option dump is exactly what produces a chunk that
  looks right to a cross-encoder and does not contain the flag.

The pool ceiling of 90.5% says the same thing from the other end. The reranker has
taken almost everything the flat pool holds; the next gain has to come from chunks
that are cut where the document says to cut them.

## Runs

Every number above is reproducible from `data/eval/results/`:

| run | pipeline |
|---|---|
| `phase2-dense` | dense only — reproduces phase 1 exactly, verifying the refactor |
| `phase2-bm25` | lexical only |
| `phase2-hybrid` | RRF, no reranker |
| `phase2-dense-rerank` | dense top-50 → rerank |
| `phase2-hybrid-rerank` | RRF top-50 → rerank |
| `phase2-rerank` | answers, no gate, no grammar |
| `phase2-rerank-grammar` | answers, grammar only |
| `phase2-rerank-gate` | answers, gate only, retrieval-proxy threshold |
| `phase2-full` | answers, gate 0.65 + grammar |
