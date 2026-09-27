# Phase 11: robustness to how people actually ask

Status: **pre-registered 2026-09-27, before any pool run.** The plan is in
`docs/robustness-plan.md`. This part covers R3 (baseline map). R4-R6 get their own
pre-registration sections, written after R3's numbers and before their own runs.

## Question

With the shipped system unchanged, how much does a question's wording change the outcome?
Wording here means typos, synonyms, a chatty or terse register, or leaving out the tool's
name. And for each kind of wording, does the loss happen at retrieval (the right page never
reaches the model) or at reading (the page arrives and the answer is still wrong)?

## Pool

Every row is labelled from its clean base: same tokens, same gold sections, and
`gold.aliases_for` falls back to the base qid.

| file | rows | what |
|---|---|---|
| `data/eval/questions.jsonl` | 166 | 98 clean man-page bases + their 68 older paraphrase/typo/terse/no-tool variants |
| `data/eval/emacs_questions.jsonl` | 78 | 62 clean Emacs bases + 16 older variants |
| `data/eval/variations_typo.jsonl` | 320 | `typo1` (1 edit) and `typo3` (3 edits) for all 160 bases, generated (`make_variants.py`) |
| `data/eval/variations_man.jsonl` | 366 | `synonym`, `casual`, `terse` for 98 bases; `no-name` for the 72 answerable |
| `data/eval/variations_emacs.jsonl` | 230 | same four kinds for the 62 Emacs bases (44 answerable get `no-name`) |
| **total** | **1,160** | |

## Configuration (the shipped system, unchanged)

`data/index/phase11.db` (man pages + Emacs manuals), **no domain filter**, dense retrieval,
0.6B reranker over 50 candidates, k=5, gate 0.65, GBNF citations, 4B generator. The
generator runs on port 8083 because another process owns 8080 on this machine; port does
not affect output.

```
.venv/bin/python scripts/eval_answers.py --db data/index/phase11.db --eval <pool.jsonl> \
    --mode dense --rerank -k 5 --candidates 50 --gate 0.65 --grammar \
    --gen-url http://127.0.0.1:8083 --stage both --name p11-pool
.venv/bin/python scripts/robustness_report.py --answers data/eval/results/p11-pool-answers.json \
    --pool <the five files above> --out data/eval/results/p11-pool-robustness.json
```

The pool file is the five files concatenated in the order above. qids are unique across
them (checked when they were built).

## What is reported

Per variant kind, overall and split by domain (linux / emacs):

- answerable: n, evidence@5, correct (aliased, recomputed from answer text)
- unanswerable: n, correctly abstained
- **paired against each variant's own clean base in the same run**: kept / lost /
  gained / net, with an exact two-sided McNemar p, for correctness and for evidence;
  for unanswerable rows, abstention lost / gained

Pairing is the primary view. A variant and its base share the label, so any strictness
in a label (for example the concept rows scored on prose) cancels out of the delta.

## How R3's result will be read (fixed now)

A variant kind is **broken** if its paired correctness net is ≤ −3 with McNemar p < 0.05.
It is **fragile** if the net is ≤ −3 with p ≥ 0.05, or its evidence@5 falls more than 10
points below its bases'.

For each broken or fragile kind, where the loss sits is read from the paired evidence
counts:
- mostly evidence lost → retrieval
- evidence kept but the answer lost → reading

Abstention: any kind whose unanswerable variants abstain less often than their bases, by 2
or more, is reported as a safety regression whatever its correctness numbers are.

R3 changes nothing and ships nothing. It chooses what R4 targets. R4's arms and decision
rule are pre-registered below it once this map exists.

## Known limits, stated before the run

- n per kind is 62-160 pairs. A single kind can only detect losses of about 5 or more
  answers.
- About 8 man-page concept rows are scored on prose (`'day of week'`,
  `'cannot be caught'`). They are strict on bases and variants alike.
- Typos are generated, not observed. They follow a keyboard-adjacency model, which may be
  cleaner or messier than real typing.
- Written variants were authored by a model against the gold section and audited on
  20-row samples, not row by row.

## Results (R3, run 2026-09-27)

Run as pre-registered, with one deviation. The first attempt crashed: the shipped query-time
reranker profile (`servers.sh start reranker-query`: context 1024, batch 768) returned HTTP 500
on a pool question whose query+chunk pair came to 1,277 tokens. The run was repeated in the
documented two-stage form: retrieval with the full reranker profile (same model, context 8192,
batch 2048), then generation with the generator alone. Reranker scores do not depend on batch
size, so the measurement is unchanged. The crash is a **product bug** that the pool found: the
same input would fail `asq`. It is fixed before R4 (see below).

Files: `data/eval/results/p11-pool-answers.json`, `p11-pool-robustness.json` (from
`scripts/robustness_report.py`). Retrieval took 3,772 s (3.25 s/question) and generation 1,122 s.
The pool file is the five files concatenated (sha256 `a5da3af8…`).

### The map

Answerable rows, correctness aliased and recomputed from answer text. **Paired** means each
variant against its own clean base in the same run.

| kind | n | evidence@5 | correct | paired lost | gained | **net** | McNemar p | evidence net |
|---|---|---|---|---|---|---|---|---|
| clean | 116 | 105 (90.5%) | 92 (79.3%) | – | – | – | – | – |
| typo1 (1 typo) | 116 | 101 | 77 | 18 | 3 | **−15** | 0.0015 | −4 |
| typo3 (3 typos) | 116 | 97 | 65 | 30 | 3 | **−27** | <0.0001 | −8 |
| synonym | 116 | 86 | 64 | 35 | 7 | **−28** | <0.0001 | −19 |
| casual | 116 | 95 | 80 | 22 | 10 | **−12** | 0.0501 | −10 |
| terse (old + new) | 136 | 108 | 74 | 43 | 11 | **−32** | <0.0001 | −15 |
| no-name | 116 | 83 | 63 | 36 | 7 | **−29** | <0.0001 | −22 |
| paraphrase (old) | 16 | 12 | 8 | 5 | 2 | −3 | 0.45 | −1 |
| typo (old) | 20 | 17 | 13 | 4 | 2 | −2 | 0.69 | −1 |
| no-tool (old) | 15 | 7 | 4 | 8 | 1 | **−7** | 0.039 | −6 |

**Read against the pre-registered rule:**
- **Broken** (net ≤ −3, p < 0.05): typo1, typo3, synonym, terse, no-name, and the older no-tool.
- **Fragile**: casual (−12, p = 0.0501). Its evidence@5 is 8.6 points under its bases'.
- **Safety: no regression.** Unanswerable variants abstain as often as their bases (per-kind
  abstention lost/gained is 1/1, 2/2, 1/2, 1/0, 1/2). Rewording makes the system *miss*
  answers. It does not make it *invent* them.

Only about 1 in 5 losses for the smallest possible perturbation (one typo) is a retrieval
loss. The system is not robust to how people actually ask. Every broken kind except the old
paraphrases loses 12-32 of ~116 answers.

### Where it breaks: cause of every lost pair (base right, variant wrong)

| kind | page never retrieved | gate refused | model abstained with the page | model answered wrong with the page |
|---|---|---|---|---|
| typo1 | 4 | 2 | 4 | 8 |
| typo3 | 6 | 8 | 7 | 9 |
| synonym | 17 | 7 | 2 | 9 |
| casual | 12 | 3 | 1 | 6 |
| terse | 16 | 4 | 11 | 12 |
| no-name | 17 | 8 | 4 | 7 |

- **Typos break the reader, not retrieval.** 12 of 18 typo1 losses had the right page in hand.
  Reading the 12: with the same extracts, a single typo pushes the 4B into "I don't know"
  (a30 "usfeul", b02 "filesywtem", b07 "whwn") or onto a neighbouring wrong option
  (`sort -g` for `-n`, `C-M-%` for `query-replace`). One of the 12 (a12, `cp -p`) is a correct
  answer the label misses. **Implication: the typo must be repaired before the model sees the
  question, not only before retrieval.**
- **Synonyms and no-name wording break retrieval.** 17 of 35 and 17 of 36 losses never
  retrieved the page, and the gate refused another 7-8. This is a vocabulary gap, and it is
  much worse on Emacs: synonym evidence net −14/44 Emacs vs −5/72 man, and no-name −18/44 vs
  −4/72. Users say "pane", "close", "copy"; the manual says window, kill, kill-ring-save.
- **Terse queries lose both ways.** 16 not retrieved; 23 read wrong or abstained.
- The gate fires more on variants (typo3 11, synonym 9, no-name 8 answerable rows gated,
  against 2 for clean bases), but it explains only 2-8 losses per kind.

### Per domain (paired correctness net)

| kind | man pages (72 pairs) | Emacs (44 pairs) |
|---|---|---|
| typo1 | −13 (p 0.002) | −2 |
| typo3 | −22 | −5 |
| synonym | −15 | −13 |
| casual | −4 | −8 |
| terse | −11 | −21 |
| no-name | −10 | −19 |

Typos hurt the man-page set more. The Emacs manuals' long identifiers seem to anchor retrieval
and reading. Vocabulary (synonym, terse, no-name) hurts Emacs far more.

## What R3 selects for R4

1. **Fix the reranker crash first.** Oversized query+chunk pairs must be truncated, not
   rejected.
2. **R4a, spelling normalisation applied to the question itself**, so the corrected text feeds
   retrieval *and* the model. Targets typo1/typo3, where the loss is mostly in reading.
3. **R4b, a 4B rewrite into documentation vocabulary** ("pane" → window, "close" → kill).
   Targets synonym, no-name and terse, where the loss is mostly in retrieval.

Each is pre-registered below before it runs.
