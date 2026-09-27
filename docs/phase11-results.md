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

## Results

(not yet run)
