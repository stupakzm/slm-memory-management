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

## R4a pre-registration: spelling normalisation (written 2026-09-27, before the build and the run)

**Mechanism.** Before a question is used for anything, snap each word the corpus does not
contain to the corpus word closest in Damerau-Levenshtein distance (≤1 for words of 4-7
letters, ≤2 for 8+). Ties go to a keyboard-adjacent substitution, then to the more frequent
word. The vocabulary is the index's own chunk text: words of 3+ letters seen at least twice.
The first letter is **not** protected. The typo generator never touches it, so protecting it
would tune the mechanism to the generator, not to people. Tokens with digits, `-`, `/`, `=` or
`.` are never touched, because flags and paths are exact. The **corrected question feeds both
retrieval and the model**, because R3 showed typos mostly break the reader.

**Arm.** Everything as in R3 plus `--normalize spell`. Only rows whose normalised question
differs from the original are re-run (`--qids`). All other rows are identical inputs to the
control, so the control's results for them are carried over (`robustness_report.py --overlay`).
Retrieval uses the full reranker profile, as in R3.

**Decision rule (ships as default only if all four hold):**
1. **Typos recover:** typo1 + typo3 paired correctness (arm vs control, same variant rows)
   gains ≥ +10 net answers, McNemar p < 0.05.
2. **Clean text is left alone:** the normaliser changes ≤ 3 of the 160 clean base questions,
   and clean correctness loses no more than 1 answer net.
3. **No new invention:** abstention on unanswerable rows, summed over all kinds, does not
   fall by 2 or more.
4. **Cheap:** normalisation adds < 10 ms per question.

Failing 2 or 3 means it does not ship, whatever 1 says. Passing 1 with p ≥ 0.05 is reported as
"direction only".

## R4b pre-registration: rewriting the question into documentation vocabulary (written 2026-09-27, before any R4b run)

**A correction to R3's framing, found while writing this.** R3's "shipped configuration" is the
eval baseline that phases 7-9 used (`--rewrites 0`). The interactive command itself
(`scripts/ask.py`, `asq`) defaults to `--rewrites 1` with the man-page-style rewrite prompt
(`ask.py`: `args.rewrites = 0 if args.retrieve_only else 1`). So R3 measured the eval
baseline, not `asq` exactly. R4b therefore compares three settings, not two.

**Mechanism.** One 4B rewrite of the question, fused with the original for retrieval
(`retrieve_fused`, unchanged; the gate still reads the original question's own top-1). The
reader sees the original question. Two prompts:
- `man`: today's `asq` prompt ("...for a Linux manual-page search engine... name the likely
  command...").
- `docs`: the new prompt ("...software documentation (Linux manual pages and the GNU Emacs
  manuals). Use the terminology that documentation itself would use... Keep any program,
  package or command name the user wrote exactly as written...").

**Rows.** The kinds R3 found broken at retrieval, plus clean for harm: clean, synonym, casual,
terse (old and new), no-name. That is 781 rows, answerable and unanswerable. Typos are left to
R4a/R4a′.

**Serving.** The rewrite needs the generator during retrieval, so all three models run in the
query-sized `serve` profile (`reranker-query`, now safe after the oversized-pair fix). Scores
are identical to the full profile except on the rare oversized pairs, which are now truncated
instead of crashing. The control (R3) is re-used as is.

**Arms.** `R4b-man` (`--rewrites 1 --rewrite-style man`) and `R4b-docs` (`--rewrites 1
--rewrite-style docs`), each against the R3 control on the same rows, and against each other.

**Decision rule.** `docs` becomes `asq`'s default rewrite style only if all four hold:
1. **Vocabulary recovers:** over synonym + no-name + terse, `R4b-docs` vs control has paired
   correctness net ≥ +15 with McNemar p < 0.05, **and** it is not worse than `R4b-man` (net ≥ 0
   vs man).
2. **Clean text unharmed:** clean paired net vs control ≥ −2.
3. **No new invention:** abstention on the unanswerable rows in these kinds does not fall by 2
   or more vs control.
4. **Affordable:** the rewrite adds ≤ 1.5 s at p50 per question (it is one short generation).

If `man` itself passes 1-3 against control, that is reported as well: it would mean the eval
baseline has been understating `asq` all along.

## R4a results (run 2026-09-27)

Diagnostic arm, as pre-registered: the 370 pool rows whose question the normaliser changes were
re-run with `--normalize spell`. The other 790 rows are identical inputs to the control. Files:
`data/eval/results/p11-r4a-answers.json`. Paired arm vs the R3 control on the same rows:

| kind | answerable rows changed | lost | gained | **net** | McNemar p |
|---|---|---|---|---|---|
| typo1 | 99 | 4 | 15 | **+11** | 0.019 |
| typo3 | 116 | 4 | 20 | **+16** | 0.0015 |
| synonym | 3 | 2 | 0 | −2 | 0.50 |
| typo (old), casual, clean | 26 | 0 | 0 | 0 | – |

Unanswerable rows changed: 126. Abstention lost 3, gained 7 (net +4).

**Against the rule:**
1. Typos recover: **+27 net (p < 0.0001). Pass.** That is 27 of the 42 answers typos cost in R3.
2. Clean text left alone: **fail.** 11/160 clean questions changed (≤ 3 allowed), mostly names of
   uninstalled tools rewritten into installed ones (`nmap`→`mmap`, `elpy`→`elpa`,
   `projectile`→`projectfile`).
3. No new invention: pass (net +4). In this run the renamed tools still drew refusals, but that is
   luck of the corpus, not a property of the mechanism.
4. Cheap: pass (p95 0.36 ms per question).

**Verdict: does not ship** (rule 2). `--normalize spell` stays default-off.

**What it establishes:** R3's reading of the typo loss was right. Repairing the words of the
question itself, not only the retrieval query, recovers most of what typos cost the 4B reader.
The open problem is doing it without renaming things the corpus has never heard of. That is
R4a′: the 4B corrects spelling under an explicit keep-every-name instruction (built, default
off). It is pre-registered next, with the same four rules.

## R4a′ pre-registration: the 4B corrects spelling, keeping every name (written 2026-09-28, before any R4a′ run)

**Mechanism.** `--llm-correct` (built in tsk_20260927_docvocab, default off). The 4B is asked to
fix spelling and nothing else, holding every program, package, command, option and file name
exactly as written (`build_correct_prompt`). Output is grammar-constrained to one line at
temperature 0. It is discarded, and the question kept, if it is empty or longer than twice the
input plus 20 characters. The corrected question feeds retrieval **and** the model, as in R4a.
No rewrites (`--rewrites 0`), so the only difference from R3 is the corrected text.

**Which rows run.** Unlike R4a's normaliser, which rows the model changes can't be known
without asking it. So:
1. **Pre-pass:** `Generator.correct` over all 1,160 pool rows. It records the corrected text
   and the wall time per question, which is rule 4's measurement.
2. **Arm:** every row whose corrected text differs from the original at all, as a byte string,
   is re-run with `--llm-correct` (`--qids`). All other rows are identical inputs to the control,
   and their control results carry over (`robustness_report.py --overlay`), as in R4a.
   If the in-run correction (cached as `question_corrected`) differs from the pre-pass on any row,
   the count is reported. The arm result uses the in-run text.

**What counts as "changed" for rule 2.** Words, not bytes: the question lower-cased and split on
anything that isn't a letter, digit, `-`, `.`, `/` or `=`. A change to capitals or punctuation
alone is cosmetic. Such rows are still re-run, but they don't count against rule 2. This is
fixed now because the model, unlike the normaliser, may tidy capitals.

**Serving.** As R4b: all three models in the `serve` profile. The control is R3 as is.

**Decision rule (ships as `asq`'s default only if all four hold).** Rules 1-3 are R4a's
unchanged; rule 4 is re-set for a model call:
1. **Typos recover:** typo1 + typo3 paired correctness (arm vs control, same rows) net ≥ +10,
   McNemar p < 0.05.
2. **Clean text is left alone:** the model changes the words of ≤ 3 of the 160 clean base
   questions, and clean correctness loses no more than 1 answer net. The three tool names R4a
   renamed (`nmap`, `elpy`, `projectile`) are reported by name, whatever the count.
3. **No new invention:** abstention over all unanswerable rows in the pool, summed over every
   kind, does not fall by 2 or more.
4. **Affordable:** the correction adds ≤ 1.5 s at p50 per question (R4b's bound for one short
   generation; R4a's 10 ms was set for a lookup and no model call can meet it).

Failing 2 or 3 means it does not ship, whatever 1 says. Passing 1 with p ≥ 0.05 is reported as
"direction only". If R4b's `docs` rewrite also ships, the two together are a separate arm
(plan R4c) and are not claimed from these two results.

## R5 pre-registration: rerank 20 candidates instead of 50 (written 2026-09-28, before any R5 run)

**Mechanism.** `--candidates 20` instead of 50: the dense retriever passes 20 chunks, not 50,
to the 0.6B reranker. Everything else is R3's configuration. The reranker's own ordering is
unchanged. A chunk can only be lost if dense ranked it 21st-50th and the reranker would have
lifted it into the top 5. The gate reads the reranked top-1, so it moves only in that case too.

**Rows.** The full pool, 1,160 rows. Speed has to be paid for everywhere, so the bill is
checked everywhere.

**Serving.** R3's full profile (`reranker`, not `reranker-query`). Both arms then score every
pair identically, and the comparison is candidate count alone. The control is R3 as is.

**Latency.** `eval_answers.py` doesn't time each question, so rule 4 is measured separately.
A fixed random sample of 100 pool rows (seed 11) is retrieved and answered end to end at 20
and at 50, on the same servers, one arm after the other, after a warm-up query. Report p50 and
p95 per question for retrieval and for end to end.

**Decision rule (20 becomes the default only if all four hold):**
1. **Correctness holds:** paired correctness over all answerable rows (arm vs control) net ≥ −2.
2. **Evidence holds:** paired evidence@5 net ≥ −3.
3. **No new invention:** abstention over all unanswerable rows does not fall by 2 or more.
4. **Actually faster:** retrieval p50 per question falls by ≥ 30 %.

Rules 1-3 test non-inferiority: at this n, "loses nothing measurable" can only mean a loss this
small or smaller. Any loss is listed by qid, whatever the count. If rules 1-3 hold and 4 fails,
it does not ship: slower-or-equal for the same answers is not worth a changed default.

## R4b results (run 2026-09-28)

Run as pre-registered: 781 rows (clean 160, synonym 160, casual 160, terse 185, no-name 116),
serve profile, `--rewrites 1` with `--rewrite-style man` and with `docs`. Files:
`data/eval/results/p11-r4b-{man,docs}-answers.json`. Paired against the R3 control on the same
rows (600 answerable, 181 unanswerable):

| kind | answerable | man vs control | docs vs control | docs vs man |
|---|---|---|---|---|
| clean | 116 | −16 (22 lost / 6 gained), p=0.004 | −14 (21/7), p=0.013 | +2 |
| synonym | 116 | −5 | −6 | −1 |
| no-name | 116 | −17 (20/3), p=0.0005 | −10 (16/6), p=0.053 | +7 |
| terse | 136 | +4 | −3 | −7 |
| casual | 116 | −13 (19/6), p=0.015 | −8 | +5 |
| **synonym + no-name + terse** | 368 | **−18**, p=0.047 | **−19**, p=0.034 | −1 |
| all | 600 | −47, p<0.0001 | −41, p=0.0002 | +6, p=0.59 |

Abstention on the 181 unanswerable rows: man lost 0 and gained 1; docs lost 0 and gained 0.
Rewrite latency on a fixed sample of 100 of these rows (seed 11, warm server): man p50 0.60 s,
p95 1.65 s; docs p50 0.19 s, p95 1.54 s.

**Against the rule:**
1. Vocabulary recovers: **fail.** docs vs control is −19 net, not the required ≥ +15. It is not
   worse than man (−1, p=1.0), but that half is moot.
2. Clean text unharmed: **fail.** −14 against a floor of −2.
3. No new invention: pass (abstention unchanged).
4. Affordable: pass (p50 0.19 s).

**Verdict: `docs` does not ship.** `man` doesn't pass 1-3 against control either. So the
pre-registered question "has the eval baseline been understating `asq`?" gets the opposite answer.
On this eval, `asq`'s default one-rewrite fusion **costs** answers compared with no rewrite.

### Diagnostic, not pre-registered: the fused path also changes candidate depth

Found while reading `retrieve_fused` after the run: in the fused path, every variant, **the original
question included**, gets `VARIANT_CANDIDATES = 20` candidates. The control gives the original 50.
So both arms changed two things at once, and the pre-registration missed it. One extra arm
separates them: `--rewrites 0 --candidates 20`, serve profile, same 781 rows
(`p11-r4b-diag-c20-answers.json`). It differs from control in depth (and in the serve profile's
truncation of rare oversized pairs), and from the R4b arms only in the rewrite.

| step | clean | synonym | no-name | terse | casual | all (600) |
|---|---|---|---|---|---|---|
| control → depth 20, no rewrite | −5 | +2 | **−14** (16/2), p=0.001 | −1 | −3 | −21, p=0.017 |
| depth 20 → + man rewrite | −11, p=0.019 | −7 | −3 | +5 | −10 | −26, p=0.013 |
| depth 20 → + docs rewrite | −9 | −8 | +4 | −2 | −5 | −20, p=0.050 |

At equal depth, the rewrite itself still costs 20-26 answers, and clean questions lose most. The
depth cut costs 21, and 14 of those are no-name questions. A question that doesn't name its tool
depends most on candidates dense ranked 21st-50th. That is a strong prior against R5
(pre-registered above), and R5 will now be read knowing it. R5's own run still decides.

**Why the rewrite hurts.** Eight sampled rewrites, both styles: the 4B mostly writes a command
line, not a search query (`man chmod -R`, `strace -e all -f myprogram.exe`,
`git blame filename.txt`). It invents names (`seekable-file-position-lookup`, `AllowRootLogin`).
Once (`docs`, e16) it repeated `dired-visit-file-other-window` until it ran out of tokens. With one
rewrite, fusion is two lists, so a bad rewrite gets half the vote over which 5 chunks the reader
sees. The `docs` prompt's "use documentation's terminology" doesn't change that: the problem is
the model's idea of what a query looks like, not its vocabulary.

**What it establishes.** Query rewriting by this 4B is net harmful on this pool, whichever prompt
is used. The vocabulary gap R3 found (synonym, no-name) is not closed by asking the reader to
rephrase. Not tested here: more rewrites (3 lists instead of 2), or fusion with the original
weighted above the rewrite. Either would need its own pre-registration. Changing `asq`'s default
to `--rewrites 0` is suggested by these numbers but was not pre-registered, so it is left as a
decision, not shipped.

## R4a′ results (run 2026-09-28)

**Pre-pass**, all 1,160 rows: the model changes the text of 393 rows and the words of 386. By
kind (words changed): typo1 151/160, typo3 159/160, old typo 25/25, terse 34/185, synonym
7/160, **clean 6/160**, casual 2/160, no-name 2/116, paraphrase and no-tool 0. Wall time per
correction: p50 0.22 s, p95 0.44 s. In the arm, the corrected text matched the pre-pass on all
393 rows. Arm file: `data/eval/results/p11-r4a2-answers.json`. Paired against the R3 control on
the same rows:

| kind | answerable rows changed | lost | gained | **net** | McNemar p |
|---|---|---|---|---|---|
| typo1 | 110 | 5 | 14 | **+9** | 0.064 |
| typo3 | 116 | 5 | 25 | **+20** | 0.0003 |
| terse | 28 | 4 | 4 | 0 | – |
| synonym | 6 | 1 | 0 | −1 | – |
| clean, old typo, no-name, casual | 30 | 0 | 0 | 0 | – |

Unanswerable rows changed: 103. Abstention lost 3, gained 4 (net +1).

**Against the rule:**
1. Typos recover: **+29 net** (typo1 + typo3: 10 lost, 39 gained, **p < 0.0001**). **Pass.** R4a's
   lookup normaliser recovered +27.
2. Clean text is left alone: **fail.** The model changes the words of 6 of the 160 clean questions
   (≤ 3 allowed). Clean correctness is unchanged (0 net over 5 answerable rows). The three names
   R4a renamed are kept, in every variant: `nmap` (u04, 6 rows), `elpy` (eu09, 5 rows),
   `projectile` (eu01/eu18, 10 rows). But the model renames two other packages the corpus doesn't
   have: `tempel` → `template` (eu07), and "Kubernetes cluster" → "kubernetes-el cluster" (eu16).
   The other four are harmless rewordings the prompt didn't ask for ("left out" → "excluded",
   "hand" → "pass", a01/a09/a30/a41).
3. No new invention: pass (net +1).
4. Affordable: pass (p50 0.22 s).

**Verdict: does not ship** (rule 2). `--llm-correct` stays default-off.

**What it establishes.** The keep-every-name instruction fixed the specific renames R4a made,
and it recovers typos as well as the lookup did (+29 vs +27). It hasn't removed the failure class:
the model still turns names it doesn't recognise into words it does, and still tidies wording
nobody asked it to change. Both R4 spelling mechanisms now fail the same way on the same rule.
The next candidate is the combination: correct only the words the index vocabulary doesn't
contain (R4a's detector), and let the model choose the replacement only for those. A name the
corpus has never seen still gets flagged by the detector, so that alone would not fix eu07/eu16.
Not pre-registered.

## R4d pre-registration: cascade — rewrite only when the plain search refuses (written 2026-09-28, before the build and any run)

**Why.** R4b showed rewriting every question costs answers: the rewrite gets half the vote in
fusion, and the fused path also cuts candidates to 20. The user proposed spending rewrites only
where the plain search has already failed. A question that is answered at tier 0 is then
untouched by construction.

**Mechanism (`--cascade`).**
- **Tier 0:** exactly R3's configuration: dense, 50 candidates, reranked against the question,
  k = 5, gate 0.65, grammar-constrained answer. If the result is **not a refusal**, stop.
- **Tier 1**, only after a tier-0 refusal: the 4B writes 1 rewrite (`--rewrite-style docs`,
  chosen for speed and its better no-name score in R4b). The rewrite's 50 dense candidates are
  **added** to the question's 50 (deduplicated by chunk id). The union is reranked **against the
  original question**, the top 5 are kept, and the gate reads that list's top-1. The rewrite widens
  the pool and never ranks anything, so the gate keeps its calibrated meaning. The reader sees the
  original question. If this is not a refusal, stop.
- **Tier 2**, only after a tier-1 refusal: the same with 2 rewrites (a fresh `n=2` call). The pool
  is the question's candidates plus both rewrites'. The tier-2 result is final, whatever it is.
- **Refusal** = gated (top-1 below 0.65) **or** the reader refused (`abstained()`, the
  all-sentences rule the eval already scores with).

**Rows.** Every pool row the R3 control refused: 412 (158 answerable, 254 unanswerable). All other
rows were answered at tier 0 and carry over from the control (`--overlay`). Each re-run row runs
the full cascade from tier 0. Tier 0 runs in the serve profile, since the cascade needs all
three models resident. Rows where tier 0 now answers although the control refused are counted
and reported.

**Recorded per row:** the final tier (0/1/2) and the cascade's added wall time beyond tier 0.

**Decision rule (the cascade becomes `asq`'s default, replacing `--rewrites 1`, only if all four
hold):**
1. **Recovers answers:** over all answerable pool rows (arm overlaid on control), paired
   correctness net ≥ +10, McNemar p < 0.05.
2. **Clean text unharmed:** clean paired net ≥ −1. It should be ≥ 0 by construction; a loss would
   mean a defect.
3. **No new invention:** abstention over all unanswerable pool rows does not fall by 2 or more.
   This is the rule the cascade is most likely to break: 254 refused unanswerable rows get two more
   chances to find something.
4. **Affordable:** on rows that reach tier 1, the added wall time is ≤ 3 s at p50. Rows answered
   at tier 0 add nothing.

Failing 3 means it does not ship, whatever 1 says. Reported as well: gains and abstention losses
split by the tier they came from. If rule 3 fails at tier 2 but not at tier 1, that is reported,
and a tier-1-only cascade needs its own run before it can ship.

## R5 results (run 2026-09-28)

Run as pre-registered: all 1,160 rows, `--candidates 20`, full profile, retrieval and generation
staged. File: `data/eval/results/p11-r5-c20-answers.json`. Paired against the R3 control
(883 answerable, 277 unanswerable):

| kind | answerable | lost | gained | **net** | McNemar p |
|---|---|---|---|---|---|
| clean | 116 | 7 | 3 | −4 | 0.34 |
| no-name | 116 | 15 | 2 | **−13** | 0.0023 |
| casual | 116 | 6 | 2 | −4 | 0.29 |
| synonym | 116 | 7 | 6 | −1 | 1.0 |
| terse | 136 | 8 | 11 | +3 | 0.65 |
| typo1 / typo3 | 116 / 116 | 3 / 4 | 10 / 6 | +7 / +2 | 0.09 / 0.75 |
| paraphrase, no-tool, typo (old) | 51 | 2 | 1 | −1 | – |
| **all** | 883 | 52 | 41 | **−11** | 0.30 |

Evidence@5: 32 lost, 19 gained, **net −13**. Abstention: 3 lost (eu03.y1, u01.c, u05.s),
2 gained, net −1. Retrieval latency on the fixed 100-row sample, two interleaved passes each:
50 candidates p50 3.35 / 3.28 s, p95 3.54 / 3.44 s; 20 candidates p50 1.39 / 1.39 s, p95 1.47 /
1.49 s (**−58 %**).

Correctness lost, by qid: a01.m a02 a02.m a03.k a04.m a04.n a09 a09.y3 a10.k a10.m a11.y1 a17.y3
a19.m a19.s a24.y1 a25.t a29 a30 a37 a37.k a38.m a41.s a43.k b03.n b08.m b14.m c02.k c02.m c06
c06.y1 e01 e01.s e03.s e05.y3 e11.c e12.c e12.k e12.m e13.c e14.c e14.m e20.c e24.y3 e25.c e25.k
e25.m e25.s e26.s e35.m e38.m e39.m e41.s.

**Against the rule:**
1. Correctness holds: **fail**, −11 against a floor of −2.
2. Evidence holds: **fail**, −13 against −3.
3. No new invention: pass (net −1).
4. Faster: pass (−58 %).

**Verdict: does not ship.** 50 candidates stays the default. The loss sits where R4b's
diagnostic said it would: no-name questions (−13 here, −14 in the diagnostic, on a different
profile and run). A question that doesn't name its tool needs the candidates dense ranks 21st-50th.
The overall −11 is not significant (p = 0.30), but R5 is a non-inferiority test, and −11 is well
outside its margin. A future speed arm would have to keep depth for questions the gate finds weak.
One option is the R4d cascade's own shape: 20 first, 50 only on refusal.

## R4d results (run 2026-09-28)

Run as pre-registered: the 412 pool rows the R3 control refused (158 answerable, 254
unanswerable), `--cascade --stage both`, serve profile. File: `data/eval/results/p11-r4d-answers.json`.
All other 748 rows were answered at tier 0 and carry over. 6,262 s wall time (15 s per row).

**Where each row ended:**

| | final tier 0 | tier 1 | tier 2, answered | tier 2, refused |
|---|---|---|---|---|
| answerable (158) | 9 answered | 21 answered | 8 | 120 |
| unanswerable (254) | 0 | 2 answered | 4 | 248 |

Tier 0 answered 9 rows the control had refused. The inputs are identical, so this is the serve
profile and server nondeterminism, not the cascade. A byte-identity check the same day found
1/40 answers reworded on identical input (b05.m, identical on 3 further reruns). 4 of those 9 are
correct.

**Paired against control (answerable):** 0 lost, **19 gained (+19, p < 0.0001)**. Of the gains,
4 come from tier 0 (noise, above), 11 from tier 1 and 4 from tier 2. Counting only the cascade's
tiers: **+15, p < 0.0001**. By kind: typo3 +7, synonym +4, typo1 +3, no-name +2, clean, terse and
casual +1 each. Evidence@5 on these rows: 2 lost, 13 gained.

**Unanswerable:** abstention **lost 6, gained 0**. Tier 1 accounts for 2 and tier 2 for 4:
- `eu03.y1` (tier 1): rewrite `treeemacs-toggle-sidebar-visibility`, answered `window-toggle-side-windows`
- `eu17.y3` (tier 1): a 9-term rewrite about magit-todos, answered with org-agenda advice
- `u25` (tier 2): answered `systemctl is-active`
- `u11.y3` (tier 2): rewrites were long grep/rg command lines, answered with git-grep `--exclude-standard`
- `u07.k` and `eu07.k` (tier 2): both are refusals in substance ("the extract does not contain
  information about…"), phrased outside `ABSTAIN_RE`. That is the known limit of the abstention
  rule (`blk_abstain_re_false_positives`), and the rule counts them as spoken.

Even without the last two, abstention falls by 4.

**Added wall time**, rows that reached tier 1 (n = 403): p50 **11.8 s**, p95 16.4 s. Most rows
went through both tiers (two rewrite calls, two reranks over 100-150 candidates in the query-sized
reranker, two generations).

**Against the rule:**
1. Recovers answers: **pass.** +19 net (p < 0.0001), +15 from the cascade's own tiers.
2. Clean unharmed: pass (+1; nothing lost anywhere, as the construction predicts).
3. No new invention: **fail.** Abstention falls by 6 (tier 1 alone: 2, which is also a fail).
4. Affordable: **fail.** p50 11.8 s against 3 s.

**Verdict: does not ship.** `asq` keeps its current default for now.

**What it establishes.** The shape works on the side it was built for. Retrying only refusals, with
the rewrite limited to widening the pool, recovered 15 answers and cost none. R4b's every-question
rewrite lost 41-47. Answered rows stay untouched by construction, and nothing was lost there. The
cost is on the side the gate protects: 6 of 254 refused unanswerable questions got two more chances
and 4-6 of them took one. Two of the four real inventions came from rewrites that invented a
package name, the same failure R4b sampled. Candidate fixes, none pre-registered:
- **tier 1 only, with a stricter gate on retries.** A retry has already failed once, so the evidence
  should be held to more than the first attempt was. This is swept from these rows' recorded scores
  before any new run.
  *Checked offline the same day, on the 35 rows answered at tier 1-2:* the final reranker scores
  don't separate them. The inventions score 0.73, 0.92, 0.97, 0.99, 0.99, 0.99. The 15 correct
  answers score 0.67-1.00. Only u25 falls below 0.80, and a gate there also drops 2 correct
  answers. **A stricter retry gate is not the fix.** The retries find pages the reranker is sure
  about; the reader then answers a question those pages don't actually settle.
- **Rewrite quality**: drop rewrites that contain a name not in the index vocabulary, which is R4a's
  detector reused as a filter.
- **Latency**: stop at tier 1, and rerank only the rewrite's new candidates, merging them with the
  question's already-scored ones.

## Phase 11 R4 summary

| arm | mechanism | answers vs control | clean | invention | latency | ships |
|---|---|---|---|---|---|---|
| R4a | spell lookup | typos +27 | 11/160 renamed | ok | 0.4 ms | no (rule 2) |
| R4a′ | 4B spelling fix | typos +29 | 6/160 reworded | ok | 0.22 s | no (rule 2) |
| R4b | docs rewrite, every question | −41 (syn+noname+terse −19) | −14 | ok | 0.19 s | no (rules 1-2) |
| R4b (asq today) | man rewrite, every question | −47 | −16 | ok | 0.60 s | (control is better) |
| R4d | rewrite only on refusal | **+15** | +1 | −6 | 11.8 s | no (rules 3-4) |
| R5 | rerank 20, not 50 | −11 (no-name −13) | −4 | ok | −58 % | no (rules 1-2) |

No R4/R5 mechanism ships. One result changes the default story, though. `asq`'s current
`--rewrites 1` loses 47 of 600 answers against plain retrieval on the same rows, and no-rewrite
plain retrieval is what every result above is measured against. Switching `asq` to `--rewrites 0`
is a decision for the user. It was not pre-registered as a shipping rule.

## R6 pre-registration: quote-grounded answers (written 2026-09-28, before any R6 run)

**Why now.** Quote mode was built in phase 10 (`4c75c90`) and parked unmeasured. At n = 127, even
perfect grounding could not reach significance. The pool is now 1,160 rows. R4d also gave it a
concrete target: its inventions came from real, confidently-ranked pages that did not answer the
question (reranker 0.73-0.99). Only a check on what the answer claims can catch that. The gate
can't.

**Mechanism.** `--answer-mode quote`: `QUOTE_SYSTEM` and the `grammar.quoted_answer` grammar, so
every claim must open with a quotation from the extract it cites. `grammar.verify_quotes` then
checks each quote mechanically: whitespace-collapsed substring of the cited extract, no judge. If
any claim fails, the answer becomes the refusal before scoring. Retrieval, gate and k are R3's,
unchanged.

**Arm.** All 1,160 pool rows, generation only, from R3's cached retrieval
(`--stage generate --cache p11-pool`). The two arms therefore read identical extracts and differ only
in how the answer is produced. The generator runs in the same profile as R3, on :8083.

**Metrics** (paired per row against the R3 control):
- **unsupported answer** (primary): a spoken answer (not abstained) on an unanswerable row, or on
  an answerable row whose gold evidence was not retrieved. Evidence is identical in both arms (same
  cache), so this counts answers the extracts could not have supported.
- correctness (aliased), over all answerable rows.
- abstention on unanswerable rows.
- quote failure rate (answers converted to refusal by `verify_quotes`), reported by kind.
- generation time per question: a fixed sample of 100 pool rows (seed 11), cite vs quote, on the same
  server one after the other, after a warm-up.

**Decision rule (quote becomes the default answer mode only if all four hold):**
1. **Fewer unsupported answers:** paired, net reduction ≥ 10, McNemar p < 0.05.
2. **Correctness holds:** paired net over all 883 answerable rows ≥ −2. This is the same margin as
   R5, so a mode that refuses its way to safety cannot pass.
3. **No new invention:** abstention on the 277 unanswerable rows does not fall by 2 or more. It is
   expected to rise, which rule 1 already counts.
4. **Affordable:** generation p50 rises by ≤ 1 s per question.

If 1 and 3 pass but 2 fails, it is reported as "safer, at a cost of N answers". It does not ship as
the default. It could be offered as an opt-in (`asq --strict`), which would be the user's decision.
Reported whatever the verdict: how many of the 3 answerable rows the R3 control got right
**without** retrieved evidence (counted from `p11-pool-answers.json`) stay right under quote mode, since quote mode should kill parametric
answers (phase 9's failure), right or wrong.

**Secondary, diagnostic, not a ship rule:** the 35 rows R4d answered at tiers 1-2 (15 correct, 6
inventions, 14 wrong), re-run with `--cascade --answer-mode quote`. Question: does quote mode
remove the cascade's inventions and keep its rescues? A clear yes would make "cascade + quote" a
candidate arm with its own pre-registration.

## Exploratory: term menu from mechanically extracted cards (2026-09-28, not pre-registered)

**Idea (user's).** Collect each documented term with the doc's own one-line description (a "card").
At question time, the question's embedding pulls the nearest cards, and the 4B would pick from that
menu. It can only choose real names, which removes R4b/R4d's invented rewrites. The go/no-go asks
whether the right term is in the 30-card menu at all. **Bar, fixed before the first run:** ≥ 50 % on
no-name + synonym Emacs rows whose R3 evidence was missing.

**v1**: 9,709 cards from the Emacs manuals: `-- Command/Function/Option:` definitions, definition
lists, and inline `‘key’ (‘command’)` and `‘M-x name’`, first mention kept. All Emacs rows:
**15/34 = 44 %, fail.** Clean 31/44, no-name 26/44, synonym 25/44 in the top 30. Hits include
the motivating cases: "closing all the other split views except the one I'm in" →
`delete-other-windows` at rank 6; "look inside a document without any risk I might change it" →
`find-file-read-only` at rank 2. Of the 13 clean misses, 7 have answers that are not command names
(files, TRAMP prefixes, use-package keywords, an eshell command), so no card could exist. The rest
are extraction defects.

**Repair round, with a frozen split** (62 Emacs bases, seed 20260928, 31/31; variants follow their
base). 8 of the inspected misses fell in the test half, disclosed. v2 fixes: definitional cards
beat passing mentions, `M-x name` with arguments, bare keys dropped. **Dev half: v1 5/12 (42 %),
v2 4/12 (33 %).** v2 is no better anywhere, and neither passes on dev, so the test half was not
spent and stays clean.

**What it establishes.** Mechanical cards carry the manual's vocabulary, and the manual's vocabulary
is the gap. "The recording I just made" does not embed near "give a command name to the most
recently defined keyboard macro". The misses that remain are not extraction bugs; they are the
problem itself. Cleaner cards will not close it. Plain-English descriptions written for each card
might: the 4B writes them once, offline, so query-time cost is unchanged. That is the next version
of this check, and it runs against the untouched test half.
