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
