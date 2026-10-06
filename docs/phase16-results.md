# Phase 16: answer quality, offline and cheap first

Source: `todays-plan.md` (2026-10-06), a ranked list of ideas. Each measured idea gets its
pre-registration here, written and committed before its run. Nothing below is measured yet.

## msr_p16-grounding pre-registration (written 2026-10-06, before the replay is run)

**Question (plan idea 2).** Does an answer that names an option, flag or Emacs key found in none of
the extracts the reader saw identify wrong answers well enough to be a check?

**Tool.** `scripts/grounding_report.py` (task `tsk_20261006_grounding`), replaying frozen answers
files against their frozen retrieval caches. It scores correctness only through
`screen_report.score_run` on `data/eval/gold_aliases_v2.json`, over the five eval files
`questions`, `emacs_questions`, `variations_typo`, `variations_man`, `variations_emacs`.

**Two patterns, both reported.**
- `draft` is the throwaway script's pattern. It matches plain hyphenated English (`null-separated`),
  so it is reported only to reproduce the draft numbers, and it decides nothing.
- `tight` is flags (`--long`, `-x`), Emacs key chords (`C-x`, `M-%`), the command name after `M-x`,
  and single-dash multi-letter flags only inside backticks. **Only `tight` decides.**

**Runs.** Primary: `p15-ctl:p15-ctlr` (the control; 1160 rows). Replication (different retrieval
pipelines, same 4B reader): `p15-r13`, `p15-r14a`, `p15-r14b`. Descriptive only: every other
answers file that has a cache, including the phase 9 30B runs; none of those enter the verdict.

**Rates.** For a bucket, rate = answered rows with at least one ungrounded identifier / answered rows.
Buckets: `correct`, `wrong`, `unanswerable-answered`.

**Decision rule (fixed before the run).** The `tight` check becomes a candidate for a grammar or
post-hoc filter iff, on the primary run:
1. `wrong` rate >= 3 x `correct` rate;
2. flagged `correct` rows <= 1% of answered-correct rows (at most 5 of 564);
3. flagged `wrong` + `unanswerable-answered` rows >= 20.

and on every replication run: `wrong` rate > `correct` rate. Otherwise the check is dropped.
Passing is not shipping: a candidate still needs a live arm (the check applied to answers, scored
through `screen_report.py`), pre-registered separately.

**Extract starts (no decision).** The same tool also counts top-5 extracts that open with a lowercase
fragment (`ord` > 0, first character alphanumeric and lowercase). It is reported to turn the plan's
draft figure into a tracked number, and it motivates idea 4 only as context.

## msr_p16-gate pre-registration (written 2026-10-06, before the fit is run)

**Question (plan idea 3).** Does a gate fit on recorded retrieval features, at each pipeline's own
operating point, beat that pipeline's single top-1 threshold on rows not used for fitting?

**Tool.** `scripts/gate_fit.py` (task `tsk_20261006_gatefit`).

**Split (fixed here, a function of the eval rows only).** Family key =
`variant_of or paraphrase_of or qid`; bucket = `sha256(key) mod 10`; buckets 0-6 are `fit`, 7-9 are
`held`. Variants of one question never straddle it. The held rows are touched once, for the verdict.

**Model.** Six features (top-1 rerank score, gap to the second, hits sharing the top-1 page, dense
distance, mean rerank score of the top 5, share of 4+ letter question words absent from the index
vocabulary `data/index/phase11.db.vocab.json`), standardised on fit rows, L2 logistic regression,
2000 full-batch steps at rate 0.1. Label: answerable and the gold passage retrieved. One model per
pipeline. No hyperparameter is tuned; the values are the tool's defaults and are fixed here.

**Operating point.** For every pipeline, the lowest threshold at which its refusals on unanswerable
fit rows (stored abstentions, plus rows the gate newly refuses) reach the control's refusals on those
rows at the control's own 0.65 gate. A replayed gate can only add refusals to the stored ones, since a
row refused at 0.65 has no stored answer. It is therefore always at least as strict as the 0.65 gate.

**Two gates per pipeline, same target.** The `calibrated` gate thresholds the logistic model's
probability. The `top1-matched` gate thresholds the top-1 score alone, the one feature the plain gate
uses (the matched-abstention diagnostic of phase 15, but with its threshold chosen on fit rows only).
The calibrated gate has to beat the top1-matched gate, not just the plain one: any retrieval change
that lowers abstention gains answers from a stricter gate.

**Runs.** Control `p15-ctl:p15-ctlr`. Arms `p15-r13`, `p15-r14a`, `p15-r14b` (own caches) and
`p15-r14c:p15-ctlr` (R14c reads the control's retrieval; the answers file names the `p13-ctl`
cache, which is not the cache of record, so the colon form is required). The L2 term is the tool's
`(1.0 / n_fit_rows) * w` on the weights, never the bias.

**Decision rule (fixed before the run), on held rows.** `net` and `abstention_net` are
`screen_report.compare` of a replayed arm against the control under its plain 0.65 gate.
- Per arm, `calibrated` PASSES iff `calibrated net` >= `top1-matched net` + 3 and
  `calibrated abstention_net` >= `top1-matched abstention_net`.
- For the control row, `calibrated` and `top1-matched` are both reported but neither decides: the
  control's target is already met by its stored refusals, so both equal the plain gate by construction.
- At least two of the four arms must PASS for the calibration to count as a candidate; one PASS or
  none closes it. A candidate still needs a live arm (its gate applied during generation, so rows
  below 0.65 can be answered and not only refused), pre-registered separately.

**Not decided here.** The final operating point of a shipped gate, and any gate looser than 0.65;
a live arm must pre-register them.

## msr_p16-grounding result

The tool reproduces the draft's figures exactly under the `draft` pattern on `p15-ctl` (cache
`p15-ctlr`): 44 of 564 correct, 27 of 163 wrong and 5 of 23 unanswerable-answered rows name an
identifier absent from the extracts, and 4273 of 5800 extracts open with a lowercase fragment
[chk_050]. So the draft numbers are now evidence, and they measure mostly English: `null-separated`,
`read-only` and similar.

Under the `tight` pattern, which decides, `p15-ctl` flags 5 of 564 correct, 8 of 163 wrong and 2 of 23
unanswerable-answered rows [chk_050]. Against the pre-registered rule:

| rule | needed | got | |
|---|---|---|---|
| 1. wrong rate >= 3 x correct rate | | 4.9% vs 0.9% (5.5x) | met |
| 2. flagged correct rows | <= 5 of 564 | 5 | met |
| 3. flagged wrong + unanswerable-answered rows | >= 20 | 10 | **not met** |
| replication: wrong rate > correct rate on r13, r14a, r14b | all three | all three [chk_051] | met |

Rule 3 fails: **the check is dropped.** It is precise but too rare to matter: it would refuse 15
answers: 5 correct, 8 wrong and 2 to unanswerable questions. The "stronger version" (identifiers inside the grammar) is not
pursued either, since the gap that motivated it came from hyphenated English. The mid-word extract
rate is unchanged across arms and motivates idea 4 only as context.

## msr_p16-gate result

Fit rows 827, held rows 333 (68 unanswerable), control target 195 of 209 unanswerable fit rows
refused. Held rows against the control under its plain gate, for each arm's `calibrated` and
`top1-matched` gate [chk_053]:

| arm | plain net | calibrated net / abstention_net | top1-matched net / abstention_net | rule |
|---|---|---|---|---|
| p15-r13 | +0 | -7 / +5 | -13 / +5 | pass |
| p15-r14a | -2 | -3 / -1 | -8 / +6 | fail (abstention) |
| p15-r14b | +7 | -1 / +4 | +1 / +4 | fail (net) |
| p15-r14c | +6 | +6 / -2 | +6 / -2 | fail (no change) |

One of four arms passes; two were required, so **the calibrated gate is closed.** The control's own
calibrated and top-1 gates equal its plain gate, as expected, since its stored refusals already meet
the target. What the table shows:
- Refusing more to match abstention is expensive for every arm: even the best matched gate
  (`p15-r13` calibrated) is 7 answers below the control, and no arm ends above it at matched
  abstention. That agrees with the phase 15 matched-abstention diagnostic.
- The model sorts rows better than the top-1 score alone for `p15-r13` (+6 over top-1 matched,
  same abstention), and not elsewhere. One arm is a hint, not a result.
- The replay can only add refusals. A gate looser than 0.65 needs a live run.

## Generation-only arms pre-registration (written 2026-10-06, before any arm is generated)

Plan ideas 4 and 5. Retrieval, embeddings, reranker scores and the gate are byte-identical across
all arms: every arm reads the control's cached retrieval (`p13-ctl`, identical to `p15-ctlr` on all
1160 rows) and differs only in what the reader is shown. Any abstention change can therefore come only
from the reader.

**Control.** `p15-ctl` (setting C, `SMM_GEN_ARGS="--parallel 1 --no-cache-prompt --cache-ram 0"`,
`SMM_GEN_PORT=8090`). The control is not regenerated: phase 15 showed its generation holds across days
(p13-ctl against p15-ctl), and arms are generated on the same day under the same setting.

**Arms (one mechanism each, never combined here).**
- **G1, fewer extracts:** `--read-k 3`. The reader sees the top 3 of the cached 5.
- **G2, reversed order:** `--read-order reverse` (task `tsk_20261006_readorder`). The reader sees all
  5 extracts, best last, i.e. nearest the question; citation numbers map to the reversed list.
- **G3, idea 4 (extract repair) and G4, idea 6 (worked examples in the reader prompt):** not built
  yet. Each needs its own build and its own pre-registration section, written before its run.

**Scoring.** `scripts/screen_report.py` with the five eval files, `--aliases
data/eval/gold_aliases_v2.json`, `--group variant_kind --group-default clean` and `--group domain
--group-default linux`.

**Decision rule, per arm** (phase 15's R14 rule, kept unchanged so results compare):
1. **Answers rise:** all 883 answerable rows net >= +8, sign test p < 0.05.
2. **Clean unharmed:** clean group net >= -2.
3. **No new invention:** abstention net over 277 unanswerable rows >= -1.

Failing 2 or 3 means no ship. Passing 1 with p >= 0.05 is "direction only". A pass is a candidate
default, not a ship: it still needs a confirmation arm on the realistic Emacs screen.

**Known limit, stated now.** Phase 8 found about half the changed outcomes flipped only because the
extracts' order changed, so rule 1 needs a net gain, not churn. G1 removes information (extracts 4
and 5); if the evidence is in extract 4 or 5 the arm loses that row, and its `lost` list says so.

## Generation-only arms result (run 2026-10-06)

Control `p15-ctl`; each arm generated under setting C on the control's cached retrieval; scored on
`gold_aliases_v2.json` over all 883 answerable and 277 unanswerable rows.

| arm | answerable correct | lost | gained | net | p | abstention net | verdict |
|---|---|---|---|---|---|---|---|
| G1 `--read-k 3` | 509 of 883 | 81 | 26 | -55 | 9.4e-8 | -4 | fail rules 1, 2, 3 |
| G2 `--read-order reverse` | 567 of 883 | 44 | 47 | +3 | 0.83 | +0 | fail rule 1 |

Registry entries: read-k arm [chk_056], reversed-order arm [chk_057]. The control is 564 of 883 correct, 711 with evidence, 254 of 277
abstained [chk_057].

- **G1 loses answers it cannot find.** Evidence the reader can see falls from 711 to 660 rows: the
  answer sits in extract 4 or 5 for 51 rows. The clean group alone loses 14 (p 0.004) and the
  synonym group 12. Phase 7's result runs the other way from the plan's hope: fewer extracts
  does not help this reader, and the extra extracts are where recall lives. **G1 is closed.**
- **G2 is churn.** Reversing the order flips 91 answers (44 lost, 47 gained), net +3, and
  changes nothing about evidence or abstention. That matches phase 8's finding that order alone moves
  answers about as much as a real change, and it puts a floor under every arm in this phase: a net
  under about +10 on 883 rows is indistinguishable from reordering. **G2 is closed as a way to
  gain answers.**
- **G2's churn is itself a signal.** Rows that flip with the order are rows the reader answers
  from luck. Idea 9 (ask twice in two orders, refuse when the named options differ) is the use of it, and
  both orders now exist for all 1160 rows (`p15-ctl`, `p16-g2`). Counting how often the two answers
  disagree and how well that separates right from wrong needs no GPU; it is the next offline check.

## Phase 16 status

| idea | verdict |
|---|---|
| 1 bracket grammar | built and merged; live check passed; full-pool run not yet done (c04 family) |
| 2 identifier grounding | dropped (rule 3) |
| 3 calibrated gate | closed (1 of 4 arms) |
| 4 extract repair | pre-registration pending its build |
| 5 order and count | G1 closed, G2 closed |
| 6-17 | not started |

## msr_p16-disagree result (idea 9, offline half)

`scripts/order_disagreement.py` compares `p15-ctl` (extracts in rank order) with `p16-g2` (the same
extracts reversed) on the 750 rows `p15-ctl` answered. Refusing every row whose two answers differ
(different identifiers, or, when neither names one, different text) would lose 280 of 564 correct
answers to catch 124 of 163 wrong ones and 19 of 23 unanswerable questions [chk_066]. That is two
correct answers lost per wrong one caught, so asking twice and refusing on disagreement is closed. A
looser definition (identifiers only, ignoring wording) is an inline, unverified draft that gave 138
correct lost against 58 wrong and 9 unanswerable caught, the same ratio; it is kept out of the
decision. The rule was not written before this run; the ratio fails any bar I would have set, so
the order of events does not change the verdict.

## msr_p16-dom pre-registration (idea 11, written 2026-10-06, before any run)

**Question.** Under an explicit domain (`--domain emacs`, the `asq.el` path), do the Emacs settings
that gained answers in phase 15 - the neutral embedder instruction (R14a) and the fine-tuned embedder
(R13) - gain them there too? With the domain given, the Emacs settings cannot cost man pages anything,
so only Emacs rows are measured.

**Rows.** The 432 Emacs rows of the pool (324 answerable, 108 unanswerable), listed in
`data/eval/results/p16-emacs-qids.json`, retrieved with `--domain emacs`.

**Arms, all the same shape (a 432-row subset run, since subset runs drift against full runs).**
- **control:** index `phase11.db`, default embedder instruction, `--domain emacs`.
- **A, instruction:** same, `--embed-task neutral`.
- **B, fine-tuned embedder:** `data/index/phase15-ft.db` with the fine-tuned query model, `--domain
  emacs`, default instruction.
Generation under setting C on each arm's own retrieval, gate 0.65.

**Decision rule per arm, on the 324 answerable Emacs rows:**
1. net >= +8, sign test p < 0.05;
2. clean group net >= -2;
3. abstention net over the 108 unanswerable rows >= -1.
An arm passing 1 and 2 but failing 3 is judged again at matched abstention (`screen_report.py
--match-abstention`): it is a candidate with a recalibrated gate iff matched net >= +8.
Anything else is closed. A passing arm becomes the `asq.el` setting only after a confirmation on the
realistic Emacs screen (`data/eval/realistic_emacs.jsonl`), pre-registered separately.

## Reader-side arms pre-registration (ideas 4, 6, 7, 10; written 2026-10-06, before any of these arms is generated)

Same frame as the G1/G2 pre-registration above: every arm reads the control's cached retrieval
(`p13-ctl`), is generated under setting C against the control `p15-ctl`, scored through
`screen_report.py` on `gold_aliases_v2.json` with `--group variant_kind --group-default clean`. Rules, per
arm, on all 883 answerable and 277 unanswerable rows:
1. net >= +8 with sign-test p < 0.05;
2. clean group net >= -2;
3. abstention net >= -1.

**Arms.**
- **G3, extract repair** (`--read-view repaired`): the reader sees extracts whose window start is
  completed from the previous window, with the option or heading line on top, and consecutive windows
  merged. Evidence, the gate and the retrieved list are the original hits.
- **G4, worked examples** (`--reader-prompt v3`): SYSTEM_V2 plus three worked examples built on manual
  pages that are the gold page of no eval question (head, tee, nl, paste, basename, fold).
- **G5, line mode** (`--answer-mode line`): the opening quotation must be an actual line of the cited
  extract (a per-question grammar). Extra rule 4: mean generation seconds per question <= control + 0.5.
  Judged on correctness, not support: a real line can be the wrong line.
- **G6, second read** (idea 10): `--read-k 2` generated for all rows (`p16-g6`), then
  `scripts/combine_answers.py` takes, for rows the control refused after the gate passed, with top score
  >= 0.9, the `p16-g6` answer. Scored as the arm `p16-s6` against `p15-ctl`. Rule 3 is the limit on
  new inventions: unanswerable rows above 0.9 that are answered count against it. The threshold 0.9 is
  the only value tried.

**Known limits.** G2 showed that reordering alone flips about 90 answers with net near zero, so rule 1
needs a net gain, not churn. G4 adds about 500 prompt tokens: its abstention is watched. G5's grammar
lists 75 to 150 literal lines per question; its sampling latency is measured, not assumed.

## Retrieval-side arms pre-registration (ideas 12, 13, 14; written 2026-10-06, before any of these arms is run)

Scoring as before: `screen_report.py`, `gold_aliases_v2.json`, `--group variant_kind --group-default clean`
and `--group domain --group-default linux`, `--match-abstention` reported for anything that changes
retrieval. Generation under setting C. Rules 1 to 3 are the R14 rules: (1) net >= +8 over the answerable
rows of the comparison with sign test p < 0.05; (2) clean group net >= -2; (3) abstention net >= -1.
Rule 4 is speed: mean retrieval seconds per question <= control + 0.5.

- **IX, the Emacs manual's concept index (idea 12).** `data/index/phase16-ix.db` = `phase11.db` plus
  one question vector per Concept Index entry, each pointing at the chunk its `(line N)` falls in
  (`scripts/build_ixvec.py`; entry text alone is embedded; only the Concept Index, because the command,
  key, variable and option indexes list names, not wordings). Retrieval with `--domain emacs
  --question-vectors 10`, on the 432-row Emacs subset, against the control `p16-dom-ctl` (same shape).
  Rule 4 is relaxed to control + 1.0 s, since R8's route cost +1.78 s on Emacs rows. If the share of
  entries that map to a chunk is below 0.9, the build refuses and nothing runs.
- **WSP, widen with the corrected question, rerank against the typed one (idea 13).** `--widen-spell`
  over the full 1160-row pool against the full-pool control `p15-ctl` (a full run of the same shape;
  subset runs drift). Extra rule 5, the name rule that failed R4a: among rows `p15-ctl` answered
  correctly, rows lost by the arm are at most 5, and every lost row is listed with the changed words.
- **HD, the section heading in the embedded text (idea 14).** `data/index/phase16-hd.db` =
  `phase11.db` re-embedded by `scripts/reembed_headings.py` (75,613 of 75,613 chunks verified against
  their source documents; windows and ids unchanged), full 1160-row pool against `p15-ctl`. The query
  side is unchanged. Window starts are not aligned (that is the reader-side repair's job).

Each is one arm of one mechanism and none is combined with another here. Passing is a candidate, not a
ship: the realistic Emacs screen (`data/eval/realistic_emacs.jsonl`) and a man-page cost check come
before any default changes.

## msr_p16-dom result (idea 11): the Emacs settings under an explicit domain

432 Emacs rows (324 answerable, 108 unanswerable), `--domain emacs`, all three runs the same shape.
The control answers 234 of 324 and abstains on 104 of 108. For reference, the same rows under the
open-domain pool control answered 183, so knowing the domain is worth about +51 on these rows by
itself.

| arm | correct | lost | gained | net | p | abstention net | verdict |
|---|---|---|---|---|---|---|---|
| A, neutral embedder instruction | 242 | 5 | 13 | +8 | 0.096 | +0 | direction only [chk_089] |
| B, fine-tuned embedder | 243 | 8 | 17 | +9 | 0.108 | +0 | direction only [chk_090] |

Both clear the size of rule 1 (net >= +8) and fail its significance (p < 0.05). Both leave abstention
and the clean group untouched, which is what the phase 15 arms could not do under the open-domain
pool: knowing the domain removes the abstention cost. Matched-abstention replays equal the plain
numbers, since abstention did not move. Between a third and a half of each arm's gains come from question families
with five or six variants (`e25`, `e07`, `e43`), so the rows are not independent and the sign test is
generous, not harsh. Neither arm becomes the `asq.el` setting on this evidence; both are candidates
for a larger Emacs set, which does not exist yet (the realistic Emacs screen has 80 rows).

## Reader-side arms, first result (G3)

**G3, extract repair** (`--read-view repaired`): net -6, 32 lost and 26 gained, p 0.51, abstention net -2,
clean group net -3 (registry entry [chk_088]). It fails rules 1, 2 and 3; closed. On real data the repair did what it
was built to do (a window opening mid-line now gets the line completed and the option line above it,
and 939 of 5800 extracts merged away), but the reader reads no better for it. The cause is not
established: merging cuts the number of extracts, and phase 7 and G1 show extracts matter.

## G5b pre-registration (written 2026-10-06 after G5, before the arm is generated)

G5 (`--answer-mode line`) lost 272 answers and gained 17 (registry entry [chk_092]). The answers show why: the 4B picks a
description sentence from the extract ("Display only lines which do NOT match the pattern.") and leaves
out the line that holds the option (`-v, --invert-match`). **G5b**, `--answer-mode line-id`, offers only
lines that name a flag, an Emacs key chord or an M-x command, so the chosen line carries the identifier.
It is a variation of the same idea, tried once. Same control, rules and scoring as G5 (rules 1 to 4);
no further variation follows whatever it shows.

## Reader-side arms result (G4, G5, G6/S6; run 2026-10-06)

Control `p15-ctl`, same retrieval, setting C, 883 answerable and 277 unanswerable rows.

| arm | correct | lost | gained | net | p | clean net | abstention net | verdict |
|---|---|---|---|---|---|---|---|---|
| G3 extract repair | 558 | 32 | 26 | -6 | 0.51 | -3 | -2 | fails 1, 2, 3 (chk_088) |
| G4 worked examples (`--reader-prompt v3`) | 588 | 29 | 53 | +24 | 0.011 | +3 | -6 | fails rule 3 (chk_091) |
| G5 line mode | 309 | 272 | 17 | -255 | 2.6e-60 | -46 | +5 | fails rule 1 and 2 (chk_092) |
| G6 `--read-k 2`, alone | 482 | 117 | 35 | -82 | 1.6e-11 | -20 | -4 | closed alone (chk_096) |
| S6 second read: G6 on rows refused above 0.9 | 576 | 0 | 12 | +12 | 0.0005 | +2 | -7 | fails rule 3 (chk_097) |

- **G4 is the strongest reader result of the phase and fails the same rule as the phase 15 reader
  arm.** It gains answers broadly (terse +9, p 0.023; Emacs +14, p 0.0066) and loses abstention: six
  more unanswerable rows are answered. At the control's abstention (gate 0.8445) its net is -5, so the
  gain comes from letting more rows through, not from reading them better. Worked examples do
  what the one-sentence prompt of phase 15 did, only more strongly. Under an explicit domain, where
  the dom arms showed that abstention cost vanishes, the arm might pass; that is a separate
  pre-registration, not a reading of this table.
- **G5 closes line mode.** The answers show the 4B choosing a description sentence and leaving out the
  line that holds the option (`-v, --invert-match`). It costs +0.1 s per question, so the grammar's
  size is not the problem; the choice is. G5b below tries lines that carry an identifier, once.
- **S6, the second read, gains 12 answers and loses none (p 0.0005), and fails only the abstention
  rule:** among 148 rows replaced, the 4B answers 7 unanswerable questions it had refused. The threshold
  0.9 was the only one pre-registered, so no other value is tried; a higher threshold is the obvious
  next knob and needs its own pre-registration.
