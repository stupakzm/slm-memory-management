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
