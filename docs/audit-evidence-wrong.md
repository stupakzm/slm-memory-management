# Audit: are the pool's "wrong with evidence" answers really wrong? (pre-registered 2026-10-03)

## Why

In the open-domain control `p13-ctl`, 195 of 883 answerable rows have the gold passage in the top 5 and
are still scored wrong (another 73 abstain). That is more than the 172 rows where retrieval missed.
Phase 12 showed that one-token labels under-credit correct alternatives. Before spending GPU on the
reader, this audit measures how much of the 195 is label noise and how much is real reading error.

## Sample (built and committed before any judgment)

- **Rows:** 60 of the 195 wrong-with-evidence rows and 20 of the 443 correct-with-evidence rows
  (controls), drawn with `random.Random(15)`, shuffled together and renamed `A01`–`A80`.
- **What the judge sees** (`data/eval/audits/p13-ctl-evidence-audit-view.jsonl`): the question, the
  model's answer, the reference answer token, and the gold section text. Nothing else: no qid, no
  category, no score.
- **The key** (`…-audit-key.jsonl`) maps each id to its qid and category. The judge never reads it.

## Judgment (per item, by a judge agent that reads only the view file)

- **CORRECT:** the answer gives a command, option or key that does what the question asks, according
  to the gold section text, even if it is not the reference token. Extra correct detail is fine.
- **WRONG:** it does not do what was asked, or it states something the text contradicts.
- **UNCLEAR:** the text provided cannot settle it.

## Decision rule

- **Judge reliability first:** the judge must call ≥ 17 of the 20 controls CORRECT. Otherwise the
  audit is inconclusive and nothing below applies.
- **Under-credit rate u** = CORRECT ÷ 60 on the wrong-with-evidence sample, reported with a 95 %
  Wilson interval and scaled to the 195.
- **If u ≥ 20 %** (≥ 12 of 60), labels are a first-order problem: the next step is blind alternative
  labels for the pool (phase 12's procedure), before any reader experiment. Paired comparisons stay
  unbiased, but they are noisier than they look.
- **If u < 20 %**, the reader is the real bottleneck, and reader experiments come next (fewer
  passages, escalation to the 30B on hard cases).

Labels are not changed by this audit. It measures; any relabelling is a separate, blind task.

## Audit 1 is VOID: the sample came from the wrong population (found 2026-10-03, before any conclusion)

The sampling code scored rows with `gold.aliases_for(aliases, qid, None)`, without the variant→base
fallback the real scorer (`screen_report.score_run`) uses. Typo and paraphrase variants therefore lost
their aliases, and 95 rows the real scorer credits were counted as "wrong". The true counts for
`p13-ctl`'s 883 answerable rows:
- 537 correct with evidence
- **101** wrong with evidence (not 195)
- 73 abstained with evidence
- 169 with no evidence (86 wrong, 83 abstained)

33 of audit 1's 60 "wrong" items were in fact credited, so its 37/60 CORRECT measures nothing about the
real wrong set. Audit 1's files are kept as a record (`…-audit-{view,key,judgments}.jsonl`), and its
result is not used. One finding from it stands on its own: 2 of its 20 controls (credited by the
scorer) were judged WRONG on reading. The scorer also over-credits sometimes: the token is present,
but the answer is wrong.

**The premise was also wrong.** Reading failures (101 wrong + 73 abstained = 174) and retrieval misses
(169) are about even. The reader is *not* the dominant bottleneck.

## Audit 2 (same criteria, judge instructions and decision rule as above, corrected population)

The sample is 60 of the true 101 wrong-with-evidence rows and 20 of the 537 correct-with-evidence
controls, drawn with `random.Random(17)` from the real scorer's output, shuffled and renamed
`B01`–`B80`. Gold text is now cut at 12,000 characters (audit 1's 6,000 hid the decisive entry in 3
items). The rule is unchanged: controls ≥ 17/20, then u ≥ 20 % means relabel first. With 60 of 101
sampled, u scales to the 101.

## Audit 2 result (2026-10-03)

- **Judge reliability:** 19 of 20 controls CORRECT. That passes (≥ 17). The one control called WRONG, B25,
  uses `cp --preserve=timestamps,permissions`, where the attribute is called `mode`: the scorer credited
  a wrong answer.
- **Under-credit:** of the 60 truly-wrong-with-evidence answers, **32 CORRECT**, 25 WRONG, 3 UNCLEAR. So
  **u = 53 %** (Wilson 95 %: 41–65 %), about **54 of the 101** (41–66).
- **Strict reading:** the judge marked 11 answers CORRECT on knowledge beyond the gold text (e.g.
  `sshd -p`, `du -h`, `git-blame`). Recounting all of them as UNCLEAR gives 23/60 = 38 % (27–51 %),
  still above the bar.
- **By domain:** man pages 25 of 46 CORRECT; Emacs 7 of 14.

**Decision (pre-registered rule): u ≥ 20 %, so labels are a first-order problem.** About half the
"wrong with evidence" answers are right. The project's true answer rate on the pool is closer to
(540 + ~54)/883 ≈ 67 % than to the scored 61 %. Before any reader experiment, the pool needs blind
alternative labels, built by phase 12's procedure and extended to man pages.

What this does and does not change:
- **Paired comparisons are unbiased by under-credit,** because every arm is scored by the same labels.
  But they are noisier, and a gain made of correct alternatives is invisible. Phases 11 to 14's
  verdicts stand as measured.
- **Some "correct" answers are correct only for another domain** (e.g. `git-blame` for an Emacs
  question that does not say Emacs). That is domain ambiguity again, and labels cannot settle it.
- **The scorer also over-credits** (B25, and audit 1's A04/A38: the token is present, the answer is
  wrong). That error runs in the other direction and is smaller (1 of 20 controls).
