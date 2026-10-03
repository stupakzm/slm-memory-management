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
