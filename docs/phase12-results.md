# Phase 12: a fast realistic screen, and question-folded chunk vectors (R8c)

Status: **pre-registered 2026-09-29, before the question set exists and before any run.**

## Why

Phase 11's 1,160-row pool takes about 90 minutes of retrieval per arm, and most of its rows are
mechanical variants of 160 bases. R8 (generated questions as extra index vectors) was the only
mechanism with no measured side cost. It missed significance by one row and missed latency, because
the extra candidates must be reranked. R8c keeps R8's questions but folds them into each chunk's
**own** vector, so the candidate count and latency are unchanged by construction. This phase adds a
small, realistic screen so ideas like this can be triaged in about 10 minutes.

## The screen set (built next, under these constraints)

`data/eval/realistic_emacs.jsonl`: 100 new Emacs questions. 80 answerable, 20 unanswerable
(tool-not-installed, as in `emacs_questions.jsonl`). Written the way people ask: the need described,
the command name usually unknown. Styles: no-name 25, casual 20, synonym 15, terse 10, typo 10
(answerable). Every answerable row asks for a need **none** of the 62 existing Emacs bases covers (no
shared answer token). Labels are one answer token and a gold section, validated by `resolve_gold.py`
(leak check on), with keybinding aliases derived by `derive_gold_aliases.py --merge`. **Blind to R8's
generated questions:** any question with token Jaccard ≥ 0.5 to any of the 57,138 generated questions
is dropped and rewritten. 15 random answerable rows are hand-audited (the gold section answers the
question as asked) before any arm runs.

## Arms (all on the 100 rows, full-profile retrieval, generation under setting C)

- **control**: R3's configuration on `phase11.db`.
- **R8**: `phase11-qx.db --question-vectors 30` (reference; already known to fail latency).
- **R8c**: `phase11-qc.db`, a copy of `phase11.db` whose Emacs chunk vectors are re-embedded from
  `prefix + text + "\n\nQuestions this passage answers:\n" + the chunk's 3 R8 questions`. The stored
  `text` the reranker and reader see is unchanged, and so is the candidate count (50). Man-page vectors
  are untouched.

## Screen rule (per arm, paired against control)

An arm **advances to a full-pool confirmation run** only if all three hold on the 100 rows:
1. answerable correctness net ≥ +4;
2. answerable correctness lost ≤ 1;
3. abstention on the 20 unanswerable rows: net ≥ 0.

Reported alongside: evidence@5, retrieval p50 over the 100 rows (control vs arm, interleaved, after a
warm-up), and losses by qid. The screen proves nothing on its own. With 80 answerable rows, a +4 net
gain with no losses is p = 0.125. It decides only what earns the 90-minute run.

**Confirmation (only for an arm that advances):** the 1,160-row pool, R8's five rules unchanged.
Rule 5 (retrieval p50 +≤ 1.5 s) is expected to hold trivially for R8c, and is measured anyway. The
control is `p11-pool-r3d`.

## Known limits, stated now

- The screen set is written by a model (Claude), against the manuals, as the phase 11 variants were.
  "Realistic" is the author's judgment, not observed user logs.
- R8c's chunk vectors move away from the manual's wording, which may cost questions phrased in that
  wording. The screen has few such rows, so any cost there shows only in the confirmation run
  (clean rule).
