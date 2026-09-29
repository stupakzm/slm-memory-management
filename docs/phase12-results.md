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

## Screen results (run 2026-09-29)

Set: `data/eval/realistic_emacs.jsonl` (9f2abf2). A 15-row label audit passed before any run, with two
caveats: er59 is under-credited for `M-x holidays`, and er57's token is a substring of a correct variant.
Retrieval in the full profile (`p12-{ctl,r8,r8c}-retrieved.json`). Generation under setting C
(`p12-*-answers.json`).

| arm | correct /80 | lost | gained | **net** | p | evidence /80 (lost/gained) | abstention /20 | retrieval p50 |
|---|---|---|---|---|---|---|---|---|
| control | 23 | – | – | – | – | 49 | 20 | 3.35 s |
| R8 (M = 30) | 26 | 1 (er37) | 4 | **+3** | 0.375 | 53 (1/5) | 20 | (+1.78 s in phase 11) |
| R8c (folded) | 26 | 1 (er79) | 4 | **+3** | 0.375 | 53 (1/5) | 20 | **3.17 s** |

**Screen rule** (net ≥ +4, lost ≤ 1, abstention held): **neither arm advances.** Both are +3 with 1 loss,
one gain short. R8c does what it was built for on cost: retrieval p50 is 3.17 s against 3.35 s for the
control, with no extra candidates. Its benefit matches R8's (+3 answers, evidence +4) at no latency.
Gains by style: R8 no-name +3, synonym +1, casual −1. R8c synonym +2, no-name +1.

**What the screen found that matters more than either arm.** Realistic questions are far harder than
the phase 11 pool. The control answers 23/80 (29 %; the pool's clean Emacs rows: 37/44). It retrieves
the right page for 49/80, so **26 rows fail at reading with the evidence in hand.** A look at 14 of
them:
- **Domain ambiguity.** Most realistic questions don't say "Emacs". The man pages then compete and
  often win on a question they genuinely answer: `uniq` for "remove repeated lines",
  `git-stripspace` for "strip trailing spaces". Some answers are just wrong (`nsswitch.conf` for
  "reload a file when another program changes it").
- **Label under-credit.** At least 5 of the 14 are correct alternatives the one-token label doesn't
  accept: `string-insert-rectangle`, `C-x C-+` (the key for `text-scale-adjust`, not in the derived
  aliases), `find-file-read-only`, `rename-uniquely`, `C-x C-k n`. Paired comparisons are unbiased by
  this, but noisier.

**Verdict:** neither arm goes to the full pool. The screen set needs two fixes before it can screen
well:
1. accept documented alternative commands as aliases (a derivation rule, not hand edits);
2. either scope the screen to `--domain emacs` or accept man-page answers where they're genuinely
   right.
R8c is the cheaper form of R8, with equal benefit on this set, and is the arm to carry forward.
