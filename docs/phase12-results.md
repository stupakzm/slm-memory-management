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

## Re-screen pre-registration (2026-10-02, before any label or run)

The screen's verdict asked for two fixes to the set. Both are fixed here **before** any new label is
written or any arm re-run, and are applied identically to control, R8 and R8c.

**Fix 1: documented alternatives (labels).** A purely mechanical rule cannot do this. Of the five
under-credited control answers named above, one is the gold command's own key, documented in prose the
keybinding rule does not parse (`C-x C-+`). The other four are *different commands* in the same gold
section, and "accept any command in the gold section" would also credit wrong neighbours (`find-file`
for "stop me changing files I only want to read"). So:
- `scripts/derive_gold_aliases.py --alternatives FILE` (tsk_20261002_altrule) merges adjudicated
  alternatives as rule `alternative`, and adds each alternative command's documented keys mechanically
  (`alternative-key`). `--check` enforces: every alternative is quoted verbatim from a line of the row's
  own gold section, is not in the question text, and is command- or key-shaped.
- `data/eval/realistic_alternatives.json` (tsk_20261002_altadj) is written by a fresh agent from the
  question and its gold section(s) only. It does not open `data/eval/results/` or this document, and its
  instructions name no example. It considers all 80 answerable rows and gives one reason per alternative.
  An alternative counts only if it does what the asker asked, not a neighbouring task.

**Fix 2: domain.** The re-screen runs with `--domain emacs` in every arm. The set is an Emacs set. Its
20 unanswerable rows are Emacs tool-not-installed questions, so they are unaffected in kind. This
measures what R8c changes *within* Emacs. Whether open-domain competition is acceptable is a separate
question for the full pool, which keeps every domain.

**Runs.** Retrieval and generation for control (`phase11.db`), R8 (`phase11-qx.db --question-vectors
30`) and R8c (`phase11-qc.db`), all with `--domain emacs`, full profile, generation under setting C,
named `p12b-{ctl,r8,r8c}`. Scored with the merged aliases.

**Rule: unchanged from the first screen.** An arm advances to the full-pool confirmation only if, on
the 100 rows paired against `p12b-ctl`: answerable net ≥ +4, lost ≤ 1, abstention net ≥ 0. Retrieval
p50 is reported. The confirmation and its five rules are as pre-registered above.

**Informational, decides nothing:** the existing open-domain `p12-*` answers rescored with the new
aliases (`rescore_answers.py`). This shows how much of the 23/80 was label under-credit.

**Stated now:** this is a second screen on the same 100 rows, run after R8c came up one short of the
bar. Only two things protect it from being a re-roll until something passes: the labels are fixed
blind to every answer, and every arm is changed in the same way. If R8c passes by exactly +4, that is
weak evidence, and the full-pool run is what decides. There is no third screen on this set.

## Re-screen results (run 2026-10-02)

Labels: `data/eval/realistic_alternatives.json` (tsk_20261002_altadj). 25 blind-adjudicated
alternatives over 22 rows, plus 7 derived keys, merged into `gold_aliases.json`. The blind adjudicator
accepted three of the five under-credited answers named above (`string-insert-rectangle`, `C-x C-+`,
`find-file-read-only`). It rejected `rename-uniquely` ("frees the name, does not give another name") and
did not list `C-x C-k n`. Those judgments stand: labels are not revised after seeing answers. Scorer:
`scripts/screen_report.py` (tsk_20261002_screen), which reproduces the first screen's table byte for byte
from the committed `p12-*` answers with the old aliases. Retrieval: full profile, `--domain emacs`.
Generation: setting C. The generator ran on :8090 because an unrelated process holds :8080; the port
does not affect outputs.

| arm | correct /80 | lost | gained | **net** | p | evidence /80 | abstention /20 | retrieval wall /100 q |
|---|---|---|---|---|---|---|---|---|
| control `p12b-ctl` | 35 | – | – | – | – | 62 | 20 | 356 s |
| R8 `p12b-r8` | 36 | 1 (er79) | 2 | **+1** | 1 | 60 | 20 | 468 s |
| R8c `p12b-r8c` | 35 | 2 (er02, er79) | 2 | **+0** | 1 | 61 | 20 | 346 s |

**Rule (net ≥ +4, lost ≤ 1, abstention held): neither arm advances.** As pre-registered, there is no
third screen on this set, and R8 and R8c do not go to the full pool.

**Informational (decides nothing), and the finding that matters:**

| comparison (paired, same 100 rows, new labels) | correct | net | p | evidence |
|---|---|---|---|---|
| first-screen answers, open domain: control `p12-ctl` | 28 | – | – | 49 |
| … R8 `p12-r8` vs that control | 32 | +4 (1 lost) | 0.22 | 53 |
| … R8c `p12-r8c` vs that control | 31 | +3 (1 lost) | 0.375 | 53 |
| **domain known: `p12b-ctl` vs `p12-ctl`** | **35** | **+7** (9 gained, 2 lost) | 0.065 | **62** |

- **Label under-credit was real but modest.** Blind alternatives lift the open-domain control from 23 to
  28 /80. They change no verdict on their own.
- **Question vectors were mostly a domain signal.** In the open domain, R8 and R8c gain +3 to +4, largely
  by pulling Emacs passages above man pages. Once retrieval knows the domain, they add +1 and +0. The
  domain alone is worth +7 net and +13 evidence on realistic Emacs questions. That is the largest effect
  this phase has seen, though at p = 0.065 on one 100-row set it is not yet a confirmed one.
- **What it does not show:** every question here is an Emacs question, so `--domain emacs` is an oracle.
  A real asker's domain has to be inferred. The open question for the next phase is how much of the +7 a
  *router* can recover without costing the man-page rows. Only the full pool, which mixes both domains,
  can measure that.

**Verdict.** R8 and R8c are closed: neither is worth its build or latency once the domain is known.
`phase11-qx.db` and `phase11-qc.db` stay on disk until the router experiment decides whether a domain
signal is needed at all. Next: pre-register a domain-routing experiment on the full pool.

Frozen outputs (committed): `p12b-{ctl,r8,r8c}-answers.json`, sha256 `e2af0abb…f93`, `88d19bcd…8fd`,
`335b443d…645`. Noise: setting C, measured deterministic at 40/40 ×3 (2026-09-29).
