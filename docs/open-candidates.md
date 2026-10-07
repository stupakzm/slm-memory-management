# Open candidates after phase 16 (written 2026-10-07)

Everything in `todays-plan.md` was taken as far as local hardware allows (`docs/phase16-results.md`). Nothing
ships as a default. What is left is below, in the order I would do it. Each item says what it needs, what it
costs, and the one rule it must be judged by, written here so the pre-registration starts from it, not from
the result. Nothing in this file is measured.

## 0. The pattern behind all of it

Every reader or retrieval change that gained answers on the mixed pool also answered more unanswerable
questions (worked examples +24 answers, abstention -6; second read +12, abstention -7; the phase 15
retrieval arms +19 to +25, abstention -5 to -6). Under an explicit domain that cost vanished (per-domain Emacs
arms: abstention +0). So two directions remain: make the gain survive without the invention cost (items 3, 5,
7), or only apply it where the domain is known (items 1, 2, 4).

## 1. A bigger Emacs evaluation set (prerequisite for 2, 3 and 4)

- **Why:** the per-domain arms are +8 and +9 with p 0.096 and 0.108 on 324 answerable Emacs rows, and
  between a third and a half of each arm's gains come from question families with five or six variants. The
  realistic Emacs screen has 80 rows. A decision needs more independent questions, not more variants.
- **What:** at least 300 independent Emacs questions (families, not variants), answerable and unanswerable,
  written in everyday wording and with typos and terse forms as extra rows only, labelled the way phase 12
  and 14 did it: one-token labels plus blind-adjudicated alternatives (`orch-judge` through
  `scripts/judge_pack.py`; see the judge's reliability caveat in item 8).
- **Cost:** question writing (the 4B or the 30B drafting, a person or a blind judge checking), no GPU beyond
  generation. Split families between a dev set and a held-out set before any arm is run.
- **Rule:** none; it is an instrument. Check it first against the existing control: the control must score
  near 234/324-equivalent, and label noise on a 60-row sample must be below the audit's 20% bar.

## 2. Re-test the per-domain Emacs arms on that set

- **Arms:** A, `--embed-task neutral`, and B, the fine-tuned embedder, each under `--domain emacs`, one
  mechanism each, never combined here.
- **Rule:** the R14 rules (net >= +8 and p < 0.05, clean net >= -2, abstention net >= -1) on the held-out
  families. A pass makes the setting the `asq.el` default; man pages are untouched because the domain is
  explicit.
- **Cost:** about 25 min retrieval plus 8 min generation per arm per 432 rows; scales with the new set.

## 3. The worked-examples prompt (v3) under an explicit domain

- **Why:** G4 gained +24 overall and +14 on Emacs rows (p 0.0066) and lost 6 on abstention. Under an
  explicit domain the abstention cost may not appear, as it did not for the retrieval arms.
- **Arm:** `--reader-prompt v3 --domain emacs` on the item-1 set, control the same run with the default
  prompt. Also try a variant with two refusal examples instead of one, because the arm's cost is
  inventions, and a prompt that shows refusing twice might cut them; one variant, pre-registered, no more.
- **Rule:** R14 rules on the held-out families.

## 4. The `c04` bracket arm on the full pool (idea 1, closing it)

- **Why:** the grammar now admits `[:alpha:]`-style answers (live check: `[:alpha:] [1]`), but the full
  pool was never run. Seven `c04` rows have gold `[:alpha:]`.
- **Arm:** default grammar, `p13-ctl` retrieval, setting C, 25 min.
- **Rule (the usual +8 is impossible with seven rows):** the `c04` family gains at least 3 correct answers;
  no other row loses more than 2 correct answers in total; abstention net >= -1.

## 5. The second read at a higher threshold

- **Why:** S6 (rows the control refused above gate score 0.9, answered again from the top two extracts) gained 12
  and lost 0 (p 0.0005) but answered 7 unanswerable questions. Only 0.9 was pre-registered, so no other value
  was tried.
- **What:** split the pool into a dev half and a held-out half by family (the `gate_fit.py` split rule), pick the
  threshold from {0.95, 0.97, 0.99} on the dev half by the rule below, freeze it, judge once on the held-out
  half. `p16-g6` already holds the read-k 2 answers for every row, so this needs no GPU:
  `scripts/combine_answers.py --min-score T`.
- **Rule:** held-out net >= +8 and abstention net >= -1. If no threshold passes on dev, close it.

## 6. RAFT fine-tune of the reader (idea 15), end to end

- **State:** `scripts/build_raft_data.py` (plan, teach, assemble), `scripts/train_reader.py` (LoRA; fp16 and
  4-bit options) and `docs/train-reader-free.md` (Colab free, Kaggle, the local 6 GB card) are built and
  tested on their pure parts. The torch half has never been run.
- **Steps:** (a) run `plan` and check the eligible-pair counts per domain (about 11,600 Emacs and 54,300 man
  chunks remain after excluding every eval gold page); (b) `teach` a 1,500 to 2,000 example pilot on the local
  30B (about 8 s per example, 3.5 to 4.5 h); (c) `assemble`, read 30 examples by eye; (d) 5 to 10 training
  steps on a 20 example file; (e) the real run; (f) convert to GGUF, Q4_K_M, serve as the reader; (g) the
  pre-registered generation arm against `p15-ctl` with the R14 rules and a judged pass.
- **Rule:** the R14 rules. A stronger bar for shipping: abstention net >= 0, because the training data
  includes refusals on purpose.

## 7. Cut the invention cost directly

- **Idea:** a refusal check that asks whether the cited extract answers this question, not whether it supports
  the claim (phase 6's lesson), applied only to rows the reader answered while the gate score was below 0.9.
  Candidates: the 30B as a yes/no judge on those rows only (about 300 rows, no new training).
- **Rule:** on G4's answers, the check must refuse at least 4 of the 6 extra inventions while losing at most 4
  of G4's 53 gained rows; measured offline from `p16-g4-answers.json`.

## 8. Judge reliability

The blind judge met its reliability bar exactly (17 of 20 hidden controls). Before leaning on judged numbers
again: 30 hidden controls, including known-wrong answers (not only known-correct), two passes with the
item order shuffled differently, and report the agreement between passes. UNCLEAR was 19 of 82 items for
the worked-examples arm; decide beforehand how UNCLEAR rows count.

## 9. Older threads still open (from earlier phases, not touched in phase 16)

- Typo repair that does not rename tools needs a context signal (three rules failed in phase 11; phase 16's
  widen-with-corrected-question arm also closed).
- SEE ALSO traversal; `servers.sh wait_ready` waiting 120 s on a dying server.
- Quote mode (phase 10) only if the 4B stops mis-copying its own quotes (27% of answers).
- The cause of batched-versus-single reranker scoring differences (per-slot context 2048 versus 8192) is untested;
  it also explains why a full-pool arm that changes request composition can flip rows that were not changed
  (phase 16: 12 of the widen-spell arm's 17 lost rows were not widened).
