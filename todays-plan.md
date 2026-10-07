# Status 2026-10-07: every idea in this file has been taken as far as local hardware allows

Details, pre-registrations and registry entries: `docs/phase16-results.md`. Nothing here ships as a
global default. "Closed" means its pre-registered rule failed.

| idea | outcome |
|---|---|
| 1 brackets in the grammar | built and merged; live check ok; full-pool c04 run still to do |
| 2 identifier grounding | closed: too rare (10 of 20 needed) |
| 3 calibrated gate | closed: 1 of 4 arms beat the top-1 baseline |
| 4 extract repair | closed: net -6 |
| 5 order and number of extracts | closed: read-k 3 net -55, reversed order +3 (churn) |
| 6 worked examples | strongest reader arm (+24, p .011) but 6 more inventions: fails abstention; judged net +28 |
| 7 line grammar | closed: -255, and line-id -282 |
| 8 less compressed reader | closed: Q5/Q6/Q8 net -3/+2/+9 (p .27 at best), abstention -4/-5/-3 |
| 9 ask twice, refuse on disagreement | closed: loses 2 correct per wrong caught |
| 10 second read on strong-evidence refusals | +12, 0 lost, but 7 more inventions: fails abstention |
| 11 per-domain settings | Emacs arms +8 / +9 (p .10 / .11) under an explicit domain, no abstention cost: candidates, need a larger Emacs set |
| 12 Emacs manual index | closed: null (20 of 432 lists changed, 0 labels) |
| 13 widen with the corrected question | closed: net +3 |
| 14 heading in the embedded text | closed: net -5, evidence identical |
| 15 fine-tune the reader (RAFT LoRA) | data builder and training script built, not run: needs a GPU; free-hardware runbook in `docs/train-reader-free.md` |
| 16 judge the differing rows | done: judged nets G4 +28, Emacs A +10, B +7, second read +11; 17 of 20 controls |
| 17 evidence line under the answer | built and merged (`asq` prints it; `--no-evidence` turns it off) |

---

# Today's plan (2026-10-06): ways to improve answer quality

A ranked list of everything worth trying next, why, and how. **Nothing here is measured yet.**
Each item needs its own pre-registration before its run, and every code change goes through
`orch-task`.

> **Draft numbers.** Two figures below (marked *draft*) came from a throwaway script on `p15-ctl`,
> scored on `gold_aliases_v2.json`. No tracked code prints them yet, so they are not evidence. Per
> `CLAUDE.md` they stay out of `docs/` and decisions until committed code reproduces them. Step 1
> of "Today" makes them tracked.

## Where answers are lost now

In the open-domain control `p13-ctl`, 883 rows have an answer in the corpus
(`docs/audit-evidence-wrong.md`):

| outcome | rows |
|---|---|
| correct | 537 |
| **wrong, gold passage retrieved** (about half are really right; the labels miss them) | 101 |
| **"I don't know", gold passage retrieved** | 73 |
| gold passage never retrieved | 169 |

So about half the losses happen in reading and half in finding. Phases 7–9 show the 4B reads badly:

- With 8 extracts instead of 5, it was right less often when it had the evidence (84.4% → 80.2%).
- Of 17 changed outcomes in phase 8, about 8 flipped only because the other four extracts were in a
  different order.
- One typo makes it misread a page it was given: 12 of 18 typo1 losses had the right page.

Most ideas below target reading, the cheaper half to fix.

---

## A. No GPU needed: replay files we already have

### 1. Let the answer grammar contain brackets
- **Why:** `src/smm/grammar.py` allows answer text `[^\[\]\n]+`, so an answer can never contain `[`
  or `]`. The `c04` family (7 rows, gold `[:alpha:]`) cannot be answered by construction. Shell
  patterns, regex classes and some Emacs keys are blocked the same way.
- **How:** Allow `[` and `]` in text, and treat only `[` + digit 1–9 + `]` as a citation.
  `CITE_RE` already matches only `[1-9]`, so reading citations back needs no change.
- **Cost:** a few lines plus a grammar test; one generation run to confirm.

### 2. Check that options named in the answer appear in the extracts
- **Why:** *Draft:* an option or key missing from all five extracts appears in 8% of correct
  answers (44/564), 17% of wrong ones (27/163) and 22% of answered unanswerable rows (5/23). That
  is a 2–3× difference, from a mechanical check. Phase 9's open question was how to forbid
  ungrounded answers: the 30B gained +10 by answering from memory (`htop`, `strace`).
- **How:**
  - Tighten the pattern to real flags (`-x`, `--long`) and Emacs keys (`C-x …`, `M-x …`). The draft
    pattern also matched plain hyphenated English (`null-separated`, `colon-separated`), which is
    most of the hits on correct answers.
  - The draft method: identifiers taken from the answer text with citations stripped, each checked
    as a substring of `prefix + text` of the top 5 hits in `p15-ctlr-retrieved.json`.
  - Replay it over every existing answers file, including phase 9's 30B runs. Count wrong answers
    caught against correct answers lost.
  - Stronger version, in the grammar: options go in backticks, and only identifiers that occur in
    the extracts are allowed there.
- **Cost:** none on the GPU for the replay.

### 3. Replace the single-threshold gate with a calibrated one
- **Why:** In phase 15 every retrieval change gained answers (R13 +19, R14a +21, R14b +25) and
  failed only on abstention (−5 to −6). The 0.65 threshold was tuned for one setup
  (`docs/phase15-results.md`, matched-abstention diagnostic).
- **How:**
  - Fit a small logistic model on features the retrieval caches already record: top-1 score, gap to
    top-2, how many of the top 5 share a page, domain, dense distance, and the share of question
    words not in the index vocabulary (a typo signal).
  - Fit on a dev split, freeze it, judge it once on held-out rows.
  - It replays recorded scores, so it needs no GPU. It also lets R13, R14a and R14b each be judged
    at their own calibrated gate. This generalises phase 15's next step 2.

## B. Generation only: about 20 min per full-pool run, retrieval cache reused

### 4. Repair the extracts before the model reads them
- **Why:** *Draft:* 4,273 of 5,800 top-5 extracts (74%) start with a lowercase fragment, such as
  `pposed to a simple grep regex`. `split_text` in `src/smm/chunk.py` aligns a window's end to a
  newline but not its start. A window that opens inside an option's description has lost the
  option's name, which fits the "picked the neighbouring option" errors (`sort -g` for `-n`,
  `C-M-%` for `query-replace`). That link is not checked.
- **How:**
  - The index stores `char_start`, and `flatten()` reproduces the page text with headings inline.
    Use them to put the section heading and the option line above each window's start, and to drop
    the leading word fragment.
  - Merge neighbouring windows from the same page; they now repeat 150 characters of overlap.
  - Change only what the reader reads. Embeddings, reranker scores and the gate stay byte-identical,
    so any change in abstention can only come from the reader.

### 5. Change the order and number of extracts
- **Why:**
  - The question comes after the extracts, so the best extract, [1], ends up farthest from it.
  - Phase 8 showed that order alone flips answers.
  - Phase 7 only tried more extracts, never fewer.
- **How:**
  - One run with the order reversed (keep citation numbers mapped to the right chunks when scoring).
  - One run with `--read-k 3`, which `eval_answers.py` already supports (no code).

### 6. Add worked examples to the reader prompt
- **Why:** R14c added one sentence about misspelt and everyday wording and got +20 overall and +10
  on terse questions. Worked examples are the stronger form of the same lever, and none has been
  tried.
- **How:**
  - Add 2–3 examples built on pages that are not gold pages of any eval question:
    - a misspelt question read correctly;
    - an answer found in extract 3 rather than 1;
    - only distracting extracts, answered "I don't know".
  - About +500 prompt tokens. Watch abstention.

### 7. Make the answer start by choosing a real line from the extracts
- **Why:** R6 quote mode cut unsupported answers by 59 and gained 16 abstentions. It still lost 258
  correct answers, because the 4B mis-copied its own quotes 27% of the time and every miscopy became
  a refusal. If the grammar lists the extracts' actual lines, a miscopy becomes impossible, and the
  model only has to choose a line, which is easier than writing one.
- **How:**
  - Add a new grammar in `grammar.py` with each extract's lines as literal alternatives (about 75–150).
    The answer becomes `[n] "<chosen line>"` followed by the cited claim.
  - Measure the grammar's sampling latency.
  - Phase 6's lesson still applies: a real line can still be the wrong line, so judge this on
    correctness, not on support.
  - This is also a bounded "reason first" step. The unbounded alternative is Qwen3-4B-Thinking-2507,
    at a cost of seconds per answer.

### 8. Try a less compressed reader
- **Why:** The reader is `Qwen3-4B-Instruct-2507-Q4_K_M` (2.5 GB). The task is copying exact option
  spellings, the kind of precision 4-bit compression is likely to cost a small model. Never tested.
- **How:**
  - Run Q5_K_M, Q6_K and Q8_0 as generation-only runs. Evals load one model at a time, so VRAM is
    not a limit there.
  - For everyday serving, Q6_K only fits beside the embedder and reranker on 6 GB with a quantized
    KV cache (`--cache-type-k q8_0 --cache-type-v q8_0`) or a smaller context.

### 9. Ask twice with different orders, and refuse when the answers disagree
- **Why:** Phase 6 concluded that a useful check has to ask whether the cited extract answers this
  question, not just whether it supports the claim. Disagreement between two orderings is a direct
  symptom of the unreliable reading phase 8 measured.
- **How:**
  - Generate with the original and the reversed order. If the named options differ, refuse, or hand
    the question to the 30B with check 2 applied.
  - About +0.9 s p50 per question.
  - Score how well disagreement separates right from wrong answers with `scripts/eval_verifier.py`.

### 10. A second read when the model refuses with strong evidence
- **Why:** 73 answerable rows say "I don't know" with the gold passage in hand.
- **How:** When the top score is high (e.g. ≥ 0.9) and the reader refuses, ask again with only the
  top 1–2 extracts. Pre-register the limit on new inventions on unanswerable rows that pass the gate.

## C. Retrieval and index

### 11. Configure each domain separately (phase 15's planned next step 1)
- **Why:** Already measured: on Emacs the neutral embedder instruction gave +26 (R14a) and the
  fine-tuned embedder +21 (R13); on man pages the reranker instruction gave +23 (R14b).
- **How:** Use the Emacs settings when `--domain emacs` is given explicitly (the `asq.el` path),
  measured on Emacs rows, and the reranker instruction on man pages. Under an explicit domain, the
  Emacs settings cannot cost man pages anything.

### 12. Use the Emacs manual's own index
- **Why:**
  - `scripts/extract_info.py` skips every node whose name ends in "Index", so thousands of
    human-written index entries that map wordings to sections are thrown away.
  - R8 used the same mechanism with 57k 4B-written questions. It was phase 11's only retrieval gain
    with no measured side cost (+23 evidence against 1 lost), and it missed on latency (+1.78 s).
  - Human-written entries are fewer and aimed at Emacs's vocabulary gap: synonym −13 and no-name −19
    on Emacs (phase 11).
- **How:** Parse the index nodes (`* entry: Node. (line N)`). Embed each entry as an extra vector
  pointing at its target chunk, reusing the R8 question-vector code (`build_qvec.py`,
  `store.search_questions`).

### 13. Union the typed and spell-corrected searches, but rerank against what was typed
- **Why:** R4a spell correction won back +27 typo answers but renamed tools (nmap → mmap). If the
  reranker scores against the user's own wording, a renamed tool's pages should not rank well.
  Whether this keeps R4a's gain is untested.
- **How:** `Retriever.retrieve_widened` already has this exact shape. Feed it the vocabulary
  corrector's output (0.4 ms, `smm.normalize`) instead of a 4B rewrite. Keep the candidate pool at a
  fixed size to protect latency.

### 14. Add the section heading to what gets embedded
- **Why:** In phase 3 the heading trail raised dense recall@5 from 50.8% to 60.3%. It was only
  tested together with structure-aware chunking, which broke the gate, so the trail alone is
  untested.
- **How:** Add it to the prefix of the existing flat windows, and align window starts too. This
  needs a 41-minute rebuild plus a gate re-sweep. Do it after idea 4, which gives the reader the same
  heading for free.

## D. Training

### 15. Fine-tune the reader on this exact task (RAFT-style LoRA on the 4B)
- **Why:** Reading has been the bottleneck since phase 8, and prompting can only go so far.
  Fine-tuning a model on retrieved contexts that include distracting passages is a known method for
  this (RAFT).
- **How:**
  - Start from the 102k questions R13 already generated, each tied to its chunk.
  - Context: the retriever's real top 5 for that question, shuffled.
  - Target: a short cited answer written by the 30B from the gold chunk alone, kept only if its
    option appears in the chunk.
  - Remove the gold chunk in about 25% of examples, with the target "I don't know".
  - Add typos to about 30% of questions with `make_variants.py`.
  - Exclude eval gold chunks, using R13's leakage check.
  - 4-bit LoRA training of a 4B at about 2.5k tokens is at the edge of 6 GB; rent a GPU for a few
    hours. The 30B teacher (about 8 s per question) limits data to about 10k examples a day.
- **Cost:** the highest here, with probably the largest payoff.

## E. Measurement and display

### 16. Have a judge score the rows that differ between runs
- **Why:** The labels miss about 53% of answers marked wrong-with-evidence (audit 2), and v2
  recovered only about half. Several arms missed by one row (R8 p 0.0625; R14c rule 3 by one), so
  label noise is hiding real gains.
- **How:** Run the blind `orch-judge` only on the rows where control and arm differ (about 50–120
  per comparison). Also grow a held-out split for tuning the gate.

### 17. Show the evidence line under the answer
- **Why:** It changes nothing in the model and shows the user the answer comes from their manual.
- **How:** Print the line from the cited extract that contains the named option, plus a
  `man 1 tar` or Info node pointer. Idea 7 produces this line for free.

---

## Suggested order

1. **Ideas 1, 2 and 3.** No GPU; they replay existing files.
2. **Ideas 4, 5, 6 and 8.** One retrieval cache serves all of them, at about 20 min per pool run.
3. **Idea 7.**
4. **Ideas 11 and 12.**
5. **Idea 15**, if the cheap ones level off.

## Today

- [ ] Restart Claude Code, so tasks are dispatched to the ORCH v5 agent definitions.
- [ ] Make the two draft checks tracked (identifier grounding; extract starts), so their numbers
      become evidence or are dropped.
- [ ] Idea 1: the grammar bracket fix (`orch-task`).
- [ ] Idea 2: pre-register the identifier-grounding replay, then run it offline.
- [ ] Idea 3: pre-register the calibrated gate (dev/held-out split fixed first), then fit it offline.
- [ ] If GPU time is left: pre-register ideas 4 and 5 as generation-only arms on one cache.

## Rules that apply to every item

- Pre-register before running: arms, rules and thresholds fixed first.
- Generate under setting C (`SMM_GEN_ARGS="--parallel 1 --no-cache-prompt --cache-ram 0"`), against
  a control regenerated the same way. Use `SMM_GEN_PORT=8090` because :8080 is taken.
- Score through `scripts/screen_report.py` on `gold_aliases_v2.json`. For anything that changes
  retrieval, also report `--match-abstention`.
- Compare retrieval runs of the same shape (subset runs drift 3/50 by request order).
