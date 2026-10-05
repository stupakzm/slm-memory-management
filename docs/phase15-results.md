# Phase 15: answering in people's words, not the manual's

## Why

Phase 11 mapped where wording breaks the system (docs/phase11-results.md, "The map"): typos break
the **reader** (the page arrives, and the 4B misreads the question), while synonyms, unnamed tools
and terse queries break **retrieval** (the page never arrives). Everything that spent the 4B at
question time lost answers or added inventions (R4b, R4d). The two mechanisms that recovered
answers without measured side cost did their work *before* the question, in the index (R8), or
repaired the question's own words (R4a, R4a′). The repairs failed only on renaming tool names
the corpus has never seen.

This phase runs four parts, in the order the user set (2026-10-04):

| part | what | kind |
|---|---|---|
| R12 | typo repair for the reader, using only words in the evidence | query-side, CPU, measured |
| R13 | the 0.6B embedder fine-tuned on plain-English questions written for each chunk | index-side, one-time training, measured |
| asq.el | `M-x asq` asks with `--domain emacs` against the index that holds the Emacs manuals | product, no measurement |
| R14 | domain-aware instructions for the embedder, reranker and reader | query-side, measured |

Each measured part gets its own pre-registration here, written and committed before its run.

## R12 pre-registration: repair typos from the evidence (written 2026-10-04, before the build and any run)

**Why this signal.** R4a, R4a′ and R4a″ all recovered typo answers (+27, +29) and all failed rule 2
on names: `nmap`→`mmap`, `elpy`→`elpa`, `nmap`→`map`. Whether a word is a typo can't be decided
from the word and a dictionary alone (R4a″ verdict). The evidence can decide it: a misspelt
`windw` sits one edit from `window`, and `window` is in the extracts the reader is about to read. An
unknown tool name has nothing close to it in the extracts *about that question*. So R12 corrects
only toward words the reader is shown.

**Mechanism.**
- Retrieval and the gate run on the question **exactly as typed**. Phase 11 found typos rarely
  break retrieval (about 1 in 5 typo1 losses), so nothing there is touched.
- When the gate passes, each question word that is eligible under R4a's own rule (alphabetic,
  4+ letters, not in the index vocabulary) is corrected to the closest word that occurs in the
  extracts the reader reads. An extract is what the reader is shown for one hit: its document id,
  prefix and text.
- Distances, tie-break order, case and punctuation handling are R4a's: ≤1 edit for 4-7 letters,
  ≤2 for 8+. Ties go to a keyboard-adjacent substitution, then to the word more frequent in the
  extracts.
- Only the reader sees the repaired question. A gated question is never read, so it is never
  repaired.

**Arm.** Control: `p13-ctl` (open domain, `phase11.db`, dense, reranked over 50, k 5, gate 0.65,
grammar, generated under setting C). R12 changes nothing before the reader, so it reuses the
control's retrieval cache `p13-ctl-retrieved.json` (sha256 recorded at run time).
1. **CPU pre-pass**, before any GPU:
   `src/smm/normalize.py --mode local --db data/index/phase11.db --retrieved
   data/eval/results/p13-ctl-retrieved.json --gate 0.65` over the five pool files. It fixes the set
   of rows whose reader input changes. Rule 2a is decided here. If it fails, the GPU arm is not run.
2. **Generation** of all 1,160 rows with `--normalize local` under setting C (`p15-r12`). Rows the
   pre-pass leaves unchanged have byte-identical reader input to the control, so they double as
   the noise check. **If any of them differs in answer text from `p13-ctl`, the control is
   regenerated today with `--normalize off` (`p15-ctl`) and used in its place.**

**Scoring.** `scripts/screen_report.py --group variant_kind` over the five pool files, with
`--aliases data/eval/gold_aliases_v2.json`. The scorer's default group holds the 160 clean bases
**and** the 19 older paraphrases, which carry no `variant_kind`. That group has 132 answerable rows,
and rule 2b reads it as "clean".

**Decision rule.** `--normalize local` becomes `asq`'s default only if all five hold:
1. **Typos recover:** typo1 + typo3 together net ≥ +10, sign test over their pooled discordant pairs
   p < 0.05.
2. **Clean text left alone:** (a) the pre-pass changes the reader's question on ≤ 3 of the 160 clean
   base questions; (b) the clean group (above) nets ≥ −1.
3. **No new invention:** abstention net over all 277 unanswerable rows ≥ −1.
4. **Cheap:** the pre-pass p95 is < 10 ms per question.
5. **Nothing else pays:** all other answerable rows together (synonym, casual, terse, no-name, typo,
   no-tool) net ≥ −2.

Failing 2 or 3 means no ship, whatever 1 says. Passing 1 with p ≥ 0.05 is reported as "direction
only".

**Known limits, stated now.**
- A typo in a tool name (`tsr` for `tar`) is repaired only if the right page was retrieved
  anyway. Words of 3 letters are never touched (R4a's rule).
- An unanswerable question that passes the gate *and* has a 1-edit neighbour of its tool name in
  the extracts can still be renamed. Rule 2a counts the clean ones, and rule 3 catches any that
  turn into answers.
- The typo pool was generated mechanically (`scripts/make_variants.py`). People's typos differ.

## R13 pre-registration: teach the embedder people's words (written 2026-10-04, before any training run)

**Why.** Synonyms and unnamed tools lose answers at retrieval: the page never arrives (phase 11
R3). R8 showed the 4B can write plain-English questions for each chunk, and adding them to the index
gained evidence with no measured side cost. But it reranked up to 30 more candidates (+1.78 s), and it
covered Emacs only. R13 trains the same knowledge into the 0.6B embedder itself. At question time
nothing changes: one vector per chunk, the same model size, the same 50 candidates.

**Training data.** No eval question is ever used.
- Emacs: the existing R8 cache `data/index/qvec-emacs.json`: 19,046 chunks × 3 questions, prompt
  sha256 `b878e89e…`.
- Man pages: 15,000 chunks sampled from the 56,567 (`build_qvec.py --domain linux --prompt linux
  --sample 15000 --seed 20261004`), 3 questions each, written by the 4B at temperature 0. The
  prompt sha256 is recorded at run time.
- About 1 % of chunks are held out by hash (`sha256("20261004:" + chunk_id)`) as a dev split.

**Training.** `scripts/train_embedder.py` on `Qwen/Qwen3-Embedding-0.6B` (HF weights, sha256
`0437e45c…`), under `.venv-train` (`requirements-train.txt`).
- Hand-written LoRA, rank 16, alpha 32, on every attention and MLP projection.
- In-batch-negative InfoNCE: batch 32, temperature 0.05, lr 1e-4, 1 epoch, seed 20261004.
- Queries are wrapped in the runtime instruction (`smm.embed.query_text`). Chunks are embedded
  exactly as the index embeds them (prefix + text).
- Merged, converted to GGUF Q8_0 (`convert_hf_to_gguf.py --outtype q8_0`), served with
  `SMM_EMBED_MODEL`.

**Two gates before any pool run.** These are not rules: a fail stops R13 before the re-index.
- **G1, dev.** Dev-split recall@10 (held-out chunks' questions, ranked among held-out plus 5,000
  training chunks) rises by ≥ 5 points over the untouched base model, both in HF bf16. If not, R13
  is reported null and stops.
- **G2, parity.** `embed_parity.py`: the fine-tuned GGUF served by llama-server matches the
  fine-tuned HF model at min cosine ≥ 0.99, for queries and for chunks (200 each). The same check on
  the base model is reported, as calibration. A fail is a conversion defect to fix, not a result.

**Arm.** `data/index/phase15-ft.db` is `phase11.db` with all 75,613 chunk vectors re-embedded by the
fine-tuned embedder (`scripts/reembed.py`; chunks table identical). Questions are embedded by the
same model. Everything else is `p13-ctl`'s configuration: open domain, dense, reranked over 50, k 5,
gate 0.65, grammar, generation under setting C. All 1,160 pool rows (`p15-r13`).

**Control drift check** (phase 13's procedure). Control retrieval is re-run today on 50 pool rows
(seed 13) with the base embedder on `phase11.db`. If more than 2 of the 50 differ from
`p13-ctl-retrieved.json` in top-5 chunk ids or in the gate decision, the control is regenerated in
full today and used instead. If R12 already regenerated the control, that one is used.

**Scoring.** `scripts/screen_report.py --group variant_kind --aliases data/eval/gold_aliases_v2.json`
over the five pool files. The default group is "clean" as defined under R12.

**Decision rule.** The fine-tuned embedder becomes the default only if all five hold:
1. **Vocabulary recovers:** synonym + no-name + terse + casual together (484 answerable rows) net
   ≥ +10, sign test over their pooled discordant pairs p < 0.05.
2. **Clean unharmed:** the clean group (132 answerable) nets ≥ −2.
3. **Typos unharmed:** typo1 + typo3 + typo (old) together (252 answerable) net ≥ −3.
4. **No new invention:** abstention net over all 277 unanswerable rows ≥ −1.
5. **Same speed:** mean retrieval seconds per question (from the retrieve log) ≤ control + 0.3 s.

Failing 2, 3 or 4 means no ship, whatever 1 says. Passing 1 with p ≥ 0.05 is reported as "direction
only".

**Reported, not rules:**
- Per domain (`--group domain`) and evidence net.
- The 100-row realistic Emacs set (`realistic_emacs.jsonl`, human-phrased), control and arm both run
  today.
- **Leakage.** The maximum token Jaccard between every eval question and all training questions.
  Any gained row at ≥ 0.8 is listed, and rule 1 is restated without it.

**Known limits, stated now.**
- Training questions are written by the same 4B whose wording R8 used. The pool's variants were
  written by hand in earlier sessions, and the realistic set is the closest to real users. So a win
  that shows on the pool but not on the realistic set will be called out.
- 15,000 of 56,567 man-page chunks get questions. The rest of the man corpus is learned only through
  transfer.
- Re-embedding changes every vector, so any retrieval-noise effect (batch composition, phase 14)
  lands in this comparison too. The drift check bounds it for the control, not for the arm.

## R12 results (run 2026-10-04)

**Pre-pass** (`data/eval/results/p15-r12-prepass.json`, over `p13-ctl-retrieved.json`).
- The reader's question changes on 210 rows: typo1 74, typo3 116, typo (old) 19, and 1 clean row
  [chk_013]. 206 rows are gated as before.
- The only clean change is `eu01`: "using projectile" → "using projection", an unanswerable row about
  a tool the corpus lacks.
- On the typo rows the repair reverts the generator's own recorded edits (typo1 69, typo3 214) with
  **0 new wrong edits**.
- p95 is 3.45 ms per question.

**Control.** The pre-registered noise check compares the unchanged rows' answer text with `p13-ctl`.
No tracked tool does that comparison, and an inline script may not decide which control is used.
So the pre-registered fallback was taken instead: the control was regenerated today under setting C
with `--normalize off` (`p15-ctl`). It matches `p13-ctl` exactly, 0 lost and 0 gained over 883
answerable rows and the same abstentions [chk_011]. With no noise, the two paths give the same
answer. Setting C held across two days.

**Arm** `p15-r12` against `p15-ctl` [chk_012]:

| group | answerable | lost | gained | net |
|---|---|---|---|---|
| typo1 | 116 | 0 | 2 | +2 |
| typo3 | 116 | 5 | 0 | −5 |
| clean (with older paraphrases) | 132 | 0 | 0 | 0 |
| every other kind | 519 | 0 | 0 | 0 |
| all | 883 | 5 | 2 | −3 |

Abstention net is +1. Lost: a06.y3, a10.y3, a37.y3, b02.y3, e16.y3. Gained: a29.y1, b02.y1.

**Against the rule:**
1. Typos recover (pooled net ≥ +10, p < 0.05): **fail.** Pooled typo1 + typo3 is −3 (2 gained, 5 lost).
2. Clean left alone: pass. (a) 1 of 160 changed (≤ 3). (b) Clean net 0.
3. No new invention: pass (+1; `eu01` still abstains).
4. Cheap: pass (p95 3.45 ms).
5. Nothing else pays: pass (0).

**Verdict: does not ship.** `--normalize local` stays default-off.

**What it establishes.** With the five extracts held fixed, correcting the question's spelling does
not help the 4B reader. The repairs are right: "lsit a directoryy … fiirst" becomes "list a directory …
first", "wihtout havign" becomes "without having". But the answers flip in both directions on the same
evidence. `a06.y3` was answered correctly while misspelt and became "I don't know" once spelled
correctly. `b02.y1` and `b02.y3` moved in opposite directions on one base question.

Phase 11's reading of R3 was "typos break the reader: the page arrives and the 4B misreads the
question". It needs restating. When the page was in hand, what differed between a typo variant and its
clean base was the **context**: the other extracts and their order. It was not the misspelt word. R4a
gained +27 because it repaired the question **before retrieval**, so the reader got the clean base's
context.

So typo repair belongs before retrieval. The rename problem has to be solved there. The evidence-local
signal can be moved there, unrun and not pre-registered: retrieve with the raw question, repair from
its extracts, and retrieve again only when something was repaired. That costs one extra retrieval on
about a sixth of the pool's rows.

## R14 pre-registration: say what the corpus is (written 2026-10-04, before the build and any run)

**Why.** Three models are told what they are doing, and all three are told something narrower or
vaguer than the truth. None of these instructions was ever measured:
- the embedder's query instruction says "a question about using a Linux system … the manual page
  passage" (`src/smm/embed.py`), even for Emacs questions;
- the reranker runs the instruction built into its GGUF, "Given a web search query, retrieve relevant
  passages that answer the query";
- the reader's system prompt says "a Linux system … manual page extracts", and nothing warns it that a
  question may be misspelt, terse or in everyday words.

Phase 13 found that knowing the domain is worth +40 answers. Part of that may be instruction mismatch.

**Arms**, one mechanism each, never combined in this pre-registration:
- **R14a, embedder:** `--embed-task neutral`: "Given a question about using the software on this
  computer, Linux commands and configuration or the GNU Emacs editor, retrieve the documentation
  passage that answers it". Query side only, so the index is unchanged. Retrieval and generation over
  all 1,160 pool rows.
- **R14b, reranker:** a derived reranker GGUF (`scripts/rerank_instruct.py`) whose rerank template
  says "Given a question from a user of Linux command-line tools or the GNU Emacs editor, possibly
  misspelt or in everyday words, judge whether the documentation passage answers it". Only that
  metadata string differs, which a key dump shows before the run. Served with `SMM_RERANK_MODEL`.
  Retrieval and generation, all rows. The gate stays 0.65, so a shift in score scale shows up in
  rule 3 and in the gate-fire rate.
- **R14c, reader:** `--reader-prompt v2` (text in `src/smm/generate.py` `SYSTEM_V2`, sha256 recorded
  at run time). Generation only, on the control's retrieval.

**Control.** The system's default configuration on the day of the run, generated under setting C.
If R13 has shipped by then, that means the R13 embedder and index. If a retrieval control from today
exists, it is reused. Otherwise phase 13's drift check (50 rows, seed 13, more than 2 differing means
full regeneration) decides.

**Scoring.** `scripts/screen_report.py --group variant_kind` (default group "clean", as under R12)
and `--group domain --group-default linux`, `--aliases data/eval/gold_aliases_v2.json`.

**Decision rule, per arm.** It becomes the default only if all four hold:
1. **Answers rise:** all 883 answerable rows net ≥ +8, sign test p < 0.05.
2. **Clean unharmed:** clean group net ≥ −2.
3. **No new invention:** abstention net over 277 unanswerable rows ≥ −1.
4. **Same speed** (R14a, R14b): mean retrieval seconds per question ≤ control + 0.3 s.

Failing 2 or 3 means no ship. Passing 1 with p ≥ 0.05 is "direction only". If two or more arms pass,
their combination gets its own pre-registration. It is not assumed to add up.

**Known limits, stated now.** R12 showed the 4B's answers flip in both directions when only the
question's wording changes. A new reader prompt will churn answers the same way, so rule 1 needs a
net gain, not churn. The reranker arm changes the score the gate reads. A fail on rule 3 there may
be calibration rather than invention, and that is reported as such, without re-tuning the gate here.

## R13 results (run 2026-10-04 to 2026-10-05)

**Data.** 102,138 question–passage pairs: Emacs 19,046 × 3 (R8's cache) and man pages 15,000 × 3,
newly written (prompt sha256 `127b2e51…`). The hash split holds out 332 chunks (996 questions) as
dev, and trains on 101,142 pairs. **Leakage:** the closest any of the 1,160 pool questions comes to
any training question is token Jaccard 0.778, below the 0.8 flag [chk_020]. No gained row is flagged.

**Training.** 3,161 steps over four resumable stages, 5.5–5.9 s per step on the RTX 3060 Laptop
(about 5 h). Loss fell from 0.37 to about 0.04. The log is frozen as
`data/eval/results/p15-r13-train_log.json`.

**G1, dev: pass.** recall@10 rose from 0.746 to 0.961 [chk_020], against a bar of five points. recall@1 rose
from 0.388 to 0.650. These are 4B-written questions about held-out chunks, so they are the most
favourable test.

**G2, parity: pass.** The fine-tuned GGUF served by llama-server matches the HF model at min cosine
0.9988 for questions and 0.9988 for chunks (200 each, threshold 0.99). The base model's own
conversion scores 0.9989 / 0.9991, as calibration. Run with `scripts/embed_parity.py`. These need a
live server, so they are not registered. Converting the base HF model with the same command
reproduces the official GGUF's 310 tensors byte for byte (only `general.name` differs), so the
conversion step is exact.

**Re-index.** `data/index/phase15-ft.db`: all 75,613 vectors re-embedded at 21.7 chunks/s (58 min).
The chunks table is identical to `phase11.db`.

**Control drift check.**
- The 50-row sample (seed 13) differs from `p13-ctl` on 3 rows (a29.y3, a30, eu05.y3), all in
  top-5 order, none in the gate [chk_023]. That exceeds 2, so the control retrieval was regenerated
  in full today, as pre-registered.
- The full run reproduces `p13-ctl` on **all 1,160 rows** [chk_025]. So `p15-ctl`, which was
  generated on that retrieval, is the control.
- The 3 rows are phase 13's same 3, and today's 50-row run matches phase 13's 50-row run exactly
  [chk_024]. So a row's retrieval depends on which rows ran before it in the same server session.
  A subset run and a full run disagree. Runs of the same shape agree across days. This is a
  property of the drift check's design, not of the code. The cause is untested; the reranker
  server's prompt cache is the obvious suspect.

**Arm** `p15-r13` against `p15-ctl` [chk_026, chk_027, chk_028]:

| group | lost | gained | net |
|---|---|---|---|
| synonym + no-name + terse + casual (484) | 27 | 36 | +9 (p 0.31) |
| clean (132) | 4 | 8 | +4 |
| typo1 + typo3 + typo (252) | 14 | 21 | +7 |
| **all answerable (883)** | 46 | 65 | **+19** (p 0.087) |
| **Emacs (324)** | 11 | 32 | **+21 (p 0.002)** |
| man pages (559) | 35 | 33 | −2 |

Evidence rises from 711 to 726. Abstention falls by **6** overall: 12 unanswerable rows newly
answered and 6 newly abstained. Emacs abstention is +1; man pages are −7.

**Against the rule:**
1. Vocabulary recovers (≥ +10, p < 0.05): **fail** (+9, p 0.31).
2. Clean unharmed (≥ −2): pass (+4).
3. Typos unharmed (≥ −3): pass (+7).
4. No new invention (≥ −1): **fail** (−6).
5. Same speed (≤ +0.3 s): pass (3.27 against 3.24 s per question).

**Verdict: does not ship as the default embedder.** It fails rules 1 and 4.

**Realistic Emacs set** (reported) [chk_029]. 34/80 against 28/80, +6 (10 gained, 4 lost, p 0.18).
Abstention is 20/20 in both arms.

**What it establishes.**
- **On the Emacs manuals the fine-tune works.** It gives +21 answers at p 0.002 with no loss in
  abstention, and +6 on the human-phrased realistic set. This is the first mechanism in phases 11–15
  with a significant gain on the domain where vocabulary breaks retrieval. It costs nothing at
  question time: the same model size, one vector per chunk, and +0.03 s per question.
- **On man pages it buys nothing and costs safety.** The newly answered unanswerable rows are all
  man-page tools the corpus lacks. `strace` now retrieves `vdso.7` and `proc_pid_syscall.5`;
  `tcpdump` retrieves `tc.8` and `ss.8`. The reranker scores these above the 0.65 gate, and the
  reader answers, sometimes from its own knowledge (`tcpdump -i <interface> -w <filename>`). The
  embedder got better at what a question means. The gate was tuned for the old embedder's weaker
  near-misses.
- **Two follow-ups,** not run and not pre-registered:
  - Adopt the fine-tuned embedder for the Emacs domain only (`asq.el` always passes `--domain emacs`).
    That needs a pre-registered check on Emacs rows under `--domain emacs`.
  - Re-sweep the gate for the fine-tuned embedder on man pages (phase 8's gate re-sweep procedure).
    It runs first, before any adoption that touches man pages.

## R14 results (run 2026-10-05)

All three arms against `p15-ctl`, generated under setting C. R14c's prompt sha256 is `a778ffa4…`.
R14b's derived reranker differs from the original in 1 of 40 metadata keys
(`tokenizer.chat_template.rerank`), and all 311 tensors are identical.

| arm | all answerable | clean | abstention net | Emacs | man pages | retrieval s/q | gate fired |
|---|---|---|---|---|---|---|---|
| R14a, embedder instruction [chk_032, chk_033] | 53 gained, 32 lost, +21 (p 0.030) | −3 | −5 | **+26 (p 0.00002)** | −5 | 3.27 | 196 |
| R14b, reranker instruction [chk_034, chk_035] | 73 gained, 48 lost, +25 (p 0.029) | −4 | −6 | +2 | **+23 (p 0.015)** | 3.46 | 181 |
| R14c, reader prompt [chk_030, chk_031] | 38 gained, 18 lost, +20 (p 0.011) | +2 | −2 | +11 (p 0.007) | +9 | (control's) | 206 |

The control's gate fired on 206 rows, at 3.24 s per question.

**Against the rule:**
- **R14a:** rule 1 passes; rule 2 **fails** (−3); rule 3 **fails** (−5); rule 4 passes.
- **R14b:** rule 1 passes; rule 2 **fails** (−4); rule 3 **fails** (−6); rule 4 passes (+0.22 s).
- **R14c:** rule 1 passes; rule 2 passes; rule 3 **fails by one row** (−2).

**Verdict: no arm ships.** No combination is pre-registered, because none passed.

**What R14 establishes.**
- **The three instructions matter, and each helps a different place.**
  - The embedder instruction is worth +26 on Emacs. The old "Linux system … manual page"
    instruction was costing the Emacs questions. A one-line change beats R13's 5-hour fine-tune
    (+21) there.
  - The reranker instruction is worth +23 on man pages.
  - The reader prompt is worth +10 on terse questions (p 0.006) and +11 on Emacs.
- **Every arm pays in abstention.**
  - R14a and R14b change retrieval, and the gate fires less: 196 and 181 rows against 206. The 0.65
    threshold was set for the old score distribution.
  - R14c changes no retrieval. Its 3 newly answered unanswerable rows come from the prompt itself.
    "Answer what the user most plausibly means" draws the 4B into answering from its own knowledge:
    `strace`, and `tcpdump -i <interface> -w <filename>` alongside an admission that the extracts
    do not say it. The third, `u07.k` "gdb set breakpoint", is answered from Emacs's own GUD manual
    (`C-x C-a C-b`, `gud-break`). That answer is documented, so the row's tool-not-installed label is
    arguable. It is left as labelled.

## Diagnostic, not pre-registered: every arm at the control's abstention (2026-10-05)

Every arm that changed retrieval lowered the number of rows the 0.65 gate refuses: R13 192, R14a 196,
R14b 181, against the control's 206. So part of each gain may be the gate letting more questions
through, answerable and unanswerable alike.
`scripts/screen_report.py --match-abstention` (tsk_20261005_matchabst) re-scores each arm at the
lowest gate where its abstention equals the control's (254/277). It replays the recorded top scores,
so no GPU run is needed. **This decides nothing.** It was not pre-registered, and it shows where the
next phase should look [chk_038].

| arm | matched gate | all answerable | Emacs | man pages |
|---|---|---|---|---|
| R13, fine-tuned embedder | 0.834 | +1 | **+19 (p 0.007)** | −18 (p 0.06) |
| R14a, embedder instruction | 0.765 | +9 (p 0.41) | **+25 (p 0.00004)** | −16 (p 0.044) |
| R14b, reranker instruction | 0.866 | +12 (p 0.33) | +1 | +11 (p 0.29) |
| R14c, reader prompt | 0.685 | **+16 (p 0.048)** | +11 (p 0.007) | +5 |

At equal safety, the Emacs gains from R13 and R14a hold, and both cost man pages about as much again.
The reader prompt keeps most of its gain with a slightly stricter gate.

## Phase 15 closing (2026-10-05)

| part | mechanism | verdict | the number that decided it |
|---|---|---|---|
| R12 | typo repair for the reader, from the extracts | no ship | typos pooled −3 [chk_012] |
| R13 | 0.6B embedder fine-tuned on 102k 4B-written questions | no ship (rules 1 and 4) | vocabulary +9; abstention −6 [chk_026, chk_028] |
| asq.el | `M-x asq`, Emacs manuals with `--domain emacs` | shipped (code) | — |
| R14a | neutral embedder instruction | no ship (rules 2 and 3) | clean −3; abstention −5 [chk_032] |
| R14b | reranker instruction in the GGUF | no ship (rules 2 and 3) | clean −4; abstention −6 [chk_034] |
| R14c | reader prompt for misspelt, terse and everyday questions | no ship (rule 3) | abstention −2 [chk_030] |

**What phase 15 established:**
- **Typo damage is context, not words.** Repairing the reader's question on fixed evidence does
  nothing (R12). The gain R4a saw came from changing what retrieval returns.
- **The Emacs vocabulary gap can be closed.**
  - A one-line embedder instruction gives +26 on Emacs [chk_033], and still +25 at the control's
    abstention [chk_038].
  - The fine-tuned embedder gives +21 [chk_027], at no extra cost per question.
  - Both cost man pages, so a single global setting cannot carry them.
- **Each model wanted a different fix.** The embedder instruction helped Emacs, the reranker
  instruction helped man pages, and the reader prompt helped terse questions.
- **The gate is calibrated to one pipeline.** Every retrieval change shifted it. A new configuration
  has to be judged at matched abstention, or with a re-swept gate, not at 0.65 by default.
- **Measurement.**
  - Full-pool retrieval reproduces exactly across days [chk_025], and so does generation under
    setting C [chk_011].
  - Subset retrieval runs drift on 3 of 50 rows because of the order requests reach the server
    [chk_023, chk_024]. A drift check has to compare runs of the same shape.

**Next, for pre-registration** (none run):
1. **Domain-aware configuration.** With `--domain emacs` (the `asq.el` path), use the neutral or an
   Emacs-specific embedder instruction, and measure on Emacs rows under `--domain emacs`. The man-page
   path is untouched, so it cannot cost man pages.
2. **A gate re-sweep per configuration**, chosen on a dev split and judged once on held-out rows,
   before adopting R14b or R14c globally.
3. **R14c at a pre-registered stricter gate.** The diagnostic suggests about +16 at matched
   abstention; that has to be confirmed on rows it was not chosen on.
