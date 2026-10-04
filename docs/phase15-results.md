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
