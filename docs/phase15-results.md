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
