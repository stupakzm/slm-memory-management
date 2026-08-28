# Evaluation sets

Two files, both written before the system they measure, which is the whole point:
they are the only things that can say whether a phase actually helped.

- `questions.jsonl` — 166 documentation questions (phase 0)
- `tool_questions.jsonl` — 80 tool-calling requests (phase 4)

# Phase 0 evaluation set

166 questions against the man pages installed on this machine.

## Composition

| | count | |
|---|---|---|
| Answerable | 127 | 71 unique + 13 paraphrases + 55 query-noise variants |
| Unanswerable | 39 (23%) | |
| Distinct gold documents | 57 | |
| Paraphrase pairs | 13 | tests the 11–19% paraphrase fragility from Finding 05 |

Answerable questions break down as 82 flag lookups, 27 config-format questions
(man5), 16 concept questions (man7), and 40 tagged `exact-token` — queries whose
answer is a literal string like `--exclude-from`, `@reboot` or `status=LEVEL`.
Phase 2 tested the expectation that BM25 would win on those and found it does not,
even with the tokenizer chosen to favour it.

Query-noise variants (55) cover how questions are really typed: `terse` (20),
`typo` (20) and `no-tool` (15), the last dropping the tool name entirely.

Unanswerable questions split three ways, and the distinction matters:

- `tool-not-installed` (18) — docker, rsync, tmux, nmap, strace, gdb, kubectl and
  friends. Realistic questions with no page on this machine. This is the case a
  cloud model answers confidently and wrongly.
- `out-of-corpus` (6) — C library and language questions; man2/man3 are not indexed.
- `requires-execution` (6) — "how much space is left on /home". Answerable only by
  running something, never by retrieval. **Phase 4 flips the expected behaviour of
  these from abstain to tool-call**; until then they must abstain.

## Schema

```json
{
  "qid": "a01",
  "question": "...",
  "kind": "answerable" | "unanswerable",
  "doc": "tar.1",
  "answer_contains": ["--exclude-from"],
  "gold_sec_ids": ["tar.1#OPTIONS/Local file selection"],
  "gold_primary": "tar.1#OPTIONS/Local file selection",
  "tags": ["flag", "exact-token"],
  "paraphrase_of": null,
  "unanswerable_reason": "tool-not-installed",
  "unanswerable_detail": "docker"
}
```

`gold_sec_ids` holds every section that contains the answer; retrieving any one of
them counts as a hit. `gold_primary` is the single section a reader would be sent
to, for MRR.

## How it stays honest

`scripts/resolve_gold.py` is the validator, and it is not optional — run it after
any edit to the questions or any change to extraction:

```bash
.venv/bin/python scripts/resolve_gold.py          # check
.venv/bin/python scripts/resolve_gold.py --write  # check and refill gold ids
```

It enforces three things:

1. **No gold by assertion.** Every answerable question must name a real document
   containing a section where all its answer tokens literally appear. This caught
   three wrong facts on the first run: `dd` documents `status=LEVEL` not
   `status=progress`; `shadow.5` writes "minimum password age" in lower case; and
   `proc.5` is split into per-file pages on Debian 13, so the load average lives in
   `proc_loadavg.5`. All three are cases where general knowledge about Linux is
   wrong about *this machine*.
2. **No leaked answers.** A question may not contain its own answer token, or it
   measures string matching rather than retrieval. Six questions were rewritten.
3. **Genuinely unanswerable.** Every `tool-not-installed` question is checked
   against the corpus; if the tool turns out to have a page, the question is an
   error rather than an abstention case. Every `out-of-corpus` question must name
   the page it would need in `unanswerable_detail`, and that page must genuinely be
   absent.

   That second half did not exist until phase 4, and its absence is how a wrong
   label survived three phases. `u22` — *how do I install Python packages with
   pip* — was tagged `out-of-corpus` while `pip.1`, `pip-install.1` and
   `pip3-install.1` were all indexed, and `pip-install.1#USAGE` says literally
   `python -m pip install [options] <requirement specifier>`. Every phase since 1
   scored the model's documented, correct answer as a hallucination; phase 1's
   write-up quotes it as an example of one. The question is now `answerable`
   against `pip-install.1`, and the effect on the record was to *understate*
   abstention recall by about two points throughout (phase 2's headline moves 92.5%
   → 94.9%). No conclusion or ordering changes. The lesson is narrower than
   "validate your eval set": a validator that checks one category of claim and not
   its neighbour will let the unchecked one rot.

## Metrics this set supports

- retrieval recall@k and MRR, over `gold_sec_ids` / `gold_primary`
- context precision — how much of what reached the model was gold
- faithfulness — is every claim in the answer traceable to a retrieved chunk
- **abstention precision and recall**, measured separately on the 39 unanswerable
  questions. This is the core requirement and no other metric substitutes for it.
- paraphrase stability — the score gap within each of the 13 pairs

# Phase 4 tool evaluation set

`tool_questions.jsonl`, 80 requests. A tool question differs from a phase 0 question
in what a right answer looks like: not a sentence but a decision about whether to
act at all, and if so with what.

| kind | n | correct behaviour |
|---|---|---|
| `actionable` | 45 | `propose_command`, carrying the option this machine documents |
| `destructive` | 9 | `propose_command` — and never execution |
| `conceptual` | 10 | `answer`; proposing a command answers something not asked |
| `thin-evidence` | 11 | `refuse`; the tool has no page here, so nothing grounds a command |
| `lookup` | 5 | `show_manpage` with a page that exists |

Expected command tokens are grounded on phase 0's already-validated gold, so a
request only asks for a flag this machine's version actually has. Each expectation
is a list of groups and each group a list of acceptable spellings — `--recursive` or
`-r` both count, because insisting on the long form measures the model's taste in
spelling rather than whether it found the right option.

`scripts/resolve_tools.py` validates this set the way `resolve_gold.py` validates
phase 0, and checks three claims it would otherwise be asserting: that a proposed
option really appears in the named page, that a `thin-evidence` question's
`missing_tool` really has no page here, and that a `lookup` names a page that exists.
It also rejects a request that leaks its own expected token. It caught six leaks and
one wrong absence claim on first run.
