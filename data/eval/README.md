# Phase 0 evaluation set

111 questions against the man pages installed on this machine. Written before any
retrieval system exists, which is the whole point: it is the only thing that can
tell you whether a later phase actually helped.

## Composition

| | count | |
|---|---|---|
| Answerable | 81 | 71 unique + 10 paraphrases |
| Unanswerable | 30 (27%) | 27 unique + 3 paraphrases |
| Distinct gold documents | 56 | |
| Paraphrase pairs | 13 | tests the 11–19% paraphrase fragility from Finding 05 |

Answerable questions break down as 48 flag lookups, 18 config-format questions
(man5), 13 concept questions (man7), and 24 tagged `exact-token` — queries whose
answer is a literal string like `--exclude-from`, `@reboot` or `status=LEVEL`,
where BM25 should beat dense retrieval and phase 2 can prove it.

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
   error rather than an abstention case.

## Metrics this set supports

- retrieval recall@k and MRR, over `gold_sec_ids` / `gold_primary`
- context precision — how much of what reached the model was gold
- faithfulness — is every claim in the answer traceable to a retrieved chunk
- **abstention precision and recall**, measured separately on the 30 unanswerable
  questions. This is the core requirement and no other metric substitutes for it.
- paraphrase stability — the score gap within each of the 13 pairs
