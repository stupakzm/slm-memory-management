# Phase 9 — a stronger generator, pre-registered

This document is written before phase 9's own arms produce a single number —
no 166-question answer from the new generator exists yet, and none is
predicted here. What follows fixes the question, the model and its
provenance, a premise gate that was written and hashed before the candidate
model produced any answer, the four arms and exactly what differs between
them, and the pre-registered win criterion, in that order, so that none of it
can be adjusted after seeing an arm's numbers. The four results sections at
the end are placeholders.

## 1. Question

Phases 6-8 moved the finding side of the pipeline but not correctness. Phase
8's stronger reranker put evidence in the top 5 for 104/127 answerable
questions versus 90 under the shipped 0.6B reranker, yet aliased correct only
moved 81→82/127 (+9/−8 discordant, sign test p=1.0;
`docs/phase8-results.md`, "Per-arm results" and "Verdict under §6"). Of B's 8
losses, 5 had the *same* gold evidence in context under both rerankers and
were misread once the other four extracts in the context changed
(`docs/phase8-results.md`, "Reranker vs gate", "Discordant losses under B").
The diagnosis phase 8 leaves open is that the reader — Qwen3-4B-Instruct-2507
Q4_K_M, the generator every arm through phase 8 has shared — is the binding
constraint, not the retriever. This phase asks: does a stronger reader
convert evidence already in context into correct answers?

## 2. Model and provenance

- **Candidate**: Qwen3-30B-A3B-Instruct-2507, Q4_K_M, from
  `unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` — the same quantizer
  (unsloth) as the current 4B generator, so the quantization-recipe variable
  is held constant between old and new generator. 18,556,686,752 bytes,
  sha256 `6c997b8af17debdfb01d890214400ccbab00db6acc0ba8da5de1cc906c4774d0`.
- **Architecture**: mixture-of-experts, ~3B active parameters per token
  despite the 30B total. Served as:
  ```
  llama-server -m models/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf \
      --cpu-moe -ngl 99 -c 4096 --temp 0.0 -t 12 --port 8083
  ```
  Expert tensors stay in system RAM (`--cpu-moe`); attention and the rest of
  the graph run on the GPU (`-ngl 99`), at roughly 1.5 GB VRAM — cheap enough
  to coexist with the embedder and reranker already resident, unlike the
  4B-parameter reranker candidate phase 8 costed out at ~4.2 GB
  (`docs/phase8-results.md`, "Cost: latency and memory").
- **Alternatives considered, and why not chosen**: Qwen3-8B (official) is a
  pre-2507 hybrid-thinking checkpoint and runs ~5.9 GB with KV cache on a
  6 GB card — a much smaller safety margin than the MoE's ~1.5 GB, for a
  dense model that predates the 2507 instruction-tuning refresh the current
  4B benefits from. Qwen3-4B Q8_0 (same parameter count, higher-precision
  quantization) was rejected because it tests quantization noise, not reader
  capacity — it would not answer this phase's question at all.

## 3. Premise gate: pre-fixed rule and result

The rule was written and hashed **before** the 30B produced any answer.
Quoted verbatim from
`/tmp/claude-1000/-home-stupakzm-projects-slm-memory-management/c8ea9a11-ce22-4155-8f41-1fa70f61f461/scratchpad/p9-in/p9-gate-rule.txt`,
sha256 `64231dfa3bbc58a05f94cceac9870c5ee47c48cd278ce65388e88877a4d674d0`
(re-derived below), timestamped 2026-09-26T19:23:18+02:00:

```
Phase 9 premise gate - fixed BEFORE the 30B-A3B produced any answer (2026-09-26).
Set: 29 answerable qids (p9-probe-qids.json): 17 where the 4B generator's aliased
correctness differs between the 0.6B-reranked and 4B-reranked top-5 contexts
(p8-A0u vs p8-B4u, ungated), plus 12 wrong-with-evidence under both.
4B baseline on this set: right in both contexts 0/29; cells correct 17/58.
PASS iff the 30B-A3B (ungated, grammar, k=5, same two caches) is
  (1) right in BOTH contexts on >= 10 of 29, AND
  (2) correct on >= 29 of 58 cells.
Caveat: the set is selected on the 4B's failures, so any other model benefits from
regression to the mean; passing licenses a full pre-registered 166-question phase,
it does not show a win.
```

Hash check:

```
$ sha256sum /tmp/claude-1000/.../scratchpad/p9-in/p9-gate-rule.txt
64231dfa3bbc58a05f94cceac9870c5ee47c48cd278ce65388e88877a4d674d0  .../p9-gate-rule.txt
```

Matches the hash above exactly.

**Re-derivation.** The rule references four files, all read from the
orchestrator's scratchpad at
`/tmp/claude-1000/-home-stupakzm-projects-slm-memory-management/c8ea9a11-ce22-4155-8f41-1fa70f61f461/scratchpad/p9-in/`
except `p8-A0u-answers.json` and `p8-B4u-answers.json`, which are also
present and identical in this worktree at `data/eval/results/`:

- `p9-probe-qids.json` — 29 qids, confirmed by count below.
- `p9probe-ctx06-answers.json` / `p9probe-ctx4b-answers.json` — the 30B on
  those 29 questions, ungated (`gate: 0.0`), grammar on, k=5 shown to the
  model, reading the 0.6B-reranked cache (`cache: p7-k20`) and the
  4B-reranked cache (`cache: p8-k20`) respectively (configs confirmed
  identical apart from `cache`).
- `p8-A0u-answers.json` / `p8-B4u-answers.json` — the 4B generator, ungated,
  on all 166 questions, same two caches (`p7-k20` / `p8-k20` respectively) —
  used here only for the 29-qid subset the rule names.

```
$ python3 -c "
import json
qids = json.load(open('.../p9-in/p9-probe-qids.json'))
print('n qids', len(qids))
"
n qids 29
```

```
$ python3 -c "
import json

def index(path):
    d = json.load(open(path))
    return {r['qid']: r for r in d['results']}

ctx06_30b = index('.../p9-in/p9probe-ctx06-answers.json')
ctx4b_30b = index('.../p9-in/p9probe-ctx4b-answers.json')
A0u_4b    = index('data/eval/results/p8-A0u-answers.json')
B4u_4b    = index('data/eval/results/p8-B4u-answers.json')
qids = json.load(open('.../p9-in/p9-probe-qids.json'))

def summarize(name, ctx06, ctx4b, qids):
    right_both = cells = flips = 0
    for q in qids:
        c06, c4b = ctx06[q]['correct'], ctx4b[q]['correct']
        cells += int(c06) + int(c4b)
        if c06 and c4b: right_both += 1
        if c06 != c4b: flips += 1
    print(f'{name}: right in both {right_both}/{len(qids)}; '
          f'cells {cells}/{2*len(qids)}; flips {flips}/{len(qids)}')

summarize('4B  (p8-A0u/p8-B4u)', A0u_4b, B4u_4b, qids)
summarize('30B (probe)        ', ctx06_30b, ctx4b_30b, qids)
"
4B  (p8-A0u/p8-B4u): right in both 0/29; cells 17/58; flips 17/29
30B (probe)        : right in both 11/29; cells 31/58; flips 9/29
```

Also confirmed: of the 29 qids, exactly 17 are 4B-discordant between contexts
and 12 are wrong-with-evidence under both (`differ 17`, `both wrong 12`,
summing to 29) — matching the rule's own description of how the set was
built, not just its headline counts.

**Result against the pre-fixed rule**:

| condition | threshold | 30B result | pass? |
|---|---|---|---|
| (1) right in both contexts | ≥10/29 | 11/29 | pass, by 1 |
| (2) correct cells | ≥29/58 | 31/58 | pass, by 2 |

**PASS on both conditions, narrowly.**

**Speed**: ~11 s/question for the 30B on these prompts — 328 s for the
29-question 0.6B-context run and 321 s for the 29-question 4B-context run
(`.../scratchpad/p9probe-ctx06.log`, `.../scratchpad/p9probe-ctx4b.log`;
328/29 = 11.3 s/q, 321/29 = 11.1 s/q).

**Selection-bias caveat, stated plainly**: this 29-question set was chosen
*from the 4B's own failures* on these two contexts (17 where the 4B flips
between contexts, plus 12 it gets wrong under both) — it is not a random or
representative sample of the 166-question eval. Any model that is not
identically wrong to the 4B benefits from regression to the mean on a set
built this way, independent of whether it is actually a better reader. A
pass licenses running the full pre-registered 166-question phase below; it
does not, by itself, show that the 30B is a win — that is what §5's decision
rule over all 166 questions is for.

## 4. Arms

All 166 questions: `main.db`, dense retrieval + rerank, k=5 shown to the
model, grammar on, the aliased answer label
(`docs/phase6-results.md`, "Correction (2026-09-26)"). The retrieval caches
are reused unchanged from phases 7 and 8 — p7-k20 (0.6B reranker) and p8-k20
(4B reranker) — so retrieval itself is held fixed; only the generator
changes across A/C and B/D.

| arm | generator | reranker | gate | source |
|---|---|---|---|---|
| A | 4B (shipped) | 0.6B | 0.65 | `data/eval/results/p7-A-answers.json` (81/127, 38/39, re-derived below) |
| B | 4B (shipped) | 4B | 0.7631 | `data/eval/results/p8-B4g-answers.json` (82/127, 37/39, re-derived below) |
| C | 30B-A3B | 0.6B | 0.65 | not yet generated |
| D | 30B-A3B | 4B | 0.7631 | not yet generated |

```
$ python3 -c "
import json
def stats(path):
    d = json.load(open(path))
    recs = d['results']
    ans = [r for r in recs if r['kind']=='answerable']
    una = [r for r in recs if r['kind']!='answerable']
    correct = sum(1 for r in ans if r['correct'])
    abst = sum(1 for r in una if r['abstained'])
    print(path, f'{correct}/{len(ans)}', f'{abst}/{len(una)}')
stats('data/eval/results/p7-A-answers.json')
stats('data/eval/results/p8-B4g-answers.json')
"
data/eval/results/p7-A-answers.json 81/127 38/39
data/eval/results/p8-B4g-answers.json 82/127 37/39
```

Matches the packet's own figures for A and B exactly.

The gate thresholds stay tied to their reranker, not to the generator: the
gate thresholds `top_score`, the reranker's own top-1 relevance score
(`scripts/sweep_gate.py:60`), which is fixed once retrieval and reranking are
done and does not depend on which model reads the resulting context
(`scripts/sweep_gate.py:67-70`). A generator swap cannot move `top_score`, so
re-sweeping the gate for the new generator is out of scope for this phase and
could only be reported as secondary if it were done at all.

## 5. Decision rule, pre-registered

- **PRIMARY**: C beats A on aliased correct/127, two-sided exact sign test
  p < 0.05 over discordant answerable questions only (same protocol as
  phases 7 and 8), **AND** C's unanswerable abstained ≥ A's − 1 (≥37/39),
  **AND** no win is claimed from strict-label numbers alone — a
  strict-label-only improvement with a flat or worse aliased result does not
  count.
- **SECONDARY**, reported and not a ship criterion on its own: D vs A under
  the same test, and D vs C (does the 4B reranker pay off once paired with a
  stronger reader, where it did not pay off with the 4B reader in phase 8?).
- **Also reported, not part of the primary/secondary gate**: correct given
  evidence, answered-with-no-evidence, unsupported/166, and seconds per
  question, for all four arms.
- A primary win means a separate task to make the generator selectable
  (mirroring `SMM_RERANK_MODEL`'s pattern for the reranker,
  `docs/phase8-results.md` §"What this changes" naming a stronger generator
  as the open question) and to document the RAM/latency cost measured here;
  short of a win, nothing ships and the result is reported null, the
  convention `docs/phase6-results.md`, `docs/phase7-results.md`, and
  `docs/phase8-results.md` all used.

**Power.** The sign test runs only over whatever subset of the 127
answerable questions actually flips between A and C (or A and D, or C and
D) — not all 127, and not fixed ahead of time (same caveat as phases 6-8).
The smallest discordant split that still clears p < 0.05, at a few
illustrative totals (exact two-sided sign test):

```
$ python3 -c "
import math
def sign_test(up, down):
    n = up + down
    k = min(up, down)
    p = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n) * 2
    return min(p, 1.0)
for n in (10, 15, 20, 25, 30):
    best = None
    for k in range(0, n // 2 + 1):
        up, down = n - k, k
        p = sign_test(up, down)
        if p < 0.05:
            best = (up, down, p)
    print(n, best)
"
10 (9, 1, 0.021484375)
15 (12, 3, 0.03515625)
20 (15, 5, 0.04138946533203125)
25 (18, 7, 0.04328656196594238)
30 (21, 9, 0.04277753816850279)
```

| discordant n | most balanced split still significant | p |
|---|---|---|
| 10 | 9 – 1 | 0.0215 |
| 15 | 12 – 3 | 0.0352 |
| 20 | 15 – 5 | 0.0414 |
| 25 | 18 – 7 | 0.0433 |
| 30 | 21 – 9 | 0.0428 |

Identical in structure to phases 7 and 8's own tables — this is a property
of the sign test's dependence on n alone, not of this phase's data, extended
here to n=25 and n=30 since a 30B reader changing more answers than the
4B-vs-4B reranker swap did is plausible given §3's probe result (11/29
already flip favorably on the pre-registration set alone).

## Per-arm results

RESULTS PENDING

## Reader vs finder

RESULTS PENDING

## Cost: latency and memory

RESULTS PENDING

## What this changes

RESULTS PENDING
