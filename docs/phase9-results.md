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

Re-derived directly from the answer JSONs — `data/eval/results/p7-A-answers.json`
(A), `data/eval/results/p8-B4g-answers.json` (B), and the orchestrator's
`p9-C-answers.json` / `p9-D-answers.json` (C, D), read from
`/tmp/claude-1000/-home-stupakzm-projects-slm-memory-management/c8ea9a11-ce22-4155-8f41-1fa70f61f461/scratchpad/p9-out/`
(not tracked in this worktree):

```
$ python3 -c "
import json
files = {
    'A': 'data/eval/results/p7-A-answers.json',
    'B': 'data/eval/results/p8-B4g-answers.json',
    'C': '.../p9-out/p9-C-answers.json',
    'D': '.../p9-out/p9-D-answers.json',
}
for name, path in files.items():
    d = json.load(open(path))
    recs = d['results']
    ans = [r for r in recs if r['kind'] == 'answerable']
    una = [r for r in recs if r['kind'] != 'answerable']
    correct = sum(1 for r in ans if r['correct'])
    correct_strict = sum(1 for r in ans if r['correct_strict'])
    ev = [r for r in ans if r['evidence_retrieved']]
    correct_given_ev = sum(1 for r in ev if r['correct'])
    abst = sum(1 for r in una if r['abstained'])
    no_ev = [r for r in ans if not r['evidence_retrieved']]
    answered_no_ev = sum(1 for r in no_ev if not r['abstained'])
    una_answered = sum(1 for r in una if not r['abstained'])
    unsupported = answered_no_ev + una_answered
    print(f'{name}: {correct}/127 ({correct_strict} strict) | {correct_given_ev}/{len(ev)} given evidence | '
          f'{abst}/39 unanswerable abstained | {answered_no_ev}/{len(no_ev)} answered w/o evidence | '
          f'{unsupported}/166 unsupported')
"
A: 81/127 (63 strict) | 76/90 given evidence | 38/39 unanswerable abstained | 23/37 answered w/o evidence | 24/166 unsupported
B: 82/127 (66 strict) | 78/104 given evidence | 37/39 unanswerable abstained | 14/23 answered w/o evidence | 16/166 unsupported
C: 91/127 (72 strict) | 79/90 given evidence | 35/39 unanswerable abstained | 28/37 answered w/o evidence | 32/166 unsupported
D: 89/127 (66 strict) | 83/104 given evidence | 33/39 unanswerable abstained | 17/23 answered w/o evidence | 23/166 unsupported
```

| arm | correct/127 (strict) | correct given evidence | unanswerable abstained/39 | answered w/o evidence / answerable-w/o-evidence | unsupported/166 |
|---|---|---|---|---|---|
| A | 81 (63) | 76/90 | 38 | 23/37 | 24 |
| B | 82 (66) | 78/104 | 37 | 14/23 | 16 |
| C | 91 (72) | 79/90 | 35 | 28/37 | 32 |
| D | 89 (66) | 83/104 | 33 | 17/23 | 23 |

**Sign tests** (discordant answerable questions only, exact two-sided sign
test, same protocol as phases 7 and 8):

```
$ python3 -c "
import json, math
def index(path):
    d = json.load(open(path))
    return {r['qid']: r for r in d['results'] if r['kind'] == 'answerable'}
def sign_test(up, down):
    n = up + down
    k = min(up, down)
    p = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n) * 2
    return min(p, 1.0)
def compare(ra, rb, label):
    up = down = 0
    for qid in ra:
        ca, cb = ra[qid]['correct'], rb[qid]['correct']
        if cb and not ca: up += 1
        elif ca and not cb: down += 1
    print(f'{label}: +{up}/-{down}, p={sign_test(up, down):.4f}')
A = index('data/eval/results/p7-A-answers.json')
B = index('data/eval/results/p8-B4g-answers.json')
C = index('.../p9-out/p9-C-answers.json')
D = index('.../p9-out/p9-D-answers.json')
compare(A, C, 'C vs A')
compare(A, D, 'D vs A')
compare(C, D, 'D vs C')
"
C vs A: +13/-3, p=0.0213
D vs A: +14/-6, p=0.1153
D vs C: +8/-10, p=0.8145
```

**VERDICT under the pre-registered rule (§5).**

- **PRIMARY: NULL.** C passes condition (a) — sign test p=0.0213 < 0.05 —
  and condition (c) — the win is not strict-label-only; strict correctness
  also rises, 63→72/127. But C **fails condition (b)**: unanswerable
  abstained falls to 35/39 against the pre-registered bar of ≥37/39 (A's
  38/39 − 1). The rule is an AND over (a), (b), (c); one failing condition
  is enough. **The primary is null and nothing ships**, even though the
  accuracy gain (81→91/127, sign test p=0.0213) is itself statistically
  significant. The gain is real by the test that measures it and is still
  not a win, because §5's rule was written so that accuracy cannot be
  bought with abstention — and here it was: of C's 4 unanswerable questions
  answered instead of abstained, 3 are newly lost relative to A (u13, u06.t,
  u06.z; A already answered u19), and all 4 are the tool-not-installed/
  out-of-corpus cases "Reader vs finder" documents below.
- **Secondary, reported and not a ship criterion on its own:** D vs A is not
  significant (p=0.1153); D vs C is not significant (p=0.8145) — the 4B
  reranker does not pay off even once paired with the stronger 30B reader,
  echoing phase 8's finding with the 4B reader.

## Reader vs finder

C's 13 gains over A (`a02, a11, a18, a25, a29, b07, p08, p09, a01.n, a15.n,
a21.n, a37.z, b12.t`): 6 had evidence in context (`a25, a29, b07, p09,
a15.n, a37.z`); 7 did **not** (`a02, a11, a18, p08, a01.n, a21.n, b12.t`) —
confirmed against each record's `evidence_retrieved` field:

```
$ python3 -c "
import json
C = {r['qid']: r for r in json.load(open('.../p9-out/p9-C-answers.json'))['results']}
gains = ['a02','a11','a18','a25','a29','b07','p08','p09','a01.n','a15.n','a21.n','a37.z','b12.t']
with_ev = [q for q in gains if C[q]['evidence_retrieved']]
no_ev = [q for q in gains if not C[q]['evidence_retrieved']]
print('with evidence:', len(with_ev), with_ev)
print('without evidence:', len(no_ev), no_ev)
"
with evidence: 6 ['a25', 'a29', 'b07', 'p09', 'a15.n', 'a37.z']
without evidence: 7 ['a02', 'a11', 'a18', 'p08', 'a01.n', 'a21.n', 'b12.t']
```

C's 3 losses (`a27, a30, c07`) are all answered wrong, not newly-abstained:

```
$ python3 -c "
import json
C = {r['qid']: r for r in json.load(open('.../p9-out/p9-C-answers.json'))['results']}
for q in ('a27','a30','c07'):
    print(q, 'abstained=', C[q]['abstained'], 'correct=', C[q]['correct'])
"
a27 abstained= False correct= False
a30 abstained= False correct= False
c07 abstained= False correct= False
```

Correct **without** evidence in context, over the 37 answerable questions
where no evidence reached the model:

```
$ python3 -c "
import json
def no_ev_correct(path):
    recs = json.load(open(path))['results']
    ans = [r for r in recs if r['kind']=='answerable']
    no_ev = [r for r in ans if not r['evidence_retrieved']]
    return sum(1 for r in no_ev if r['correct']), len(no_ev)
print('A:', no_ev_correct('data/eval/results/p7-A-answers.json'))
print('C:', no_ev_correct('.../p9-out/p9-C-answers.json'))
"
A: (5, 37)
C: (12, 37)
```

A 5/37 vs C 12/37 — more than half of C's net gain over A (13 discordant-up
questions) is getting answers right with nothing to read: the 30B's own
parametric knowledge of Linux, not better reading.

The four unanswerable questions C answers instead of abstaining are all
tool-not-installed or out-of-corpus cases, the 30B answering from its own
knowledge of Linux rather than the local corpus:

```
$ python3 -c "
import json
C = json.load(open('.../p9-out/p9-C-answers.json'))['results']
for r in C:
    if r['kind'] != 'answerable' and not r['abstained']:
        print(r['qid'], r['kind'], r['tags'], r['question'][:80])
"
u13 unanswerable ['tool-not-installed'] How do I sort processes by memory usage inside htop?
u19 unanswerable ['out-of-corpus'] How do I format a floating point number with printf in C?
u06.t unanswerable ['tool-not-installed', 'variant-terse'] strace syscalls of a process
u06.z unanswerable ['tool-not-installed', 'variant-typo'] trase the sytem calls a program makes
```

So more than half of C's raw gain is parametric (ungrounded), not grounded
— exactly what the pre-registered abstention bar guards against, and it
shows up in the unsupported rate too: 24→32/166 for C even as raw accuracy
rises.

The grounded part of the gain is real but small: correct given evidence
rises 76/90→79/90 for C vs A, and 78/104→83/104 for D vs B ("Per-arm
results" above) — a few points, not the ~10-point raw gain.

**Premise-gate cross-check.** §3's 29-question probe found the 30B right in
both contexts 11/29 with only 9/29 flipping between contexts, versus the
4B's 0/29 right-in-both and 17/29 flips — consistent with a more stable
reader on that adversarial subset. But on the full 166-question run the
grounded gain (correct-given-evidence) is small, and most of the raw gain
traces to parametric answers rather than better reading of retrieved
evidence.

## Cost: latency and memory

**Wall time** (orchestrator-reported, `p9-arms.log`; not independently
re-timed — no per-question timing is stored in the answer JSONs):

```
$ cat /tmp/claude-1000/-home-stupakzm-projects-slm-memory-management/c8ea9a11-ce22-4155-8f41-1fa70f61f461/scratchpad/p9-out/p9-arms.log
arm C (cache p7-k20 gate 0.65) exit=0 1383s
arm D (cache p8-k20 gate 0.7631053338514099) exit=0 1354s
```

```
$ python3 -c "print(1383/166, 1354/166)"
8.331325301204819 8.156626506024097
```

166 questions each: ~8.3 s/question for C, ~8.2 s/question for D — call it
~8.3 s/question for the 30B-A3B generator.

The corresponding figure for the shipped 4B generator is **not measured in
this phase**: docs/phase7-results.md and docs/phase8-results.md's cost
sections instrument reranker/retrieval latency (`docs/phase8-results.md`,
"Cost: latency and memory" — 0.6B vs 4B *reranker* seconds/question) but
none of phases 6-8's docs record the 4B *generator's* own seconds/question,
so no like-for-like comparison is available from prior phases; this is a
gap in this phase's own instrumentation, not a claim of parity or speedup.

**Memory** (as configured in §2, not independently re-measured here): the
30B-A3B Q4_K_M model is 18,556,686,752 bytes (~18.6 GB, §2) and, served with
`--cpu-moe`, sits in system RAM rather than VRAM; only attention and
non-expert tensors go to the GPU, at roughly 1.5 GB VRAM — cheap enough to
coexist with the embedder and reranker already resident (§2). System RAM on
this machine is 31 GB total, so an 18.6 GB model leaves headroom for the OS,
embedder, and reranker, but not a large margin beyond that.

## What this changes

**Null, and nothing ships.** Phase 9 separates two things a single
"correct" number had merged: a stronger reader is right more often
(81→91/127, sign test p=0.0213; strict also up, 63→72), but "Reader vs
finder" above shows more than half of that gain — 7 of 13 discordant-up
questions, plus 4 previously-abstained unanswerable questions now answered
— is the 30B answering from its own knowledge of Linux rather than from the
retrieved extracts, including answers about tools (`htop`, `strace`) not
installed on this machine. For a system whose stated requirement is "answer
from the manual on this machine, or abstain," that is a regression in the
property the system exists to have, not a gain — even though the top-line
accuracy number rose and is itself statistically significant. The
pre-registered rule (§5) exists to catch exactly this failure mode —
accuracy bought with reduced abstention — and condition (b) (abstained
≥37/39) does its job here: C falls to 35/39, so the primary is null.

Secondary comparisons do not change the picture: D vs A is not significant
(p=0.1153) and D vs C is not significant (p=0.8145) — the 4B reranker does
not pay off even once paired with the stronger 30B reader, echoing phase
8's finding that the reranker upgrade alone does not move end-to-end
correctness.

**Open questions**, named here and not designed in this phase:
- A gate or abstention policy recalibrated for the 30B specifically — §4
  ("The gate thresholds stay tied to their reranker...") marked a gate
  re-sweep for the new generator out of scope for this phase; it may be
  what is needed to let a stronger reader's grounded gains through without
  also letting its parametric guesses through.
- Prompting or grammar changes that forbid ungrounded answers outright, so
  a stronger model's broader knowledge cannot substitute for missing
  evidence in context.

Neither is designed here; either would need its own pre-registration before
a future phase could act on it.
