# Phase 8 — a stronger reranker, pre-registered

This document is written before phase 8's own arms produce a single number —
no phase 8 answer exists yet. What follows fixes the question, the file this
phase depends on and why it is the only third-party conversion that passed a
premise gate, the premise measured on the 0.6B's own top-20 pool, the three
arms and exactly what differs between them, the gate protocol for putting A′
and B on a fair footing, and the pre-registered win criterion, in that
order, so that none of it can be adjusted after seeing an arm's numbers. The
four results sections at the end are placeholders.

## 1. Question

Phase 7 found that widening the context the 4B generator reads finds more
evidence but reads it worse: evidence in context rose 90→101/127 at k=8, but
correct-given-evidence fell 84.4%→80.2% (`docs/phase7-results.md`, "Per-arm
results"; arm B). The lever tried there was *k shown to the model*. This
phase tries a different lever: put the evidence higher in a *fixed* k=5 by
using a stronger reranker, so the model's context does not get longer or
noisier at all. The current reranker is Qwen3-Reranker-0.6B Q8_0
(`scripts/fetch_models.py:17`). The candidate is Qwen3-Reranker-4B Q4_K_M
from `pyarn/Qwen3-Reranker-4B-Q4_K_M-GGUF`
(sha256 `5b798f2918b6bc2c79dc106d83428e052b2efdcbe54f517f05a958a5c4c4d65a`).

## 2. Why this file (provenance)

There is no official GGUF for Qwen3-Reranker-4B. Ten third-party conversions
were checked by reading their GGUF headers over HTTP range requests, without
downloading the weights. Only `pyarn/Qwen3-Reranker-4B-Q4_K_M-GGUF` declares
`qwen3.pooling_type = 4` (rank pooling) and carries a `cls.output.weight`
tensor with yes/no classifier labels — the same structure the working 0.6B
reranker uses. The most-downloaded candidate (mradermacher's conversion) was
fetched first and failed this check: no classifier head, no rank pooling
declared in its metadata. It was deleted before any use. This is reported as
the premise gate it was — a file that does not carry rank-pooling structure
cannot produce the single relevance score the pipeline's gate and reranking
both depend on, whatever its download count says about it.

## 3. Premise gate, measured

Re-derived from
`/tmp/claude-1000/-home-stupakzm-projects-slm-memory-management/c8ea9a11-ce22-4155-8f41-1fa70f61f461/scratchpad/gate2-premise.json`
— untracked, not part of this worktree, read from the orchestrator's
scratchpad — which is `{qid: {"s06": [...20 scores...], "s4": [...20
scores...], "ev": [...20 bools...]}}` for 127 answerable questions. The pool
for every qid is the 0.6B's own top 20 (`data/eval/results/p7-k20-retrieved.json`,
untracked, not in this worktree, not read directly); `s06` and `s4` rescoring
that fixed pool with the 0.6B and 4B respectively, `ev` marking which of the
20 chunks actually carries the gold evidence. Confirmed: the file's 127 qids
are exactly `p7-A-answers.json`'s 127 answerable qids, and every pool is
exactly 20 chunks long (`len(rec['s06']) == len(rec['s4']) == len(rec['ev'])
== 20` for all 127 records).

**Evidence anywhere in the pool, rank of first evidence after sorting by
each model's score, and within-pool AUC**:

```
$ python3 -c "
import json
d = json.load(open('/tmp/claude-1000/-home-stupakzm-projects-slm-memory-management/c8ea9a11-ce22-4155-8f41-1fa70f61f461/scratchpad/gate2-premise.json'))

def first_hit_rank(scores, ev):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    for rank, i in enumerate(order, start=1):
        if ev[i]:
            return rank
    return None

def auc_q(scores, ev):
    pos = [scores[i] for i in range(len(scores)) if ev[i]]
    neg = [scores[i] for i in range(len(scores)) if not ev[i]]
    if not pos or not neg:
        return None
    n = wins = 0
    for g in pos:
        for x in neg:
            n += 1
            wins += 1 if g > x else (0.5 if g == x else 0)
    return wins / n

def stats(key):
    n_any_ev = r1 = r3 = r5 = 0
    aucs = []
    for qid, rec in d.items():
        ev = rec['ev']
        if any(ev):
            n_any_ev += 1
        fhr = first_hit_rank(rec[key], ev)
        if fhr is not None:
            r1 += fhr <= 1; r3 += fhr <= 3; r5 += fhr <= 5
        a = auc_q(rec[key], ev)
        if a is not None:
            aucs.append(a)
    return n_any_ev, r1, r3, r5, sum(aucs)/len(aucs)

for key in ('s06', 's4'):
    print(key, stats(key))
"
s06 (113, 56, 80, 90, 0.7343464137706788)
s4  (113, 61, 87, 104, 0.8164244460059242)
```

For both models: evidence is present somewhere in the shared 20-chunk pool
for 113/127 questions (this does not change with the scoring model — same
pool, only the order changes). Of those 113: rank ≤1 rises 56→61, rank ≤3
rises 80→87, rank ≤5 rises 90→104. Mean within-pool AUC (evidence chunk
score vs non-evidence chunk score, averaged per question, over the 113
questions with at least one of each) rises 0.7343→0.8164.

**Movement, per question** — which model puts the first evidence chunk
higher:

```
$ python3 -c "
import json
d = json.load(open('/tmp/claude-1000/-home-stupakzm-projects-slm-memory-management/c8ea9a11-ce22-4155-8f41-1fa70f61f461/scratchpad/gate2-premise.json'))
def first_hit_rank(scores, ev):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    for rank, i in enumerate(order, start=1):
        if ev[i]:
            return rank
    return None
up = down = same = noev = 0
for qid, rec in d.items():
    ev = rec['ev']
    if not any(ev):
        noev += 1; continue
    r06 = first_hit_rank(rec['s06'], ev)
    r4 = first_hit_rank(rec['s4'], ev)
    if r4 < r06: up += 1
    elif r4 > r06: down += 1
    else: same += 1
print('up', up, 'down', down, 'same', same, 'no-evidence-anywhere', noev)
"
up 37 down 24 same 52 no-evidence-anywhere 14
```

The 4B moves the first evidence chunk to a better rank on 37 questions and
to a worse rank on 24; 52 are unchanged (same first-hit rank under both
scorings — often because the first evidence chunk is already rank 1 under
both, or because it is absent from the top-5-ish window either way); 14 have
no evidence anywhere in the shared pool, so no rescoring can move anything
for them.

**Rank check against the orchestrator's own figures**: this reproduces
rank≤1 56→61, rank≤3 80→87, rank≤5 90→104, AUC 0.734→0.816, up 37/down 24
exactly as stated in the packet.

**The 18 no-evidence-wrong questions from phase 7's arm A** (a02 a04 a11 a18
a35 a40 a42 b13 c03 c04 p02 p06 a11.t a15.z a25.n a35.z a37.t b12.t) —
first-evidence rank in the shared 20-chunk pool, under each model:

```
$ python3 -c "
import json
d = json.load(open('/tmp/claude-1000/-home-stupakzm-projects-slm-memory-management/c8ea9a11-ce22-4155-8f41-1fa70f61f461/scratchpad/gate2-premise.json'))
qids18 = 'a02 a04 a11 a18 a35 a40 a42 b13 c03 c04 p02 p06 a11.t a15.z a25.n a35.z a37.t b12.t'.split()
def first_hit_rank(scores, ev):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    for rank, i in enumerate(order, start=1):
        if ev[i]:
            return rank
    return None
for qid in qids18:
    rec = d[qid]
    ev = rec['ev']
    if not any(ev):
        print(qid, 'no evidence in top-20 pool'); continue
    print(qid, '0.6B rank', first_hit_rank(rec['s06'], ev), '4B rank', first_hit_rank(rec['s4'], ev))
"
a02   0.6B rank 16  4B rank 16
a04   0.6B rank 6   4B rank 11
a11   0.6B rank 6   4B rank 18
a18   0.6B rank 8   4B rank 2
a35   no evidence in top-20 pool
a40   0.6B rank 8   4B rank 1
a42   0.6B rank 11  4B rank 1
b13   0.6B rank 6   4B rank 5
c03   no evidence in top-20 pool
c04   0.6B rank 7   4B rank 9
p02   0.6B rank 20  4B rank 1
p06   no evidence in top-20 pool
a11.t 0.6B rank 11  4B rank 12
a15.z 0.6B rank 19  4B rank 15
a25.n 0.6B rank 18  4B rank 4
a35.z no evidence in top-20 pool
a37.t 0.6B rank 15  4B rank 3
b12.t 0.6B rank 7   4B rank 2
```

14 of the 18 have evidence somewhere in the pool; the 4 without it (a35,
c03, p06, a35.z) match phase 7 §2's own count of questions with no hit at
all in the top 20 exactly, so this is the same ceiling phase 7 already
found, not a new one. Of the 14, the 4B moves evidence to a better rank
than the 0.6B on 9 (a18, a40, a42, b13, p02, a15.z, a25.n, a37.t, b12.t) and
to a worse rank on 4 (a04, a11, c04, a11.t); one is unchanged (a02, rank 16
under both). Of the 9 improvements, 8 land at or inside rank 5 under the
4B, where the 0.6B's best rank across all 14 questions is 6 — none of them
were ≤5 under the 0.6B to begin with; the ninth (a15.z) improves from 19 to
15, still outside the top 5 either way.

**Smoke scores** (reported by the orchestrator, not independently
re-derived here — no reranker weights of either model are present in this
worktree, `find . -iname '*.gguf'` returns nothing, so no inference is
possible without a network fetch, which this task does not permit): for the
query "how do I list files sorted by modification time", the 4B scored
"-t sort by time, newest first" 0.769, "-S sort by file size" 0.042, and an
off-topic recipe 0.0.

**Caveat**: this premise gate reranks the 0.6B's own top-20 pool — the same
20 candidates the 0.6B already surfaced, just re-ordered. Phase 8's actual
arm B reranks the full 50 candidates the retriever returns, so it can also
surface evidence the 0.6B's top 20 never included at all (the pool used
here cannot show that effect either way — its ceiling is fixed at whatever
the 0.6B's own top 20 contains).

## 4. Arms

All three arms share `main.db`, dense retrieval, k=5 shown to the model,
grammar on, and the aliased answer label
(`docs/phase6-results.md`, "Correction (2026-09-26)"):

| arm | reranker | candidates reranked | gate | source |
|---|---|---|---|---|
| A | Qwen3-Reranker-0.6B Q8_0 | 50 | 0.65 (as shipped) | `data/eval/results/p7-A-answers.json`, 81/127 correct, 38/39 abstained |
| A′ | Qwen3-Reranker-0.6B Q8_0 | 50 | re-swept (§5) | not yet generated |
| B | Qwen3-Reranker-4B Q4_K_M | 50 (top 20 cached, model reads 5) | re-swept (§5) | not yet generated |

A is phase 7's arm A as shipped, unchanged, reproduced here as the fixed
reference every other arm is compared against — it is not regenerated. A′
holds the reranker fixed and only re-sweeps the gate, isolating the gate
re-sweep's own effect from the reranker swap. B swaps the reranker and
re-sweeps its gate the same way. B reranks all 50 retrieval candidates (not
just the 0.6B's top 20, unlike §3's premise check); its top 20 are cached,
and the model reads the top 5 of that cache.

## 5. Gate protocol

The 0.65 threshold currently shipped was swept on the 0.6B's own score
distribution. The 4B's scores live on a different scale (§3's smoke scores
and the `s4` distribution above are not on the same footing as `s06`'s), so
reusing 0.65 for B would repeat the fusion-gate mistake already on file
(`blk_fusion_gate_semantic_slip`): thresholding one model's scores with a
number chosen for a different model's distribution.

For A′ and B: generate answers UNGATED (`--gate 0`), then run
`scripts/sweep_gate.py --answers <run>`, which applies its default rule
(`scripts/sweep_gate.py:99-126`, `sweep_answers`) — the lowest `unsupported`
rate subject to `accuracy >= ungated_accuracy - max_accuracy_loss` (default
`--max-accuracy-loss 0.02`, i.e. accuracy no more than 2 points below the
ungated run's own accuracy) — then generate gated at the chosen threshold.
This is the same in-sample procedure that chose 0.65 in the first place;
this document says so explicitly rather than presenting a re-swept
threshold as if it were held out.

## 6. Decision rule, pre-registered

B wins only if **all** of the following hold:

(a) B beats A on aliased correct/127, two-sided sign test p < 0.05, computed
    over discordant answerable questions only (same protocol as phase 7 §4,
    `docs/phase7-results.md`);
(b) B's unanswerable abstained ≥ A's minus 1 (A = 38/39, so B needs ≥37/39);
(c) no win is claimed from strict-label numbers alone — a strict-label
    improvement with a flat or worse aliased result does not count.

B vs A′ is reported as secondary, not part of the win criterion: it
separates the reranker swap's own effect from the gate re-sweep's effect,
since A′ isolates the latter alone.

If B wins, a separate task makes the 4B the default reranker in
`scripts/ask.py` and `asq`; short of a win, nothing ships and the result is
reported null, the same convention `docs/phase6-results.md` and
`docs/phase7-results.md` both used.

Also reported, not part of the gate on either arm's accuracy numbers: mean
retrieval seconds per question for B vs the 0.6B (phase 7's k=20 retrieval
run), and VRAM (4B reranker alone, roughly 4.2 GB at the query-sized
settings this project runs at; the current 0.6B profile holds embedder +
reranker + generator together in roughly 5.6 GB) — cost figures, reported
alongside the accuracy result, not folded into (a)-(c).

**Power.** The sign test in (a) runs only over whatever subset of questions
actually flips between A and B — not all 127, and not fixed ahead of time
(same caveat as phase 6 §5 and phase 7 §5). The smallest discordant split
that still clears p < 0.05 at a few illustrative totals (exact two-sided
sign test):

```
$ python3 -c "
import math
def sign_test(up, down):
    n = up + down
    k = min(up, down)
    p = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n) * 2
    return min(p, 1.0)
for n in (10, 15, 20):
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
```

| discordant n | most balanced split still significant | p |
|---|---|---|
| 10 | 9 – 1 | 0.0215 |
| 15 | 12 – 3 | 0.0352 |
| 20 | 15 – 5 | 0.0414 |

Identical to phase 7's own table (`docs/phase7-results.md` §5) — this is a
property of the sign test's dependence on n alone, not of this phase's data,
so the same numbers apply unchanged.

## Per-arm results

RESULTS PENDING

## Reranker vs gate

RESULTS PENDING

## Cost: latency and memory

RESULTS PENDING

## What this changes

RESULTS PENDING
