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

Re-derived directly from the answer records (`data/eval/results/p7-A-answers.json`
for A, and the untracked `p8-A0g-answers.json` / `p8-B4g-answers.json` for A′
and B — read from the orchestrator's run, not present in this worktree, so
quoted with attribution and cross-checked field by field below):

```
$ python3 -c "
import json
def load(p): return json.load(open(p))
def index(d): return {r['qid']: r for r in d['results']}
A  = index(load('data/eval/results/p7-A-answers.json'))
Ap = index(load('<orchestrator p8-A0g-answers.json>'))
B  = index(load('<orchestrator p8-B4g-answers.json>'))

def row(idx):
    recs = list(idx.values())
    ans = [r for r in recs if r['kind'] == 'answerable']
    una = [r for r in recs if r['kind'] != 'answerable']
    correct = sum(1 for r in recs if r.get('correct'))
    strict = sum(1 for r in recs if r.get('correct_strict'))
    ev = sum(1 for r in ans if r.get('evidence_retrieved'))
    cge = sum(1 for r in ans if r.get('evidence_retrieved') and r.get('correct'))
    una_abst = sum(1 for r in una if r['abstained'])
    awoe = [r for r in ans if not r.get('evidence_retrieved')]
    no_ev_answered = sum(1 for r in awoe if not r['abstained'])
    gated = sum(1 for r in recs if r['gated'])
    print(f'{correct} ({strict}) | {ev} | {cge}/{ev} | {una_abst}/{len(una)} | {no_ev_answered}/{len(awoe)} | {gated}')

for name, idx in (('A', A), (\"A'\", Ap), ('B', B)):
    print(name, end=': '); row(idx)
"
A: 81 (63) | 90 | 76/90 | 38/39 | 23/37 | 33
A': 81 (63) | 90 | 76/90 | 38/39 | 23/37 | 33
B: 82 (66) | 104 | 78/104 | 37/39 | 14/23 | 43
```

| arm | correct/127 (strict) | evidence in context | correct given evidence | unanswerable abstained/39 | answered with no evidence / answerable-without-evidence | gated (all 166) |
|---|---|---|---|---|---|---|
| A shipped (0.6B, 0.65) | 81 (63) | 90 | 76/90 | 38 | 23/37 | 33 |
| A′ (0.6B, 0.5806) | 81 (63) | 90 | 76/90 | 38 | 23/37 | 33 |
| B (4B, 0.7631) | 82 (66) | 104 | 78/104 | 37 | 14/23 | 43 |

A′ is identical to A on every count above — the gate re-sweep alone changes
nothing end-to-end (§"Reranker vs gate" confirms it is the same 33 questions,
not merely the same count).

**Sign tests over discordant answerable questions** (exact two-sided, same
protocol as phase 7 §4):

```
$ python3 -c "
import json, math
def load(p): return json.load(open(p))
def index(d): return {r['qid']: r for r in d['results']}
A  = index(load('data/eval/results/p7-A-answers.json'))
Ap = index(load('<orchestrator p8-A0g-answers.json>'))
B  = index(load('<orchestrator p8-B4g-answers.json>'))
qids = sorted(q for q, r in A.items() if r['kind'] == 'answerable')

def sign_test(up, down):
    n = up + down
    k = min(up, down)
    p = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n) * 2
    return min(p, 1.0)

def compare(idx1, idx2, name):
    up = down = 0
    for q in qids:
        c1, c2 = idx1[q].get('correct'), idx2[q].get('correct')
        if c1 == c2: continue
        if c2 and not c1: up += 1
        elif c1 and not c2: down += 1
    print(f'{name}: +{up}/-{down} p={sign_test(up, down)}')

compare(A, B, 'B vs A')
compare(Ap, B, \"B vs A'\")
compare(A, Ap, \"A' vs A\")
"
B vs A: +9/-8 p=1.0
B vs A': +9/-8 p=1.0
A' vs A: +0/-0 p=None
```

Gains for B vs A (9): a18 a25 a40 p02 a01.n a11.t a15.z a37.t b12.t.
Losses (8): a27 a30 b06 b11 c12 p10 a04.z b03.n. B vs A′ is exactly the same
discordant set — A′ contributes nothing beyond A here, as expected since A′
= A on every per-record outcome. A′ vs A is 0 discordant questions, so no
sign test applies.

**Verdict under §6**: B fails (a) — beats A on aliased correct/127 (82 vs
81) but the sign test over the 17 discordant questions gives p=1.0, far
above the p<0.05 bar. It passes (b): B's unanswerable abstained is 37/39,
meeting A's 38 minus 1. It passes (c) trivially — no win is claimed from the
strict-label gap (66 vs 63) since the aliased result already fails (a). All
three of (a)/(b)/(c) must hold for a win; (a) alone sinks it. **Null.**
Nothing ships; the 0.6B stays the default reranker.

## Reranker vs gate

**A′ = A.** Re-sweeping the 0.6B's gate at 0.5805671225057373 instead of the
shipped 0.65 changes every per-arm count by exactly zero (table above), and
it is not just the same *count* of 33 gated questions — it is the same
*set*:

```
$ python3 -c "
import json
def load(p): return json.load(open(p))
def index(d): return {r['qid']: r for r in d['results']}
A  = index(load('data/eval/results/p7-A-answers.json'))
Ap = index(load('<orchestrator p8-A0g-answers.json>'))
gA  = {q for q, r in A.items()  if r['gated']}
gAp = {q for q, r in Ap.items() if r['gated']}
print(len(gA), len(gAp), gA == gAp)
"
33 33 True
```

Both thresholds (0.65 shipped, 0.5806 re-swept) gate the identical 33
questions on this pool, so the re-sweep in §5 is not silently doing work
that then gets attributed to the reranker swap in B — the comparison B vs
A′ isolates the reranker's own effect cleanly.

**Gate decomposition.** B's gate refuses 11 answerable questions (vs A's 6);
of B's 11, 6 had evidence in context (vs A's 2 of 6); of those, only 1 would
have been correct if let through ungated (vs A's 1 of 2):

```
$ python3 -c "
import json
def load(p): return json.load(open(p))
def index(d): return {r['qid']: r for r in d['results']}
A    = index(load('data/eval/results/p7-A-answers.json'))
B    = index(load('<orchestrator p8-B4g-answers.json>'))
A0u  = index(load('<orchestrator p8-A0u-answers.json>'))
B4u  = index(load('<orchestrator p8-B4u-answers.json>'))

def stats(gated_idx, ungated_idx, name):
    refused = [q for q, r in gated_idx.items() if r['kind'] == 'answerable' and r['gated']]
    with_ev = [q for q in refused if gated_idx[q].get('evidence_retrieved')]
    would_be = [q for q in refused if ungated_idx[q].get('correct')]
    print(f'{name}: refuses {len(refused)} answerable, {len(with_ev)} with evidence, {len(would_be)} would be correct ungated')

stats(A, A0u, 'A')
stats(B, B4u, 'B')
"
A: refuses 6 answerable, 2 with evidence, 1 would be correct ungated
B: refuses 11 answerable, 6 with evidence, 1 would be correct ungated
```

**Discordant losses under B**, checked against context and evidence: of the
8 losses, 5 (a27 a30 b06 c12 b03.n) had the evidence in context under
*both* rerankers and were answered wrong under B — the other extracts and
their order changed, and the 4B-context model misread the same evidence it
had before. Two (b11, a04.z) were gated by B's own re-swept threshold;
replaying them ungated (`p8-B4u`) shows only a04.z would have been correct,
b11 would still have been wrong. The last (p10) lost its evidence entirely
under B (not in the top-5 cache) and was refused by the model itself for
lack of evidence, not by the gate.

**Correct given evidence among non-gated questions**: A 76/88, B 78/98
(re-derived: answerable, not gated, evidence in context, correct — 76/88
and 78/98 respectively, matching the per-arm table's 76/90 and 78/104 once
the gated-with-evidence questions are excluded from each denominator).

**SECONDARY, not pre-registered as a criterion** — the unsupported-answer
rate (answerable-with-no-evidence-answered plus unanswerable-answered) falls
from 24/166 (23+1) = 14.5% under A to 16/166 (14+2) = 9.6% under B:

```
$ python3 -c "
import json
def load(p): return json.load(open(p))
def index(d): return {r['qid']: r for r in d['results']}
A = index(load('data/eval/results/p7-A-answers.json'))
B = index(load('<orchestrator p8-B4g-answers.json>'))

def unsupported(idx, name):
    ans_no_ev = [r for r in idx.values() if r['kind'] == 'answerable' and not r.get('evidence_retrieved') and not r['abstained']]
    una_answered = [r for r in idx.values() if r['kind'] != 'answerable' and not r['abstained']]
    total = len(ans_no_ev) + len(una_answered)
    print(f'{name}: {len(ans_no_ev)} + {len(una_answered)} = {total}/166 = {total/166:.1%}')

unsupported(A, 'A')
unsupported(B, 'B')
"
A: 23 + 1 = 24/166 = 14.5%
B: 14 + 2 = 16/166 = 9.6%
```

This is mostly mechanical: B's stronger reranker leaves only 23 answerable
questions without evidence instead of A's 37, so there are fewer
opportunities for an unsupported answer in the first place. This was never
part of §6's win criterion and is not claimed as one here — it is reported
as a candidate primary metric worth pre-registering in a future phase,
nothing more.

## Cost: latency and memory

**Retrieval evidence** (rank of first evidence chunk after reranking,
cumulative): for the 0.6B, within its own top-20 pool
(`data/eval/results/p7-k20-retrieved.json`, untracked, same file as §3's
gate2-premise numbers) rank ≤1/3/5/20 is 56/80/90/113. For the 4B reranking
the full 50 retrieval candidates (`p8-k20`, also untracked — neither cache
is present in this worktree, `find data/eval/results -iname '*k20*'` returns
nothing here), the orchestrator reports 60/87/104/115. The rank ≤5 figures
(90 vs 104) are the ones that reach the model at k=5 and match this
document's per-arm table exactly; rank ≤20 rising 113→115 shows the 4B over
the full 50-candidate pool surfaces two evidence chunks the 0.6B's own top
20 never contained at all, beyond what §3's within-pool premise check could
show.

**Latency and VRAM** (reported by the orchestrator; not independently
re-derived here — no timing instrumentation is stored in the answer JSONs,
and, as in §3's smoke scores, no reranker weights are present in this
worktree to re-run retrieval): retrieval over 166 questions took 471 s for
the 0.6B (2.8 s/q, phase 7's k=20 run) versus 1466 s for the 4B (8.8 s/q) —
3.1× slower. The 4B reranker alone uses roughly 4.2 GB VRAM at this
project's query-sized settings (`-c 1024 -b 768 --parallel 1`, nvidia-smi
4204 MiB with the lean embedder also resident), versus the current 0.6B
profile that holds embedder + reranker + generator together in roughly
5.6 GB (§6). The 4B cannot share the card with the generator at that
budget; a serving path built on it would need lazy per-model loading rather
than all three models resident at once.

## What this changes

Null; nothing ships. Across phases 6-8 the finding side of the pipeline can
still be moved — a stronger reranker lifts evidence-in-context 90→104 at
the same k=5 shown to the model — but end-to-end aliased correctness does
not follow it (81→82/127, p=1.0), because the 4B generator's reading is
brittle to context composition: of the 17 discordant answerable questions,
about 8 answers flip either way from reordering the other four extracts
alone, not from any change in whether the gold evidence is present (§
"Reranker vs gate", the 5 both-had-evidence losses). The binding constraint
is the reader, not the retriever.

The obvious next test is a stronger generator, not a stronger reranker or a
wider k. This document does not design that test; it names it as the open
question phase 8 leaves for whichever phase comes next.
