# Phase 7 — reading vs finding, pre-registered

This document is written before phase 7's own arms produce a single number —
a parallel task is building the phase right now, and no arm has been
measured yet. What follows fixes the question, the baselines every arm has
to beat (re-derived here, commands shown), the four arms and exactly what
differs between them, the scoring protocol, and the pre-registered win
criterion, in that order, so that none of it can be adjusted after seeing an
arm's numbers. The four results sections at the end are placeholders.

## 1. Question

Under the corrected (aliased) answer label (`docs/phase6-results.md`,
"Correction (2026-09-26)"), the control run (`fusion-ctl`: `main.db`,
dense+rerank, k=5, gate 0.65, grammar on) has 28 answered-and-wrong
questions out of 109 answered; 18 of those had no evidence in the 5 extracts
the model actually read (re-derived below, §2, matching phase 6's own count
of 18/28 exactly).

The gate cannot catch those 18: its score (`top_score`, the reranker's
top-1 score, the same number `sweep_gate.py` thresholds) is if anything
*higher* on them than on correct answers, not lower:

```
$ python3 -c "
import json, sys
sys.path.insert(0, 'src')
from smm.gold import is_correct, load_aliases
aliases = load_aliases('data/eval/gold_aliases.json')
questions = {json.loads(l)['qid']: json.loads(l) for l in open('data/eval/questions.jsonl') if l.strip()}
d = json.load(open('data/eval/results/fusion-ctl-answers.json'))
ans = [x for x in d['results'] if not x['gated'] and not x['abstained']]
def aliased_correct(rec):
    q = questions[rec['qid']]
    return is_correct(rec['answer'], q.get('answer_contains', []), aliases.get(rec['qid'], {}))
good = [x for x in ans if x['kind'] == 'answerable' and aliased_correct(x)]
bad = [x for x in ans if x not in good]
noev = [x for x in bad if x['kind'] == 'answerable' and x.get('evidence_retrieved') is False]
gs = [x['top_score'] for x in good]; bs = [x['top_score'] for x in noev]
def auc(a, b):
    n = wins = 0
    for g in a:
        for x in b:
            n += 1
            wins += 1 if g > x else (0.5 if g == x else 0)
    return wins / n
print('n_good', len(gs), 'n_bad(no-evidence-wrong)', len(bs))
print('AUC(top_score), P(good > bad):', round(auc(gs, bs), 4))
print('mean top_score  good:', round(sum(gs)/len(gs), 4), ' bad:', round(sum(bs)/len(bs), 4))
"
n_good 81 n_bad(no-evidence-wrong) 18
AUC(top_score), P(good > bad): 0.428
mean top_score  good: 0.9709  bad: 0.9735
```

AUC 0.428 (chance is 0.5, and the direction is *against* the gate: the
no-evidence-wrong answers score higher, not lower, on the same signal the
gate thresholds on 0.65 against). A gate built on this score cannot separate
these 18 from the 81 good answers no matter where its threshold sits.

So the lever is not the gate; it is putting the evidence in front of the
model in the first place. Does reading more extracts (a larger k), or
keeping k fixed but forcing the model to see chunks from more distinct
documents (a per-document cap), raise correct answers — or does a 4B model
read a longer, noisier context worse than a shorter, cleaner one?

## 2. Baselines, measured

**Evidence in top-k, answerable questions**, from
`data/eval/results/fusion-control-rewrites0.json` (`per_question`,
`first_hit_rank`; confirmed same run as the control answer set — `mode:
dense, rerank: True, candidates: 50, rewrites: 0` matches
`fusion-ctl-answers.json`'s config, and `recall@{1,3,5,10,20}` in the file
reproduces the k=5/k=20 counts below exactly):

```
$ python3 -c "
import json
d = json.load(open('data/eval/results/fusion-control-rewrites0.json'))
ans = [x for x in d['per_question'] if x['kind'] == 'answerable']
n = len(ans)
for k in (5, 6, 8, 10, 15, 20):
    c = sum(1 for x in ans if x['first_hit_rank'] is not None and x['first_hit_rank'] <= k)
    print(f'k={k:>2}  {c}/{n}  {100*c/n:.1f}%')
"
k= 5  90/127  70.9%
k= 6  94/127  74.0%
k= 8  101/127  79.5%
k=10  102/127  80.3%
k=15  109/127  85.8%
k=20  113/127  89.0%
```

(k=5 here, 90/127 = 70.9%, is the same number phase 6 §1 quoted as "about
71%"; k=20, 113/127 = 89.0%, matches the file's own `recall@20: 0.8898`.)

**Of the 18 no-evidence-wrong answers, where does the first evidence chunk
actually sit** (same file, same `first_hit_rank`, joined to the 18 qids
derived in §1):

```
$ python3 -c "
import json, sys
sys.path.insert(0, 'src')
from smm.gold import is_correct, load_aliases
aliases = load_aliases('data/eval/gold_aliases.json')
questions = {json.loads(l)['qid']: json.loads(l) for l in open('data/eval/questions.jsonl') if l.strip()}
d = json.load(open('data/eval/results/fusion-ctl-answers.json'))
ans = [x for x in d['results'] if not x['gated'] and not x['abstained']]
def aliased_correct(rec):
    q = questions[rec['qid']]
    return is_correct(rec['answer'], q.get('answer_contains', []), aliases.get(rec['qid'], {}))
good = [x for x in ans if x['kind'] == 'answerable' and aliased_correct(x)]
bad = [x for x in ans if x not in good]
noev_qids = {x['qid'] for x in bad if x['kind'] == 'answerable' and x.get('evidence_retrieved') is False}
rw = json.load(open('data/eval/results/fusion-control-rewrites0.json'))
pq = {x['qid']: x['first_hit_rank'] for x in rw['per_question']}
buckets = {'6-8': 0, '9-20': 0, 'none': 0}
for qid in noev_qids:
    fhr = pq[qid]
    if fhr is None: buckets['none'] += 1
    elif 6 <= fhr <= 8: buckets['6-8'] += 1
    elif 9 <= fhr <= 20: buckets['9-20'] += 1
print(len(noev_qids), buckets)
"
18 {'6-8': 7, '9-20': 7, 'none': 4}
```

(7 + 7 + 4 = 18: `first_hit_rank` 6..8 → a04, a11, a18, a40, b13, c04, b12.t;
9..20 → a02, a42, p02, a11.t, a15.z, a25.n, a37.t; none (no hit in the top
20 at all) → a35, c03, p06, a35.z.)

So 14 of the 18 no-evidence-wrong questions *do* have their first relevant
chunk somewhere in the top 20 — 7 would newly appear by k=8, another 7 only
by k=20 — and only 4 have no hit at all in the top 20 (a ceiling no k in
this phase's arms can raise).

**Document concentration in the current top-5**, from
`fusion-ctl-retrieved.json` — untracked, not part of this worktree, read
from the orchestrator's scratchpad at
`/tmp/claude-1000/-home-stupakzm-projects-slm-memory-management/c8ea9a11-ce22-4155-8f41-1fa70f61f461/scratchpad/verify-inputs/fusion-ctl-retrieved.json`
(166 qids, each a list of exactly 5 retrieved chunks — the control run's
actual k=5 context):

```
$ python3 -c "
import json, collections
d = json.load(open('/tmp/claude-1000/-home-stupakzm-projects-slm-memory-management/c8ea9a11-ce22-4155-8f41-1fa70f61f461/scratchpad/verify-inputs/fusion-ctl-retrieved.json'))
cnt = sum(1 for chunks in d.values()
          if max(collections.Counter(c['doc_id'] for c in chunks).values()) >= 4)
print(cnt, '/', len(d))
"
41 / 166
```

41 of 166 top-5 contexts (25%) already hold 4 or 5 chunks from a single
document — this is what a per-document cap would thin out, freeing slots
for other documents at the same k.

**Control, aliased** (re-derived with `src/smm/gold.py`'s `is_correct`
against `data/eval/gold_aliases.json` and `data/eval/questions.jsonl`,
joined to `fusion-ctl-answers.json`'s `results` — same method as
`docs/phase6-results.md`'s "Correction" section, reproduced here rather than
copied):

```
$ python3 -c "
import json, sys
sys.path.insert(0, 'src')
from smm.gold import is_correct, load_aliases
aliases = load_aliases('data/eval/gold_aliases.json')
questions = {json.loads(l)['qid']: json.loads(l) for l in open('data/eval/questions.jsonl') if l.strip()}
d = json.load(open('data/eval/results/fusion-ctl-answers.json'))
r = d['results']
def aliased_correct(rec):
    q = questions[rec['qid']]
    return is_correct(rec['answer'], q.get('answer_contains', []), aliases.get(rec['qid'], {}))
allans = [x for x in r if x['kind'] == 'answerable']
corr = [x for x in allans if aliased_correct(x)]
with_ev = [x for x in allans if x.get('evidence_retrieved') is True]
corr_ev = [x for x in with_ev if aliased_correct(x)]
unab = [x for x in r if x['kind'] == 'unanswerable' and x['abstained']]
unatot = [x for x in r if x['kind'] == 'unanswerable']
print(f'correct/answerable        {len(corr)}/{len(allans)}  ({100*len(corr)/len(allans):.1f}%)')
print(f'correct given evidence    {len(corr_ev)}/{len(with_ev)}  ({100*len(corr_ev)/len(with_ev):.1f}%)')
print(f'unanswerable abstained    {len(unab)}/{len(unatot)}')
"
correct/answerable        81/127  (63.8%)
correct given evidence    76/90  (84.4%)
unanswerable abstained    38/39
```

These three numbers (81/127, 76/90, 38/39) are what every arm in §3 is
compared against — arm A must reproduce them exactly, since A is defined as
the control's own retrieval and reading policy under the shared k=20 pool
(see §3).

Strict-label control, for continuity with figures reported before the
alias correction: 63/127 correct/answerable (`fusion-ctl-answers.json`'s own
`answer_accuracy: 0.496`, i.e. 63/127) — same 39-question unanswerable set,
same 38/39 abstention (abstention does not depend on the answer-token label).

## 3. Arms

All four arms share **one** retrieval run at k=20 — the same reranked list,
the same gate score (`top_score`, the reranker's top-1 score, which by
construction is identical across all four arms since none of them touch
which chunk ranks first). Arms differ only in what subset of that k=20 list
the generator is actually shown:

| arm | k shown to model | per-document cap |
|---|---|---|
| A (control) | 5 | none — must reproduce §2's 81/127, 76/90, 38/39 |
| B | 8 | none |
| C | 5 | 2 |
| D | 8 | 2 |

**The cap rule**: walk the k=20 pool in its existing rank order, keep a
chunk if its document has fewer than 2 chunks kept so far, stop once the
per-arm k is filled; if the cap leaves fewer than k chunks after one pass
over all 20 (too few distinct documents in the pool), top up with the
skipped chunks, in their original rank order, until the arm's k is reached.
The gate itself is unchanged in every arm — it reads the top-1 chunk of the
k=20 pool, which no arm's k or cap can alter, so gating and abstention
behavior differences between arms, if any, come only from what the
generator does with a different reading list, never from a different gate
decision.

## 4. Protocol

- **Primary metric**: aliased correct/answerable, compared per question
  against arm A (same qid, same gold tokens, same aliasing).
- **Significance**: a two-sided exact sign test over discordant questions
  only — questions where arm A and the other arm disagree (right in one,
  wrong in the other); concordant questions (both right or both wrong)
  carry no information for this test and are excluded from n.
- **Also reported, per arm**: correct given evidence (did the model make
  better use of the extra context it was actually given, or worse?),
  evidence-in-context (does a larger k or the cap change how often the gold
  evidence is present at all, distinct from whether the model used it),
  unanswerable abstention (does reading more change over-answering on
  questions with no correct answer), answered-with-no-evidence — denominator
  is answerable questions with no evidence in context, counted the way
  `eval_answers.py` counts it: over **all** answerable questions regardless
  of whether they were answered or abstained (`scripts/eval_answers.py:212-213`,
  `no_ev = [r for r in ans if not r["evidence_retrieved"]]`), numerator is
  the subset of those that were answered anyway (`scripts/eval_answers.py:217`,
  `spoke_blind`) — not `scripts/eval_verifier.py`'s printed line, which
  phase 6 flagged as using the wrong (127, not the ~37) denominator
  (`docs/phase6-results.md`, "Tooling note", `scripts/eval_verifier.py:253`).
  And strict-label correct/answerable, reported for continuity only, never
  as the basis for a win (see §5(c)).

## 5. Decision rule, pre-registered

An arm counts as a win only if **all** of the following hold:

(a) it beats arm A on aliased correct/answerable, two-sided sign test
p < 0.05, computed over discordant answerable questions only (§4);
(b) unanswerable correctly abstained is no worse than A's count minus 1
(A = 38/39, so an arm needs ≥ 37/39 to pass this bar);
(c) no win is claimed from strict-label numbers alone — a strict-label
improvement with a flat or worse aliased result does not count.

If more than one arm wins, prefer the smallest change over a larger one —
**C over B over D** — unless the larger change is *itself* significantly
better than the smaller one it would otherwise be passed over for (i.e. B
must clear its own sign test against C, not just against A, before it is
preferred over C; likewise D against whichever of B/C it would replace). A
win means a separate task to wire the winning arm's k/cap into
`scripts/ask.py`; short of a win, nothing ships and the result is reported
null, the same convention `docs/phase6-results.md` §5 used.

**Power.** With about 127 answerable questions, the sign test in (a) runs
only over the subset that actually flips between arm A and the other arm —
not all 127, and not bounded to any fixed number ahead of time, since a
larger k or the cap could in principle change either the 81 currently
correct or the 46 currently not-correct (127 − 81; of those 46, 27 are
answered but wrong — one less than §1's 28, which also counts the one
unanswerable question answered anyway — and 19 are gated or abstained) in
either direction. So the test's power depends
entirely on how many questions actually flip, not on the size of the eval
set (same caveat as phase 6 §5). Extending phase 6's exact two-sided
sign-test table to a few illustrative discordant totals, the most balanced
split that still clears p < 0.05 at each total:

| discordant n | most balanced split still significant | p |
|---|---|---|
| 10 | 9 – 1 | 0.0215 |
| 15 | 12 – 3 | 0.0352 |
| 20 | 15 – 5 | 0.0414 |

So at 10 discordant questions, only a near-unanimous split (9 of 10 going
one arm's way) resolves; by 20 discordant questions the design can resolve
a 3-to-1 split (15 vs 5). Anything closer to even (e.g. 6–4 at n=10, 8–7 at
n=15) will not reach significance regardless of which arm produced it — a
property of how many questions flip between two arms, not of whether one
arm is genuinely better.

## Per-arm results

Re-derived from the four arm answer files (untracked eval artifacts, produced
by `scripts/eval_answers.py --stage generate --cache p7-k20 --read-k K
--cap-per-doc C --gate 0.65 --grammar` off one shared `--stage retrieve -k
20` run; read here from the orchestrator's scratchpad at
`/tmp/claude-1000/-home-stupakzm-projects-slm-memory-management/c8ea9a11-ce22-4155-8f41-1fa70f61f461/scratchpad/p7-out/p7-{A,B,C,D}-answers.json`):

```
$ python3 -c "
import json, math
BASE = '/tmp/claude-1000/-home-stupakzm-projects-slm-memory-management/c8ea9a11-ce22-4155-8f41-1fa70f61f461/scratchpad/p7-out/p7-{}-answers.json'
def load(arm): return json.load(open(BASE.format(arm)))
arms = {a: load(a) for a in 'ABCD'}

def sign_test(up, down):
    n = up + down
    k = min(up, down)
    p = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n) * 2
    return min(p, 1.0)

def stats(d):
    ans = [r for r in d['results'] if r['kind'] == 'answerable']
    una = [r for r in d['results'] if r['kind'] == 'unanswerable']
    with_ev = [r for r in ans if r['evidence_retrieved']]
    no_ev = [r for r in ans if not r['evidence_retrieved']]
    return dict(
        correct=sum(1 for r in ans if r['correct']),
        strict=sum(1 for r in ans if r['correct_strict']),
        n_ev=len(with_ev), correct_ev=sum(1 for r in with_ev if r['correct']),
        abstain=sum(1 for r in una if r['abstained']), n_una=len(una),
        blind=sum(1 for r in no_ev if not r['abstained']), n_noev=len(no_ev),
        ans={r['qid']: r for r in ans})

S = {a: stats(arms[a]) for a in 'ABCD'}
for a in 'ABCD':
    s = S[a]
    line = (f\"{a}: {s['correct']} ({s['strict']}) | {s['n_ev']} | \"
            f\"{s['correct_ev']}/{s['n_ev']} {100*s['correct_ev']/s['n_ev']:.1f}% | \"
            f\"{s['abstain']}/{s['n_una']} | {s['blind']}/{s['n_noev']}\")
    if a == 'A':
        print(line, '| —')
        continue
    up = down = 0
    for qid, ra in S['A']['ans'].items():
        rb = s['ans'][qid]
        if ra['correct'] and not rb['correct']: down += 1
        elif rb['correct'] and not ra['correct']: up += 1
    p = sign_test(up, down)
    print(line, f'| +{up}/-{down}, p={p:.3f}')
"
A: 81 (63) | 90 | 76/90 84.4% | 38/39 | 23/37 | —
B: 85 (67) | 101 | 81/101 80.2% | 36/39 | 16/26 | +10/-6, p=0.454
C: 81 (60) | 87 | 73/87 83.9% | 37/39 | 27/40 | +3/-3, p=1.000
D: 83 (63) | 97 | 77/97 79.4% | 35/39 | 20/30 | +8/-6, p=0.791
```

Arm A reproduces `fusion-ctl-answers.json` exactly — same script confirms
166/166 identical answers (same `answer`, `abstained`, `gated`) and the
k=20 cache's top-5 chunks match the control's actual k=5 retrieval on
166/166 questions, so A is a valid stand-in for the control under the
shared pool.

| arm | aliased correct/127 (strict) | evidence in context | correct given evidence | unanswerable abstained/39 | answered blind / no-evidence | discordant vs A | (a) sign test p<0.05 | (b) abstain ≥37/39 |
|---|---|---|---|---|---|---|---|---|
| A k=5 (reference) | 81 (63) | 90/127 | 76/90 (84.4%) | 38/39 | 23/37 | — | — | pass (38) |
| B k=8 | 85 (67) | 101/127 | 81/101 (80.2%) | 36/39 | 16/26 | +10/−6, p=0.454 | fail | fail (36) |
| C k=5, cap=2 | 81 (60) | 87/127 | 73/87 (83.9%) | 37/39 | 27/40 | +3/−3, p=1.000 | fail | pass (37) |
| D k=8, cap=2 | 83 (63) | 97/127 | 77/97 (79.4%) | 35/39 | 20/30 | +8/−6, p=0.791 | fail | fail (35) |

**Verdict, per §5**: every arm is null. None of B, C or D clears (a) — the
sign test needs 13–3 or better at 16 discordant (B/D's scale) to reach
p < 0.05, and B/D land at 10–6/8–6 while C has only 3–3 discordant, nowhere
close. B and D also fail (b) outright (36/39 and 35/39 against the ≥37/39
bar), so they lose on two independent grounds, not one. C alone clears (b)
(37/39) but that is moot once (a) fails. (c) does not come into play for
any arm — no arm shows a strict-label-only improvement to disqualify.
Nothing ships.

## Reading vs finding

Correct given evidence (from the table above): A 76/90 (84.4%), B 81/101
(80.2%), C 73/87 (83.9%), D 77/97 (79.4%). Finding (evidence in context)
rises with k; reading (correct given the model was actually shown the
evidence) falls with k. The two move in opposite directions and roughly
cancel in the aliased-correct total.

**B's 16 discordant questions against A**, split by whether the gold
evidence's presence in context changed between k=5 and k=8:

```
$ python3 -c "
import json
def load(arm):
    return json.load(open(f'/tmp/claude-1000/-home-stupakzm-projects-slm-memory-management/c8ea9a11-ce22-4155-8f41-1fa70f61f461/scratchpad/p7-out/p7-{arm}-answers.json'))
A = {r['qid']: r for r in load('A')['results'] if r['kind'] == 'answerable'}
B = {r['qid']: r for r in load('B')['results'] if r['kind'] == 'answerable'}
up = [q for q in A if not A[q]['correct'] and B[q]['correct']]
down = [q for q in A if A[q]['correct'] and not B[q]['correct']]
def tag(q):
    ea, eb = A[q]['evidence_retrieved'], B[q]['evidence_retrieved']
    return 'newly-in-context' if (not ea and eb) else 'same evidence status'
print('gains (A wrong -> B right):', len(up))
for q in up: print(' ', q, tag(q))
print('losses (A right -> B wrong):', len(down))
for q in down: print(' ', q, 'no evidence in A either' if not A[q]['evidence_retrieved'] else 'evidence present in both A and B')
"
gains (A wrong -> B right): 10
  a04 newly-in-context
  a11 newly-in-context
  a18 newly-in-context
  a29 same evidence status
  a40 newly-in-context
  b07 same evidence status
  b09 same evidence status
  a11.t same evidence status
  a21.n newly-in-context
  b12.t newly-in-context
losses (A right -> B wrong): 6
  a19 no evidence in A either
  a30 evidence present in both A and B
  b06 evidence present in both A and B
  p04 evidence present in both A and B
  a09.n evidence present in both A and B
  a25.t evidence present in both A and B
```

So of B's 10 gains, 6 are a genuine finding win — the gold evidence was
absent from A's 5-chunk context and present once k widened to 8 (a04, a11,
a18, a40, a21.n, b12.t); the other 4 (a29, b07, b09, a11.t) had the same
evidence status in both arms, so the flip is a reading effect, not a
finding one. Of the 6 losses, 5 (a30, b06, p04, a09.n, a25.t) already had
the gold evidence in A's 5-chunk context and kept it in B's 8-chunk
context (`evidence_retrieved` is `True` in both) — B still got these wrong,
so the extra 3 extracts did not remove the evidence, they degraded the
model's use of it. The sixth, a19, had no evidence in either arm and was
right in A anyway (a guess or partial match A got away with, that the
noisier B context did not). B also newly answers 2 previously-abstained
unanswerable questions instead of abstaining (u18, p12) — both flip from
`abstained: true` in A to an answer in B.

**D's discordant questions**, same split (D shares B's k=8 but adds
cap=2):

```
$ python3 -c "
import json
def load(arm):
    return json.load(open(f'/tmp/claude-1000/-home-stupakzm-projects-slm-memory-management/c8ea9a11-ce22-4155-8f41-1fa70f61f461/scratchpad/p7-out/p7-{arm}-answers.json'))
A = {r['qid']: r for r in load('A')['results'] if r['kind'] == 'answerable'}
D = {r['qid']: r for r in load('D')['results'] if r['kind'] == 'answerable'}
up = [q for q in A if not A[q]['correct'] and D[q]['correct']]
down = [q for q in A if A[q]['correct'] and not D[q]['correct']]
print('gains:', len(up))
for q in up:
    ea, ed = A[q]['evidence_retrieved'], D[q]['evidence_retrieved']
    print(' ', q, 'newly-in-context' if (not ea and ed) else 'same evidence status')
print('losses:', len(down))
for q in down:
    ea, ed = A[q]['evidence_retrieved'], D[q]['evidence_retrieved']
    print(' ', q, 'evidence lost (cap)' if (ea and not ed) else ('evidence present in both' if ea and ed else 'no evidence either'))
"
gains: 8
  a11 newly-in-context
  a18 newly-in-context
  a37 same evidence status
  a40 newly-in-context
  b07 same evidence status
  p08 same evidence status
  a11.t same evidence status
  a21.n newly-in-context
losses: 6
  a30 evidence present in both
  a43 evidence present in both
  b06 evidence present in both
  p04 evidence present in both
  a01.t evidence lost (cap)
  a25.t evidence present in both
```

D's pattern matches B's for the k-driven gains (4 newly-in-context, 4 same
evidence status), but one of D's losses (a01.t) is a genuine cap-driven
evidence loss on top of the reading-degradation pattern the other 5 share
— cap=2 removes evidence k=8 alone would have kept (§"Ablation" below has
the same qid on cap's own casualty list, a01.t, confirming it is the cap,
not the k, that cost this one). D newly answers 3 previously-abstained
unanswerable questions (u13, u18, p12), one more than B (u13 only flips in
D).

**Net reading-vs-finding effect**: widening k finds strictly more evidence
(90→101 for B, 90→97 for D) but the model reads that longer, noisier
context worse (84.4%→80.2% correct-given-evidence for B, 84.4%→79.4% for
D) — the two effects roughly cancel in the aliased-correct total (81→85
for B, +4; 81→83 for D, +2), and neither reaches significance against A.

## Ablation: k and cap separately

**k alone (B, k=8, no cap)**: evidence in context rises 90→101/127, but
correct-given-evidence falls 84.4%→80.2%; aliased correct rises 81→85
(+10/−6, p=0.454, not significant); unanswerable abstention falls 38→36/39
(fails the ≥37/39 bar). Widening k finds more, reads worse, nets a small
and non-significant gain, and gives back coverage on unanswerable
questions (§"Reading vs finding").

**Cap alone (C, k=5, cap=2)**: evidence in context falls 90→87/127 (the
cap actively removes evidence the uncapped k=5 context had), yet
correct-given-evidence is nearly flat (84.4%→83.9%) and aliased correct is
unchanged at 81 (+3/−3, p=1.000, only 3 discordant pairs, no power).
Unanswerable abstention rises 38→37/39, the only arm that clears the (b)
bar — but with (a) failing outright there is nothing to combine it with.

**Both (D, k=8, cap=2)**: combines k's finding gain with the cap's own
losses. Evidence rises 90→97/127 (less than B's 90→101, because the cap
removes some of what the wider k would otherwise have added),
correct-given-evidence falls furthest of the three arms (84.4%→79.4%),
aliased correct rises 81→83 (+8/−6, p=0.791), and abstention falls to
35/39, the worst of the four arms. D inherits both k's reading penalty and
the cap's evidence removal (a01.t below is lost to the cap in D just as it
is in C); it does not compound their gains.

**The cap mechanism**: C loses evidence, relative to A, on exactly 6
questions, and in every one of them the crowding document A already
concentrated in its top 5 is the one the cap starves:

```
$ python3 -c "
import json, collections
def load(arm):
    return json.load(open(f'/tmp/claude-1000/-home-stupakzm-projects-slm-memory-management/c8ea9a11-ce22-4155-8f41-1fa70f61f461/scratchpad/p7-out/p7-{arm}-answers.json'))
A = {r['qid']: r for r in load('A')['results'] if r['kind'] == 'answerable'}
C = {r['qid']: r for r in load('C')['results'] if r['kind'] == 'answerable'}
lost = [q for q in A if A[q]['evidence_retrieved'] and not C[q]['evidence_retrieved']]
print('C loses evidence on', len(lost), 'questions:', lost)
for q in lost:
    docs = A[q]['retrieved_docs']
    doc, n = collections.Counter(docs).most_common(1)[0]
    print(' ', q, 'A top-5 docs:', docs, '-> crowding doc', doc, f'({n}/5)')
"
C loses evidence on 6 questions: ['a25', 'b12', 'p01', 'a01.t', 'a37.z', 'b12.z']
  a25 A top-5 docs: ['sed.1', 'pager.1', 'sed.1', 'busybox.1', 'sed.1'] -> crowding doc sed.1 (3/5)
  b12 A top-5 docs: ['systemd.service.5', 'systemd.service.5', 'systemd.service.5', 'systemd.service.5', 'systemd.1'] -> crowding doc systemd.service.5 (4/5)
  p01 A top-5 docs: ['tar.1', 'dpkg-source.1', 'tar.1', 'tar.1', 'tar.1'] -> crowding doc tar.1 (4/5)
  a01.t A top-5 docs: ['tar.1', 'dpkg-source.1', 'tar.1', 'tar.1', 'tar.1'] -> crowding doc tar.1 (4/5)
  a37.z A top-5 docs: ['mount.8', 'mount.8', 'mount.8', 'mount.8', 'mount.8'] -> crowding doc mount.8 (5/5)
  b12.z A top-5 docs: ['systemd.service.5', 'systemd.service.5', 'systemd.service.5', 'systemd.service.5', 'systemd.service.5'] -> crowding doc systemd.service.5 (5/5)
```

All 6 crowding documents hold at least 3 of A's 5 retrieved chunks (3, 4,
4, 4, 5, 5). Same-document crowding is mostly the *right* document, not
noise: the cap's premise — that forcing distinct documents into view helps
— is backwards here, because the document dominating the top 5 is usually
dominating it because it is the relevant one. Capping removes evidence
more often than it frees a useful slot for a different, correct document.

## What this changes

Every arm is null (§"Per-arm results"). Nothing ships — no separate task to
wire k or a cap into `scripts/ask.py`.

The corrected-label error budget this phase set out to move is the
control's 28 answered-and-wrong questions (`docs/phase6-results.md`; §1
above): 18 no-evidence, 9 wrong-with-evidence, 1 unanswerable-answered
(re-derived in §1, matching that count exactly). This phase targeted the
18 no-evidence questions specifically — the ones the gate cannot catch
(§1's AUC 0.428). Depth does put more of that evidence in front of the
model: §2 already showed 14 of the 18 have their first relevant chunk
somewhere in the top 20 (7 by k=8, another 7 only by k=20), and B (k=8)
does raise evidence-in-context on the full answerable set from 90 to
101/127. But the 4B model reads a longer, noisier context worse, not
better: correct-given-evidence drops from 84.4% to 80.2% under B and to
79.4% under D (§"Reading vs finding"), enough to erase almost all of the
gain finding produced. The budget is not movable by retrieval depth alone,
because the reading side degrades in step with the finding side.

The binding constraint this phase locates is reading capacity at a longer
context, not finding capacity — the two do not fail for the same reason,
and only one of them responds to a bigger k. 4 of the 18 no-evidence
misses (a35, c03, p06, a35.z; §2) are not in the top 20 at all, so no k
this phase tested (or could test, short of reworking retrieval itself)
reaches them — a ceiling on what depth alone can do even before the
reading regression is counted.

Open question, not designed here: how to give the model more evidence
without more distraction.
