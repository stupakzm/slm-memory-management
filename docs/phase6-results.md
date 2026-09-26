# Phase 6 — an independent verifier, pre-registered

> Gate: a mechanism ships only if, on the control run's held-out folds,
> removed_bad > removed_good with a two-sided sign test p < 0.05; the fused
> run's held-out result agrees in direction; and correct answers lost on the
> control run are ≤ 3. Otherwise it is reported null and nothing ships.

This document is written before phase 6's own verifier produces a single
number — a parallel task is building it now. What follows fixes the question,
the two measured baselines the verifier has to beat, the three mechanisms and
exactly how each is scored, the no-peeking protocol, and the win criterion,
in that order, so that none of it can be adjusted after the fact. The four
results sections at the end are placeholders.

## 1. Question

Retrieval now puts the evidence in front of the model at k=5 about 71% of the
time, but nothing checks whether an answer follows from the extract it
cites. Can a second, independent check that reads each cited claim against
its cited extract turn wrong answers into refusals without eating correct
ones?

## 2. Baselines, measured

Both runs are `main.db`, dense retrieval + rerank, k=5, gate 0.65, grammar
on, 166 questions (127 answerable, 39 unanswerable) — confirmed from each
file's own `config` and `n`:

```
$ python3 -c "import json; d=json.load(open('data/eval/results/fusion-ctl-answers.json')); print(d['name'], d['n'], d['config'])"
fusion-ctl 166 {'mode': 'dense', 'rerank': True, 'gate': 0.65, 'grammar': True, 'candidates': 50, 'expand': 0, 'db': '.../data/index/main.db', 'domain': None, 'rewrites': 0}

$ python3 -c "import json; d=json.load(open('data/eval/results/fusion-rw1-answers.json')); print(d['name'], d['n'], d['config'])"
fusion-rw1 166 {'mode': 'dense', 'rerank': True, 'gate': 0.65, 'grammar': True, 'candidates': 50, 'expand': 0, 'db': '.../data/index/main.db', 'domain': None, 'rewrites': 1}
```

`fusion-ctl-answers.json` is the control (`--rewrites 0`); `fusion-rw1-answers.json`
is the fused run (`--rewrites 1`). Both carry a per-question `results` list
with `qid`, `kind`, `correct`, `evidence_retrieved`, `gated`, `abstained` and
`cite_supported`. Computed with the one-liner below (an "answered" question
is `not gated and not abstained`; "good" is `kind == answerable and correct`;
"bad" is everything else answered; the bad split into no-evidence /
wrong-with-evidence / unanswerable-answered is exhaustive and mutually
exclusive by construction):

```
$ python3 -c "
import json,sys
f=sys.argv[1]; d=json.load(open(f)); r=d['results']
ans=[x for x in r if not x['gated'] and not x['abstained']]
good=[x for x in ans if x['kind']=='answerable' and x['correct']]
bad=[x for x in ans if x not in good]
noev=[x for x in bad if x['kind']=='answerable' and x.get('evidence_retrieved') is False]
wrongev=[x for x in bad if x['kind']=='answerable' and x.get('evidence_retrieved') is True and not x['correct']]
unabad=[x for x in bad if x['kind']=='unanswerable']
corr=[x for x in r if x['kind']=='answerable' and x['correct']]
allans=[x for x in r if x['kind']=='answerable']
unab=[x for x in r if x['kind']=='unanswerable' and x['abstained']]
unatot=[x for x in r if x['kind']=='unanswerable']
cs=[x for x in ans if x.get('cite_supported') is True]
ansable=[x for x in ans if x['kind']=='answerable']
print(f\"{d['name']}: answered {len(ans)}/{len(r)}  good {len(good)}  bad {len(bad)} (no-evidence {len(noev)}, wrong-with-evidence {len(wrongev)}, unanswerable-answered {len(unabad)})\")
print(f\"  correct/answerable {len(corr)}/{len(allans)}  unanswerable-abstained {len(unab)}/{len(unatot)}  answered-with-no-evidence {len(noev)}\")
print(f\"  cite_supported/answered(answerable) {len(cs)}/{len(ansable)}\")
" data/eval/results/fusion-ctl-answers.json
fusion-ctl: answered 109/166  good 63  bad 46 (no-evidence 23, wrong-with-evidence 22, unanswerable-answered 1)
  correct/answerable 63/127  unanswerable-abstained 38/39  answered-with-no-evidence 23
  cite_supported/answered(answerable) 73/108

$ python3 -c "... same one-liner ..." data/eval/results/fusion-rw1-answers.json
fusion-rw1: answered 109/166  good 58  bad 51 (no-evidence 20, wrong-with-evidence 28, unanswerable-answered 3)
  correct/answerable 58/127  unanswerable-abstained 36/39  answered-with-no-evidence 20
  cite_supported/answered(answerable) 67/106
```

| | control (`fusion-ctl`) | fused (`fusion-rw1`) |
|---|---|---|
| answered (not gated, not abstained) | 109 / 166 | 109 / 166 |
| — good (answerable, correct) | 63 | 58 |
| — bad (everything else answered) | 46 | 51 |
| — — answered with no evidence | 23 | 20 |
| — — wrong with evidence | 22 | 28 |
| — — unanswerable, answered | 1 | 3 |
| correct / answerable | 63 / 127 | 58 / 127 |
| unanswerable abstained / 39 | 38 / 39 | 36 / 39 |
| cite_supported among answered (answerable only) | 73 / 108 | 67 / 106 |

`cite_supported` is the existing gold-token proxy already recorded in these
files. It is reported here as a baseline number only — **the verifier being
pre-registered must not use it**, since it is exactly the shortcut a
citation-grounding check exists to replace.

Two of the fused run's 58 "good" answers have `evidence_retrieved: false`
(the model was correct without its cited extract containing the gold
evidence, by this proxy's own accounting); they count as good because the
definition above is `kind == answerable and correct`, taken literally, and
they are called out here rather than silently reclassified.

## 3. Mechanisms

Each mechanism is scored and ablated independently, on the same claim/answer
structure. **Answer score = min over claims**; **each claim takes the best
score over its in-range cited extracts**; **an out-of-range claim scores 0**
(a citation pointing outside the retrieved set cannot be checked, so it
fails the check rather than being ignored).

- **lexical** — the fraction of a claim's anchors (backtick spans,
  `-x`/`--long` flags) found verbatim in the cited extract. A claim with no
  anchors gets no opinion from this mechanism, and the answer is kept.
- **cross-encoder** — the reranker scores `(claim, cited extract)` directly.
  This is a different model from the one that generated the answer.
- **self-judge** — the generator itself answers yes/no, under a GBNF
  grammar, on whether the cited extract supports the claim. This is the only
  mechanism that asks whether the model that wrote the answer can check it.

## 4. Protocol

- The verifier only converts answers into refusals; it never rewrites an
  answer. So it is scored once per run (`scripts/eval_verifier.py --stage
  score`) and every threshold is replayed offline against that one scoring
  pass (`--stage replay`) — the same argument `sweep_gate.py --answers`
  already makes for the abstention gate (`scripts/sweep_gate.py:21-30`, the
  `--answers` mode that sweeps an existing answer run rather than
  re-generating).
- Thresholds are chosen **2-fold**, to avoid picking a threshold and scoring
  it on the same data. Questions are grouped by base qid — the part before
  the first `.`, so a paraphrase/typo/terse variant set like `a01`, `a01.t`,
  `a01.n`, `a01.z` stays together in one fold (confirmed: 55 of 166 qids in
  the committed runs carry a `.` variant suffix). Each group is assigned to
  a fold by `zlib.crc32(base) % 2`. On each fold, the rule picks the
  threshold `t` maximising `removed_bad − removed_good`, ties toward the
  lower `t`. Numbers are reported on the *other*, held-out fold; the
  headline for a mechanism is the sum of both held-out folds. In-sample
  optima (threshold picked and scored on the same fold) are reported only
  when explicitly labelled as such, never as the headline.
- The control run (`fusion-ctl`, rewrites=0) is primary. The fused run
  (`fusion-rw1`, rewrites=1) is the replication: it must agree in direction,
  not reproduce the same magnitude.

## 5. Decision rule, pre-registered

A mechanism counts as a win only if **all** of the following hold:

(a) on the control run's held-out folds, `removed_bad > removed_good` with a
two-sided sign test p < 0.05 over those removed questions;
(b) the fused run's held-out result goes the same direction;
(c) correct answers lost (good answers the mechanism converted to a refusal)
on the control run number ≤ 3.

A win means the mechanism goes into `scripts/ask.py` behind a flag, as a
separate task. Anything short of all three is reported as **null**, and
nothing ships.

**n is small: about 109 answered questions per run, of which about 46 are
bad on the control run** (§2). The sign test in (a) runs only over the
subset of those the mechanism actually flags as a refusal — not all 46 — so
its power depends on how many answers move, not on the size of the eval set.
With `math.comb`-exact two-sided sign-test p-values at p=0.5:

| removed_bad – removed_good split | n removed | p (two-sided) |
|---|---|---|
| 5 – 0 | 5 | 0.0625 (not significant) |
| **6 – 0** | 6 | **0.0312** |
| 7 – 1 | 8 | 0.0703 (not significant) |
| **8 – 1** | 9 | **0.0391** |
| 9 – 1 | 10 | 0.0215 |

So this design can resolve a mechanism that removes bad answers almost
unanimously — 6 bad and 0 good is the smallest split that clears p < 0.05 on
its own — and the smallest split that still clears it with at least one good
answer lost is 8 bad to 1 good (9 removed total, p ≈ 0.039). It cannot
resolve anything close to even odds: a 4–2 split (n=6, p = 0.6875) or a 5–3
split (n=8, p = 0.7266) will not reach significance no matter which fold
they land in.
If a mechanism removes only a handful of answers on a fold, this design will
report null even when the mechanism is doing something real — that is a
property of n=~46 bad answers split two ways, not of the mechanism.

## Per-mechanism results (control, held out)

Scored with `scripts/eval_verifier.py --stage score --run fusion-ctl --mech
lexical|xenc|judge` then `--stage replay --run fusion-ctl` (branch
`orch/tsk_20260926_e0ecf4ae`, commit `91bf099`). The orchestrator ran the
score/replay stages against live servers and handed off the resulting
`fusion-ctl-verify.json` (per-qid `{lexical, xenc, judge}` scores) and
`fusion-ctl-verify-replay.json` (fold thresholds and counts); every number
below is re-derived from those two files, not copied from the orchestrator's
printed summary:

```
$ python3 -c "
import json
rep = json.load(open('fusion-ctl-verify-replay.json'))
for mech, v in rep['mechanisms'].items():
    print(mech, 'in_sample', v['in_sample'])
    for tf in (0, 1):
        print(' ', v['folds'][str(tf)] if str(tf) in v['folds'] else v['folds'][tf])
    print('  held_out_sum', v['held_out_sum'])
"
judge in_sample {'threshold': 'off', 'counts': {'kept_good': 63, 'removed_good': 0, 'kept_bad': 46, 'removed_bad': 0}}
  {'trained_on_fold': 0, 'applied_to_fold': 1, 'threshold': 'off', 'held_out_counts': {'kept_good': 28, 'removed_good': 0, 'kept_bad': 22, 'removed_bad': 0}}
  {'trained_on_fold': 1, 'applied_to_fold': 0, 'threshold': 'off', 'held_out_counts': {'kept_good': 35, 'removed_good': 0, 'kept_bad': 24, 'removed_bad': 0}}
  held_out_sum {'kept_good': 63, 'removed_good': 0, 'kept_bad': 46, 'removed_bad': 0}
lexical in_sample {'threshold': 0.75, ...'kept_good': 50, 'removed_good': 13, 'kept_bad': 27, 'removed_bad': 19}
  {'trained_on_fold': 0, 'applied_to_fold': 1, 'threshold': 0.6667, held_out {'kept_good': 25, 'removed_good': 3, 'kept_bad': 16, 'removed_bad': 6}}
  {'trained_on_fold': 1, 'applied_to_fold': 0, 'threshold': 1.0, held_out {'kept_good': 24, 'removed_good': 11, 'kept_bad': 12, 'removed_bad': 12}}
  held_out_sum {'kept_good': 49, 'removed_good': 14, 'kept_bad': 28, 'removed_bad': 18}
xenc in_sample {'threshold': 0.9614, ...'kept_good': 47, 'removed_good': 16, 'kept_bad': 27, 'removed_bad': 19}
  {'trained_on_fold': 0, 'applied_to_fold': 1, 'threshold': 0.0008, held_out {'kept_good': 27, 'removed_good': 1, 'kept_bad': 22, 'removed_bad': 0}}
  {'trained_on_fold': 1, 'applied_to_fold': 0, 'threshold': 0.9614, held_out {'kept_good': 23, 'removed_good': 12, 'kept_bad': 16, 'removed_bad': 8}}
  held_out_sum {'kept_good': 50, 'removed_good': 13, 'kept_bad': 38, 'removed_bad': 8}

$ python3 -c "
from math import comb
def sign_p(n, k):
    k = min(k, n - k)
    tail = sum(comb(n, i) for i in range(k + 1))
    return min(2 * tail * (0.5 ** n), 1.0)
for name, bad, good in [('lexical', 18, 14), ('xenc', 8, 13), ('judge', 0, 0)]:
    n = bad + good
    print(name, 'n=', n, 'p=', round(sign_p(n, min(bad, good)), 4) if n else None)
"
lexical n= 32 p= 0.5966
xenc n= 21 p= 0.3833
judge n= 0 p= None
```

| mechanism | fold 0→1: t | held-out (kg/rg/kb/rb) | fold 1→0: t | held-out (kg/rg/kb/rb) | held-out sum removed_bad vs removed_good | in-sample t | in-sample removed_bad vs removed_good | sign-test p (n) |
|---|---|---|---|---|---|---|---|---|
| lexical | 0.6667 | 25/3/16/6 | 1.0 | 24/11/12/12 | **18 vs 14** | 0.75 | 19 vs 13 (labelled in-sample) | 0.597 (n=32) |
| xenc | 0.0008 | 27/1/22/0 | 0.9614 | 23/12/16/8 | **8 vs 13** | 0.9614 | 19 vs 16 (labelled in-sample) | 0.383 (n=21) |
| judge | off | 28/0/22/0 | off | 35/0/24/0 | **0 vs 0** | off | 0 vs 0 (labelled in-sample) | n/a (n=0, nothing ever removed) |

kg/rg/kb/rb = kept_good/removed_good/kept_bad/removed_bad, held out.

Verdict against §5(a)–(c), control run only:

- **lexical**: direction is correct (removed_bad 18 > removed_good 14) but the
  sign test is nowhere near significant, p = 0.597 ≫ 0.05 — (a) fails.
  removed_good = 14 also exceeds the (c) budget of 3 correct answers lost.
  **Null.**
- **xenc**: direction is already wrong on the primary run — removed_bad 8 <
  removed_good 13 — and p = 0.383. (a) fails outright. removed_good = 13
  also fails (c). **Null.**
- **judge**: the chosen threshold is `'off'` on both folds (§4's
  `choose_threshold` never finds a candidate score that beats "remove
  nothing"), so the mechanism never converts an answer to a refusal at all.
  removed_bad = removed_good = 0: there is nothing for a sign test to test,
  so (a) cannot be satisfied. **Null.**

## Replication (fused)

Same commands, `--run fusion-rw1`:

```
$ python3 -c "
import json
rep = json.load(open('fusion-rw1-verify-replay.json'))
for mech, v in rep['mechanisms'].items():
    print(mech, 'in_sample', v['in_sample'])
    for tf in (0, 1):
        print(' ', v['folds'][str(tf)])
    print('  held_out_sum', v['held_out_sum'])
"
judge in_sample {'threshold': 'off', ...'kept_good': 58, 'removed_good': 0, 'kept_bad': 51, 'removed_bad': 0}
  {'trained_on_fold': 0, 'applied_to_fold': 1, 'threshold': 'off', held_out {'kept_good': 26, 'removed_good': 0, 'kept_bad': 25, 'removed_bad': 0}}
  {'trained_on_fold': 1, 'applied_to_fold': 0, 'threshold': 'off', held_out {'kept_good': 32, 'removed_good': 0, 'kept_bad': 26, 'removed_bad': 0}}
  held_out_sum {'kept_good': 58, 'removed_good': 0, 'kept_bad': 51, 'removed_bad': 0}
lexical in_sample {'threshold': 0.8333, ...'kept_good': 40, 'removed_good': 18, 'kept_bad': 30, 'removed_bad': 21}
  {'trained_on_fold': 0, 'applied_to_fold': 1, 'threshold': 'off', held_out {'kept_good': 26, 'removed_good': 0, 'kept_bad': 25, 'removed_bad': 0}}
  {'trained_on_fold': 1, 'applied_to_fold': 0, 'threshold': 1.0, held_out {'kept_good': 19, 'removed_good': 13, 'kept_bad': 15, 'removed_bad': 11}}
  held_out_sum {'kept_good': 45, 'removed_good': 13, 'kept_bad': 40, 'removed_bad': 11}
xenc in_sample {'threshold': 0.00037, ...'kept_good': 58, 'removed_good': 0, 'kept_bad': 50, 'removed_bad': 1}
  {'trained_on_fold': 0, 'applied_to_fold': 1, 'threshold': 0.00037, held_out {'kept_good': 26, 'removed_good': 0, 'kept_bad': 25, 'removed_bad': 0}}
  {'trained_on_fold': 1, 'applied_to_fold': 0, 'threshold': 0.9927, held_out {'kept_good': 14, 'removed_good': 18, 'kept_bad': 13, 'removed_bad': 13}}
  held_out_sum {'kept_good': 40, 'removed_good': 18, 'kept_bad': 38, 'removed_bad': 13}
```

| mechanism | fold 0→1: t | held-out (kg/rg/kb/rb) | fold 1→0: t | held-out (kg/rg/kb/rb) | held-out sum removed_bad vs removed_good | in-sample t | in-sample removed_bad vs removed_good | sign-test p (n) |
|---|---|---|---|---|---|---|---|---|
| lexical | off | 26/0/25/0 | 1.0 | 19/13/15/11 | **11 vs 13** | 0.8333 | 21 vs 18 (labelled in-sample) | 0.839 (n=24) |
| xenc | 0.00037 | 26/0/25/0 | 0.9927 | 14/18/13/13 | **13 vs 18** | 0.00037 | 1 vs 0 (labelled in-sample) | 0.473 (n=31) |
| judge | off | 26/0/25/0 | off | 32/0/26/0 | **0 vs 0** | off | 0 vs 0 (labelled in-sample) | n/a (n=0) |

**Both lexical and xenc go the wrong direction on the fused run**: lexical
removes 11 bad answers but 13 *good* ones (removed_good > removed_bad,
reversed from control's 18-vs-14); xenc removes 13 bad against 18 good, the
same wrong-direction relationship it already had on the control run (8
vs 13) — the sign is consistent between the two runs, but the sign itself is
the one a verifier is not supposed to have (removing more correct answers
than wrong ones). Judge again removes nothing on either fold.

Against §5(b) ("the fused run's held-out result goes the same direction"):
moot for xenc and judge, since (a) already failed on the control run; for
lexical, (b) fails on its own terms — control went the right direction (bad
> good) and fused reverses it. No mechanism reaches a win. **Null on all
three, confirmed by replication rather than contradicted by it.**

## Ablation: combinations

Fixed rules need no threshold search, hence no fold split — they are scored
once, directly against the labels (`good` = `kind == "answerable" and
correct`, from the committed `fusion-ctl-answers.json` / `fusion-rw1-answers.json`,
same as §2), over the full 109 answered questions of each run:

```
$ python3 -c "
import json
from math import comb

def sign_p(n, k):
    k = min(k, n - k)
    tail = sum(comb(n, i) for i in range(k + 1))
    return min(2 * tail * (0.5 ** n), 1.0)

def labels(run):
    d = json.load(open(f'data/eval/results/{run}-answers.json'))
    return {r['qid']: r['kind'] == 'answerable' and bool(r.get('correct'))
            for r in d['results'] if not r.get('gated') and not r.get('abstained')}

for run in ('fusion-ctl', 'fusion-rw1'):
    scores = json.load(open(f'{run}-verify.json'))
    lab = labels(run)
    counts = {'lex_and_judge': [0, 0], 'lex_or_judge': [0, 0], 'lex0': [0, 0]}
    for qid, good in lab.items():
        v = scores.get(qid, {})
        lex, judge = v.get('lexical'), v.get('judge')
        lex_lt1 = lex is not None and lex < 1
        judge_no = judge is not None and judge == 0.0
        rules = {'lex_and_judge': lex_lt1 and judge_no,
                 'lex_or_judge': lex_lt1 or judge_no,
                 'lex0': lex is not None and lex == 0.0}
        for k, removed in rules.items():
            if removed:
                counts[k][0 if not good else 1] += 1
    for rule, (bad, gd) in counts.items():
        n = bad + gd
        print(run, rule, f'{bad} vs {gd}', 'p=', round(sign_p(n, min(bad, gd)), 3) if n else None)
"
fusion-ctl lex_and_judge 13 vs 10 p= 0.678
fusion-ctl lex_or_judge 30 vs 32 p= 0.899
fusion-ctl lex0 8 vs 8 p= 1.0
fusion-rw1 lex_and_judge 15 vs 16 p= 1.0
fusion-rw1 lex_or_judge 32 vs 34 p= 0.902
fusion-rw1 lex0 6 vs 9 p= 0.607
```

| rule | control removed_bad vs removed_good (p) | fused removed_bad vs removed_good (p) |
|---|---|---|
| `lex<1 AND judge=no` | 13 vs 10 (p=0.678) | 15 vs 16 (p=1.0) |
| `lex<1 OR judge=no` | 30 vs 32 (p=0.899) | 32 vs 34 (p=0.902) |
| `lex==0` | 8 vs 8 (p=1.0) | 6 vs 9 (p=0.607) |

None of the three fixed combinations separates good from bad on the control
run either, and two of the three (`OR`, `==0`) already remove more good than
bad. Combining mechanisms does not rescue any of them.

Threshold-free view, independent of any t: `P(score_good > score_bad)` over
all pairs (good, bad) with an opinion from that mechanism, ties counted 0.5
(re-derived directly from `*-verify.json`, no replay/folds involved):

| mechanism | control AUC | fused AUC |
|---|---|---|
| lexical | 0.560 (n_good=44, n_bad=41) | 0.474 (n_good=40, n_bad=44) |
| xenc | 0.530 (n_good=63, n_bad=46) | 0.436 (n_good=58, n_bad=51) |
| judge | 0.539 (n_good=63, n_bad=46) | 0.488 (n_good=58, n_bad=51) |

All six numbers sit in [0.44, 0.56] — chance is 0.5. No mechanism separates
good from bad answers, at any threshold, on either run.

**Why: grounded-but-wrong.** The judge's own yes/no vote is nearly identical
on good and bad answers — control: 35/63 good answers judged "supported" vs
22/46 bad (56% vs 48%); fused: 27/58 vs 25/51 (47% vs 49%). Lexical grounding
(every anchor verbatim in the cited extract, score == 1.0, restricted to
answers where lexical had an opinion at all) is common on both good and bad
answers and does not track correctness: control, bad answers are fully
grounded 22/41 of the time (54%) and good answers 30/44 of the time (68%) —
bad answers are *not* less grounded, if anything the reverse. Fused: bad
23/44 (52%), good 21/40 (52%) — indistinguishable. Even bad answers with **no evidence retrieved at all**
are lexically fully grounded about half the time — control 10/19, fused
5/17 (restricted the same way, to answers where lexical had an opinion) —
because the model still cites *something*, and every anchor in the claim
happens to appear in whatever irrelevant extract it pointed at. Wrong
answers are mostly faithful to an extract that does not answer the
question; a claim-vs-extract support check tests grounding, not relevance
to the question asked.

**Tooling note.** `scripts/eval_verifier.py --stage replay`'s printed
`answered_with_no_evidence` line (e.g. `14/127` for lexical, control) uses
127 — the count of all answerable questions — as its denominator
(`scripts/eval_verifier.py:253`, `len(ans_recs)`). The right denominator is
the answerable questions that actually had no evidence retrieved (whether
answered or abstained), matching `eval_answers.py`'s own accounting:

```
$ python3 -c "
import json
for run in ('fusion-ctl', 'fusion-rw1'):
    d = json.load(open(f'data/eval/results/{run}-answers.json'))
    ans = [x for x in d['results'] if x['kind'] == 'answerable']
    noev = [x for x in ans if x.get('evidence_retrieved') is False]
    print(run, len(noev), '/', len(ans))
"
fusion-ctl 37 / 127
fusion-rw1 35 / 127
```

So e.g. lexical's "14/127 answered with no evidence" on control should be
read against a denominator of 37, not 127 — this is a reporting bug in the
script's printed line, not in the numbers used for the decision rule above
(those are computed straight from `held_out_sum`, which does not go through
this line).

**Power, at the ns this phase actually reached.** Extending §5's table: at
n = 32 (lexical, control's held-out total), the smallest split that still
clears p < 0.05 is 23 vs 9; lexical's actual split, 18 vs 14, is well inside
the non-significant region. At n = 46 (illustrative — roughly the total
count of control's bad answers, larger than any single fold reached here),
the smallest surviving split is 31 vs 15. No mechanism in this phase came
close to either bar; this is a property of how few answers any mechanism
actually flags, not evidence that the mechanisms are inert in principle.

## What this changes

The verdict against §5's decision rule is **null on all three mechanisms** —
lexical, cross-encoder and self-judge each fail at least (a), and lexical
additionally fails the fused-run replication (b). Nothing ships to
`scripts/ask.py`.

The diagnosis is **grounded-but-wrong**: a claim can be lexically and
semantically well-supported by the extract it cites and still answer the
wrong question, because the model retrieved and cited *something* rather
than the extract that actually addresses the question asked. All three
mechanisms here re-read a claim against its own cited extract and ask "does
this extract support this claim" — a grounding question. None of them ask
"does this extract answer this question" — a relevance question. The AUCs
in the mid-0.4s-to-mid-0.5s range (Ablation, above) and the near-identical
judge yes-rates on good vs bad answers are the same finding from two
different angles: grounding and correctness are close to independent in
this data.

This reframes rather than closes open thread #1 (verifier). The next check
that could plausibly move the needle has to be question-aware — checking
whether the cited extract answers *this* question, not whether the claim's
words appear in it — rather than a fourth variant of claim-vs-extract
support. This is consistent with what phases 2 and 3 already found from the
retrieval side: the reranker's own top-1 score does not separate right
answers from wrong ones either (phase 2 flat, median 0.991 when evidence was
retrieved vs 0.987 when it was missed — `docs/phase3-results.md:119-120`,
the table comparing phase 2 flat against phase 3's structured chunking). A
check built from the same signal that already failed to separate outcomes
upstream was never likely to separate them downstream; this phase confirms
that directly for citation-level grounding rather than top-1 retrieval
score.

The verifier code (`src/smm/verify.py`, `scripts/eval_verifier.py`) stays in
the repo as the measurement harness this phase used — it is what will score
whatever question-aware check comes next, not something to discard. No
specific next design is proposed here.

## Also in this phase

`--gen-url` now reaches `eval_answers.py`'s generation stage, fixing a gap
found in `tsk_20260926_7d90f5d1`: the flag reached the rewrite generator
(`--rewrites > 0`) but not the main answer-generation stage, which
instantiated its own `Generator()` regardless of what was passed on the
command line.

## Correction (2026-09-26)

**What was wrong with the label.** Answer correctness has always been
`all(t in answer for t in row["answer_contains"])` — plain substring
matching against the gold tokens in `data/eval/questions.jsonl`
(`scripts/eval_answers.py`). Those gold tokens are mostly whichever form a
man page lists first, usually the long-form option (`--no-clobber`,
`--lines`, `identity_file` as an argument placeholder), so a correct answer
written in the documented short form (`cp -n`, `wc -l`, `ssh -i`) was scored
wrong. That noise sits under every answer-accuracy figure quoted anywhere in
this repo, including every number in §2 and §5 above, and this section
supersedes none of the *retrieval* numbers, only the answer-accuracy ones.

**The derivation.** `scripts/derive_gold_aliases.py` (`tsk_20260926_a51d0707`,
commits `c83be63`, `06ab7b4`) reads only `data/corpus/man.jsonl` and
`data/eval/questions.jsonl` — never an answer, never a result — and for each
answerable question's gold token searches the gold doc's own option lines
for three kinds of match (`scripts/derive_gold_aliases.py:12-33`):

- **synonym** — the token is itself one of the options on an option line
  (`--no-clobber` on `-n, --no-clobber`); the line's other options become its
  aliases.
- **argument** — the token is the argument placeholder on an option line
  (`identity_file` on `-i identity_file`); that line's options become its
  aliases.
- **description** — the token is a *phrase* (does not start with `-`) that
  occurs verbatim in an option entry's description text (`sort by time`
  under `-t`); that entry's options become its aliases. A flag token never
  gets a description alias, because man pages routinely cross-reference
  other flags by name in prose and that is not a synonym.

Every alias is recorded with the rule, section and option line that produced
it, so any alias traces back to one man-page sentence, and every alias,
regardless of rule, must itself be a flag (start with `-`) or it is dropped.
`src/smm/gold.py` applies the aliases at scoring time: originals keep
substring matching unchanged; aliases match on option-token boundaries
(`(?<![\w-])alias(?![\w-])`), so `-r` matches `scp -r dir` but not inside
`--recursive` or `-rf` (`src/smm/gold.py:28-40`).

**The over-generation caught at review.** The first derivation pass
over-generated on two real corpus defects, both fixed before any rescoring
(second commit, `06ab7b4`):

- the description rule fired on flag tokens too (a flag's own description
  mentioning another flag's long form made that long form an alias of the
  first flag, and vice versa) — 139 bogus aliases from this alone, closed by
  restricting the description rule to phrase tokens only.
- an option line's inline description (`-t     sort by time, newest first`,
  description on the same physical line) was fed whole into the comma/space
  option parser, producing bogus "options" like `newest` — closed by
  splitting the line at the first run of 2+ spaces or a tab before parsing.

**The 10 rejected aliases.** The final derivation gave 79 aliases; the user
reviewed the full list and rejected 12 alias entries (10 distinct alias
strings), leaving 67 over 31 distinct gold-token strings
(re-derived directly from the reviewed `gold_aliases.json`: `python3 -c
"import json; d=json.load(open('gold_aliases.json')); print(sum(len(v) for
p in d.values() for v in p.values()))"` → `67`). The rejections:

- `-print0` → `-fprint0` (qids a09, its `.n`/`.t`/`.z` variants, p03): a
  `description` match — `-print0`'s own man-page description happens to
  mention `-fprint0` — but `-fprint0` writes to a *file*, not stdout; it is
  not an equivalent way to answer the question.
- `"Select all processes"` → `-a`, `-d`, `--deselect`, `-N` (a19): these
  option lines' descriptions are "select all processes **EXCEPT** ...", the
  opposite of what the question asks. `-A` and `-e` (literally "Select all
  processes. Identical to -A/-e") are kept.
- `list-unit-files` → `-a`, `--all`, `--with-dependencies` (a44): the gold
  token is a systemctl *command*, not an option; these option entries merely
  mention the command name in their description text. `gold_aliases.json`
  now carries no entries for a44 at all — every candidate alias for that
  question's token was rejected.

Two accepted aliases worth noting rather than rejecting: a38's `-L`
(mke2fs) is also matched via `mkntfs -L` by the `argument`/`description`
rules on the same option line, which is correct for the question as posed;
grep's `-NUM` alias is a literal placeholder token in the man page and is
harmless.

**The rescore table**, re-derived with `python3` against `src/smm/gold.py`'s
`is_correct`, joining each `*-answers.json` run's `results` back to
`data/eval/questions.jsonl` for `answer_contains` (every figure below
reproduces the orchestrator's numbers exactly; commands and full script
available on request):

| run | strict correct / answerable | aliased correct / answerable | gain |
|---|---|---|---|
| fusion-ctl | 63/127 (49.6%) | 81/127 (63.8%) | +18 |
| fusion-rw1 | 58/127 (45.7%) | 76/127 (59.8%) | +18 |
| phase1-prefix | 60/126 (47.6%) | 76/126 (60.3%) | +16 |
| phase2-full | 61/126 (48.4%) | 80/126 (63.5%) | +19 |
| phase2-rerank | 63/126 (50.0%) | 80/126 (63.5%) | +17 |
| phase2-rerank-gate | 56/126 (44.4%) | 73/126 (57.9%) | +17 |
| phase2-rerank-grammar | 63/126 (50.0%) | 82/126 (65.1%) | +19 |
| phase3-expand | 59/126 (46.8%) | 77/126 (61.1%) | +18 |
| phase3-expand-ungated | 66/126 (52.4%) | 84/126 (66.7%) | +18 |
| phase3-noexpand | 57/126 (45.2%) | 75/126 (59.5%) | +18 |

Zero answers flip from right to wrong under aliasing, on any run (checked
directly: no qid's strict-correct answer becomes aliased-incorrect anywhere
above). Every run gains 16–19 correct answers out of 126–127 answerable
questions — a near-uniform shift, so between-phase *comparisons* mostly hold,
but every absolute answer-accuracy figure reported anywhere before this
correction was about 13–15 points low.

**The phase 6 replay under the aliased label.** Re-scoring `fusion-ctl` and
`fusion-rw1` as "good" (`kind == answerable and aliased-correct`) and
re-running the same 2-fold held-out threshold search from §4 changes which
answers are "bad" (28 on control, down from 46; 33 on fused, down from 51)
and, for lexical and judge, removes the modest signal §5's table found under
the strict label entirely:

| mechanism | control held-out sum (rb vs rg) | fused held-out sum (rb vs rg) |
|---|---|---|
| lexical | **0 vs 0** (threshold "off" both folds) | **0 vs 0** (threshold "off" both folds) |
| judge | **0 vs 0** (threshold "off" both folds, as before) | **0 vs 0** (threshold "off" both folds) |
| xenc | **3 vs 7** (wrong direction) | **0 vs 0** (threshold "off" both folds) |

rb/rg = removed_bad/removed_good, held out. Under the strict label, lexical's
control held-out sum was 18 bad vs 14 good — the right direction, though not
significant (p = 0.597). Under the corrected label, that signal is gone: with
fewer, more genuinely "bad" answers left to separate, no lexical threshold on
either training fold nets more bad removed than good, so "off" (remove
nothing) wins on both folds. xenc still goes the wrong direction on control
(removed_bad 3 < removed_good 7, same qualitative failure as strict-label's
8-vs-13), and both its removed_good (7) and removed_bad-vs-good direction
still fail §5(a) and (c) outright. Judge removes nothing under either label,
for the same reason as before (§4's threshold search never finds a candidate
that beats "remove nothing").

Against §5(a)–(c): lexical and judge have nothing for a sign test to test (a
fails by construction); xenc fails (a) on direction and fails (c) on budget
(7 > 3 good lost) even setting direction aside; (b) is moot for all three
since (a) already fails on control. **The verdict stays null on all three
mechanisms, under the corrected label, exactly as under the strict one** —
the correction changes *why* two of the three mechanisms fail (lexical no
longer even clears a nominal, non-significant signal) but not the outcome.

**The AUCs** (aliased label, `P(score_good > score_bad)` over all pairs, ties
0.5, re-derived from `*-verify.json` joined to the aliased label, restricted
to the 109 answered questions per run):

| mechanism | control AUC | fused AUC |
|---|---|---|
| lexical | 0.626 (n_good=62, n_bad=23) | 0.523 (n_good=58, n_bad=26) |
| xenc | 0.528 (n_good=81, n_bad=28) | 0.435 (n_good=76, n_bad=33) |
| judge | 0.612 (n_good=81, n_bad=28) | 0.560 (n_good=76, n_bad=33) |

Compare to the strict-label AUCs in the Ablation table above (0.560/0.474,
0.530/0.436, 0.539/0.488): lexical's control AUC moves from 0.560 to 0.626
and judge's from 0.539 to 0.612 — both now sit modestly above chance on the
control run — but neither replicates on the fused run (lexical 0.523, judge
0.560, both back near chance), and xenc barely moves (0.528, 0.530). This is
a control-run-only signal, not a replicated one: a mechanism that only
separates good from bad on one of two runs meant to agree in direction is not
a win under §5(b) regardless of AUC.

**Refined diagnosis.** Under the aliased label, of the 109 answered
questions per run:

- **control**: 28 wrong (down from 46 strict) — 18 had no evidence retrieved
  at all, 9 were wrong despite evidence being retrieved, 1 was an
  unanswerable question answered anyway. (81 good.)
- **fused**: 33 wrong (down from 51 strict) — 17 no-evidence, 13
  wrong-with-evidence, 3 unanswerable-answered. (76 good.)

Fully lexically grounded (every anchor verbatim in the cited extract, among
answers where lexical had an opinion at all):

- control: bad 10/23 (43%) vs good 42/62 (68%).
- fused: bad 13/26 (50%) vs good 31/58 (53%).

**Which "What this changes" conclusions stand, and which are superseded.**
The headline verdict stands: null on all three mechanisms, confirmed again
under the corrected label (above). The **grounding is not the discriminator**
conclusion mostly stands too — the AUCs are still in the 0.44–0.63 range
(not the 0.7+ that would make any mechanism usable), and control-only,
non-replicating gains do not change that. But one specific claim is
**superseded**: §"Why: grounded-but-wrong" reported, under the strict label,
that on control "bad answers are *not* less grounded [than good], if
anything the reverse" (54% bad grounded vs 68% good grounded). Under the
corrected label that reverses again — control bad-grounded drops to 43% while
good-grounded stays 68%, so bad answers *are* somewhat less grounded than
good ones on control, in the intuitive direction. That earlier reversal was
an artefact of the label: many of the "grounded but labelled wrong" answers
on control were answers using a short-form option, genuinely grounded *and*
genuinely correct, mislabelled bad. Once those are relabelled good, the
remaining bad-on-control answers look, on average, somewhat less grounded
than good ones — a small, control-only, non-replicating effect (fused stays
indistinguishable, 50% vs 53%), not a rehabilitation of any mechanism.

**What it implies.** Across both runs, the largest single wrong-answer
category under the corrected label is still "no evidence retrieved" (18/28
on control, 17/33 on fused) — questions the gate let through to generation
despite nothing relevant having been retrieved, not the model misreading
evidence it had. "Wrong with evidence" (9/28 control, 13/33 fused) is smaller
and is where a grounding-only verifier could in principle help, but it is a
minority of the remaining errors, and none of the three mechanisms measured
here separates it from the good answers regardless. The biggest remaining
error class points back at the gate and retrieval, not at a claim-vs-extract
verifier: a verifier that only checks "does the cited extract support the
claim" cannot catch an answer generated with no supporting evidence in the
first place — the gate is supposed to prevent that case, and 18-and-17 out of
28-and-33 says it still doesn't, often enough to be the majority failure
mode.
