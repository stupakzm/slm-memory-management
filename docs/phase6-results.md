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

RESULTS PENDING

## Replication (fused)

RESULTS PENDING

## Ablation: combinations

RESULTS PENDING

## What this changes

RESULTS PENDING

## Also in this phase

`--gen-url` now reaches `eval_answers.py`'s generation stage, fixing a gap
found in `tsk_20260926_7d90f5d1`: the flag reached the rewrite generator
(`--rewrites > 0`) but not the main answer-generation stage, which
instantiated its own `Generator()` regardless of what was passed on the
command line.
