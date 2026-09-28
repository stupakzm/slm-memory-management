"""The R4d cascade: rewrite only after the plain search refuses
(docs/phase11-results.md, "R4d pre-registration").

Tier 0 is exactly today's plain path: dense candidates, reranked against the
question, gated, a grammar-constrained answer. If that is not a refusal,
nothing else runs - the 4B's rewrites() is never called, and the generator
pays no extra latency.

Tier 1 (only after a tier-0 refusal) asks for one docs-style rewrite and
widens the candidate pool with it (`Retriever.retrieve_widened`); the union
is reranked once against the ORIGINAL question, never the rewrite - there is
no rank fusion in this path (unlike `retrieve.fuse_variants`), so the
resulting top-1 is a genuine cross-encoder score the gate can read directly
(blk_fusion_gate_semantic_slip). The reader also always sees the original
question, never the rewrite. If tier 1 is still a refusal, tier 2 asks for a
fresh pair of rewrites (a new `n=2` call, not tier 1's rewrite reused) and is
final, whatever it returns.

`run_cascade` is the whole loop, shared by scripts/ask.py and
scripts/eval_answers.py, taking the retriever/generator/answer callable as
plain arguments so it is testable with stub objects - no server, no GPU, no
index.

`ABSTAIN_RE`/`abstained` live here (moved from scripts/eval_answers.py,
stdlib-only - see blk_abstain_re_false_positives) because the cascade is the
thing that has to decide, tier by tier, whether the reader refused; the eval
script imports them back under the same names so `eval_answers.abstained`
keeps working unchanged and tests/test_scoring.py passes with no edits.
"""

from __future__ import annotations

import re
import time
from typing import Callable

from . import grammar
from .retrieve import KEEP, gate_score

ABSTAIN_RE = re.compile(
    r"\bi\s*(?:do\s*n[o']?t|don'?t)\s+know\b"
    r"|\bnot\s+(?:in|contained\s+in|found\s+in|covered\s+by)\s+the\s+extracts?\b"
    r"|\bthe\s+extracts?\s+do\s*(?:es)?\s*n[o']?t\s+contain\b"
    r"|\bno\s+(?:relevant\s+)?information\b",
    re.I,
)


def abstained(text: str) -> bool:
    """True iff EVERY sentence in `text` matches ABSTAIN_RE (empty text is not
    an abstention). The 30B hedges: it appends "I don't know. [n]" after real,
    cited claims, and ABSTAIN_RE.search(text) alone would call that a refusal
    just because the phrase appears somewhere. A claim followed by "I don't
    know" is still an answer - only a text with no speaking sentence at all is
    a genuine abstention. Citation markers are stripped first so a trailing
    `[n]` never affects where a sentence ends."""
    stripped = re.sub(r"\s*\[\d\]\s*", " ", text)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", stripped) if s.strip()]
    return bool(sentences) and all(ABSTAIN_RE.search(s) for s in sentences)


def is_refusal(gated: bool, answer: str) -> bool:
    """Refusal = gated (top-1 below the gate) OR the reader abstained -
    the all-sentences rule `abstained()` already scores with."""
    return gated or abstained(answer)


AnswerFn = Callable[[str, list], str]


def _tier_result(tier: int, hits: list, score: float, gated: bool,
                  answer: str, rewrites: list, seconds: float) -> dict:
    return {
        "tier": tier, "hits": hits, "score": score, "gated": gated,
        "answer": answer, "rewrites": rewrites, "seconds": seconds,
    }


def run_cascade(retriever, generator, question: str, answer_fn: AnswerFn,
                 gate: float, k: int = KEEP, rewrite_style: str = "docs") -> dict:
    """Run the tier 0/1/2 cascade for one `question`, returning a dict with
    keys `tier` (0/1/2), `hits` (the FINAL tier's hits - what the answer/
    citations are scored against), `score` (that tier's gate_score), `gated`,
    `answer`, `rewrites` (the rewrite texts used at the FINAL tier - `[]` at
    tier 0), and `seconds` (wall time spent from the start of tier 1 onward;
    `0.0` when the result is tier 0's, since nothing beyond tier 0 ran).

    `retriever` needs `.retrieve(question, k)` (tier 0) and
    `.retrieve_widened(question, extra_queries, k)` (tiers 1/2).
    `generator` needs `.rewrites(question, n, style)`. `answer_fn(question,
    hits)` is called with the ORIGINAL `question` at every tier, never a
    rewrite - it is the caller's job to actually invoke the reader (or return
    `grammar.REFUSAL` directly, as this function does when a tier is gated,
    without calling `answer_fn` at all: the model is never invoked on thin
    evidence, tier by tier, exactly as the plain gate does at tier 0)."""

    def run_tier(hits: list) -> tuple[float, bool, str]:
        score = gate_score(hits)
        gated = bool(gate) and score < gate
        answer = grammar.REFUSAL if gated else answer_fn(question, hits)
        return score, gated, answer

    hits = retriever.retrieve(question, k=k)
    score, gated, answer = run_tier(hits)
    result = _tier_result(0, hits, score, gated, answer, [], 0.0)
    if not is_refusal(gated, answer):
        return result

    t0 = time.time()
    for tier, n in ((1, 1), (2, 2)):
        rewrites = generator.rewrites(question, n=n, style=rewrite_style)
        hits = retriever.retrieve_widened(question, rewrites, k=k)
        score, gated, answer = run_tier(hits)
        result = _tier_result(tier, hits, score, gated, answer, rewrites,
                              time.time() - t0)
        if tier == 2 or not is_refusal(gated, answer):
            break
    return result
