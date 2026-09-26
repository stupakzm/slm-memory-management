"""The verifier: a second, independent read of a grammar-shaped answer against
the extract(s) it cites.

The grammar (see smm.grammar) makes citation *structurally* mandatory and
in-range, but it cannot make it right: nothing stops the model writing a
parametric answer and pointing at extract 2. `verify_citations` in that module
already does one check - do the answer's own gold tokens appear in the cited
extract - but that check is only available because gold labels exist for the
eval set. In production there is no gold token to look for. The verifier is
the version of that check that never needs one: it re-reads each claim against
the extract it names and asks whether the extract actually supports it, using
only what the model was shown. That is why this module must never import the
eval set's own gold-answer helper, gold tokens, or any other eval label -
anything it can only compute from an eval file cannot run at answer time, and
the entire point of scoring a verifier offline (see scripts/eval_verifier.py)
is to pick a policy that then generalises to unlabelled traffic.

Like the gate (scripts/sweep_gate.py), the verifier only ever decides
keep-or-refuse; it never rewrites what the model said.
"""

from __future__ import annotations

import re
from typing import Callable

from . import grammar

# A claim is `text` up to and including its `[n]` marker(s) - see
# grammar.TEMPLATE's `claim ::= text " [" ref "]"`. The pattern below finds the
# free-text run immediately followed by one or more bracketed integers (no
# intervening prose), so "Use -C. [2] Or `grep -A 3`. [1]" splits into two
# claims rather than one claim swallowing the whole string.
_CLAIM_RE = re.compile(r"([^\[\]]+?)\s*((?:\[\d+\])+)")

_BACKTICK_RE = re.compile(r"`([^`]+)`")
# Option flags: one or two leading dashes, then a letter, then word chars/dashes.
# The negative lookbehind stops a flag being matched mid-word (e.g. the "-1" in
# "utf-8-1" is not a flag). A trailing "=VALUE" is left unmatched on purpose -
# the character class excludes "=", so "--context=NUM" yields "--context".
_FLAG_RE = re.compile(r"(?<![\w-])--?[A-Za-z][\w-]*")


def claims(answer: str) -> list[tuple[str, list[int]]]:
    """Split a grammar-shaped answer into (claim_text, cited_indices).

    The refusal (`grammar.REFUSAL`) carries no claims at all."""
    if answer.strip() == grammar.REFUSAL:
        return []
    out = []
    for m in _CLAIM_RE.finditer(answer):
        text = m.group(1).strip()
        idxs = [int(n) for n in re.findall(r"\d+", m.group(2))]
        out.append((text, idxs))
    return out


def anchors(text: str) -> list[str]:
    """The checkable literals in a claim: backtick spans (inner text) and
    option-flag tokens, deduplicated in order of first appearance."""
    found = []
    for m in _BACKTICK_RE.finditer(text):
        found.append((m.start(), m.group(1)))
    for m in _FLAG_RE.finditer(text):
        found.append((m.start(), m.group(0)))
    found.sort(key=lambda pair: pair[0])
    seen: set[str] = set()
    out = []
    for _, a in found:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out


def lexical_support(claim_text: str, extract_text: str) -> float | None:
    """Fraction of the claim's anchors that occur verbatim in the extract.
    None ("no opinion") when the claim carries no checkable anchor at all."""
    a = anchors(claim_text)
    if not a:
        return None
    return sum(1 for x in a if x in extract_text) / len(a)


# A scorer is any callable (claim_text, extract_text) -> float | None.
Scorer = Callable[[str, str], "float | None"]

lexical: Scorer = lexical_support


class CrossEncoderScorer:
    """Wraps an `smm.rerank.Reranker` as a scorer: how well the extract, read
    as a single candidate document, matches the claim, read as the query."""

    def __init__(self, reranker):
        self.reranker = reranker

    def __call__(self, claim_text: str, extract_text: str) -> float | None:
        return self.reranker.scores(claim_text, [extract_text])[0]


class JudgeScorer:
    """Wraps an `smm.generate.Generator` as a scorer via `Generator.judge`."""

    def __init__(self, generator):
        self.generator = generator

    def __call__(self, claim_text: str, extract_text: str) -> float | None:
        return self.generator.judge(claim_text, extract_text)


def score_answer(answer: str, chunks: list[dict], scorer: Scorer) -> float | None:
    """Score a whole answer against the chunks it was generated from.

    Per claim: the best (max) score over its in-range cited extracts - a
    citation pointing outside 1..len(chunks) scores 0.0 outright, and a claim
    where every in-range extract came back with no opinion (None) is itself
    None. The extract text is `chunk.get('prefix','') + chunk['text']`, same
    convention as `grammar.verify_citations`.

    The answer's score is the MIN over its claims' scores, ignoring None
    claims - the worst-supported claim sets the answer's grounding, exactly
    like the gate is architectural rather than an average. None if every
    claim is None, or if the answer is the refusal or otherwise has no claims.
    """
    cl = claims(answer)
    if not cl:
        return None
    claim_scores: list[float | None] = []
    for text, cites in cl:
        in_range = [c for c in cites if 1 <= c <= len(chunks)]
        if not in_range:
            claim_scores.append(0.0)
            continue
        best = None
        for c in in_range:
            chunk = chunks[c - 1]
            extract_text = chunk.get("prefix", "") + chunk["text"]
            s = scorer(text, extract_text)
            if s is None:
                continue
            if best is None or s > best:
                best = s
        claim_scores.append(best)
    non_none = [s for s in claim_scores if s is not None]
    if not non_none:
        return None
    return min(non_none)
