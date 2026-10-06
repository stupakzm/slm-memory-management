"""GBNF grammars that make an uncited claim structurally impossible.

Phase 1 asked for citations in the system prompt and got citation-shaped text:

    What does malloc return when the allocation fails?  ->  "NULL [3]"

Extract [3] said nothing of the kind. The model had the answer parametrically,
wrote it out, and appended a marker because the prompt asked for one. Prompting
produces the *shape* of grounding, which is worse than no grounding at all - it
survives inspection.

Be precise about what a grammar fixes and what it does not. Constrained decoding
guarantees the citation is well formed and inside the range of extracts actually
supplied. It cannot guarantee the citation is *right*: nothing stops the model
emitting a parametric answer and pointing at extract 2. What it does buy is that
the claim is now attached to a specific, machine-checkable pointer - so
`verify_citations` below can go and read extract 2 and see whether it supports the
claim. Phase 1 could not run that check at all, because "[3]" was free text the
model could have omitted or malformed.

So the grammar is not the guarantee. It is what makes the guarantee checkable.
"""

from __future__ import annotations

import re

REFUSAL = "I don't know."

# Each claim carries its own citation, so a two-part answer cannot cite once and
# smuggle the rest in uncited. Capped at four claims: these are man-page lookups,
# and a longer answer at 4B is padding, not detail.
#
# Claim text may contain "[" and "]" (a gold answer like [:alpha:] must be
# producible), but "[" may never be followed by a digit 1-9, so a citation stays
# the only place a bracketed digit can occur. GBNF has no lookahead, so the
# unit rule consumes "[" runs together with the one char after them.
TEMPLATE = r'''
root    ::= refusal | claims
refusal ::= "I don't know."
claims  ::= claim (" " claim){0,3}
claim   ::= text " [" ref "]"
ref     ::= REFS
text    ::= unit+
unit    ::= [^\[\]\n] | "["+ [^1-9\[\n] | "]"
'''


def cited_answer(n_extracts: int) -> str:
    """Grammar admitting only the refusal, or claims citing extracts 1..n."""
    n = max(1, min(n_extracts, 9))
    refs = " | ".join(f'"{i}"' for i in range(1, n + 1))
    # str.format is unusable here: GBNF's own `{0,3}` repetition is not a field.
    return TEMPLATE.replace("REFS", refs).strip() + "\n"


# Phase 10 (a 30B reader that stays grounded, not just cited): a claim that
# opens with an exact quotation copied from the extract it names is harder to
# fabricate than a bare citation - "[3]" costs nothing to type, but a quoted
# span that isn't actually in extract 3 is mechanically checkable the same way
# verify_citations checks tokens (see verify_quotes below). Same house idiom:
# REFS filled by string replacement, never str.format.
QUOTE_TEMPLATE = r'''
root    ::= refusal | claims
refusal ::= "I don't know."
claims  ::= claim (" " claim){0,3}
claim   ::= "\"" quote "\" " text " [" ref "]"
ref     ::= REFS
quote   ::= [^"\n]+
text    ::= [^\[\]\n]+
'''


def quoted_answer(n_extracts: int) -> str:
    """Grammar admitting only the refusal, or quote-opened claims citing
    extracts 1..n. Mirrors cited_answer exactly, one grammar swapped for the
    other."""
    n = max(1, min(n_extracts, 9))
    refs = " | ".join(f'"{i}"' for i in range(1, n + 1))
    return QUOTE_TEMPLATE.replace("REFS", refs).strip() + "\n"


# Line mode (tsk_20261006_linemode). Quote mode lost correct answers because the
# 4B mis-copied its own quotes and every miscopy became a refusal. Here the
# grammar lists each extract's actual lines as literal alternatives, so a
# miscopy is impossible and the model only has to choose a line. A real line can
# still be the wrong line: this fixes the copying, not the choosing. One claim
# per answer, and one rule pair per extract, so the quoted line and the closing
# citation number cannot disagree.
LINE_TEMPLATE = r'''
root    ::= refusal | claim
refusal ::= "I don't know."
claim   ::= CLAIMS
text    ::= [^\[\]\n]+
'''


def extract_lines(chunk: dict, max_lines: int = 40, max_len: int = 160) -> list[str]:
    """The usable lines of chunk["text"] (never chunk["prefix"]): stripped,
    at least 4 characters, cut at max_len, first occurrence only, in order."""
    out: list[str] = []
    for raw in chunk["text"].split("\n"):
        line = raw.strip()
        if len(line) < 4:
            continue
        line = line[:max_len]
        if line in out:
            continue
        out.append(line)
        if len(out) >= max_lines:
            break
    return out


def gbnf_literal(s: str) -> str:
    """A GBNF string literal: backslash and double quote escaped, control
    characters replaced by a space, everything else (non-ASCII too) kept."""
    s = "".join(" " if ord(c) < 32 else c for c in s)
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def line_answer(extracts: list[dict]) -> str:
    """Grammar admitting only the refusal, or one claim `"<line>" <text> [<n>]`
    whose <line> is one of extract n's own lines. Extracts without a line get
    no alternative; with none at all the grammar is the refusal alone."""
    ids, rules = [], []
    for n, chunk in enumerate(extracts[:9], 1):
        lines = extract_lines(chunk)
        if not lines:
            continue
        ids.append(n)
        rules.append(f'claim{n} ::= "\\"" line{n} "\\" " text " [{n}]"')
        rules.append(f"line{n} ::= " + " | ".join(gbnf_literal(l) for l in lines))
    if not ids:
        return 'root    ::= refusal\nrefusal ::= "I don\'t know."\n'
    g = LINE_TEMPLATE.replace("CLAIMS", " | ".join(f"claim{n}" for n in ids))
    return g.strip() + "\n" + "\n".join(rules) + "\n"


CITE_RE = re.compile(r"\[([1-9])\]")


def citations(answer: str) -> list[int]:
    """Extract numbers the answer points at, in order, deduplicated."""
    return list(dict.fromkeys(int(m) for m in CITE_RE.findall(answer)))


def verify_citations(answer: str, chunks: list[dict], tokens: list[str]) -> dict:
    """Does a cited extract actually contain the answer's key tokens?

    This is the check the grammar exists to enable. `supported` is the honest
    version of "grounded": the model named an extract, and that extract really does
    carry the token the answer turns on.
    """
    cites = citations(answer)
    in_range = [c for c in cites if 1 <= c <= len(chunks)]
    supported = any(
        all(t in f"{chunks[c-1].get('prefix','')}{chunks[c-1]['text']}" for t in tokens)
        for c in in_range
    )
    return {"citations": cites, "n_citations": len(cites),
            "uncited": not cites, "out_of_range": len(cites) - len(in_range),
            "cite_supported": supported}


# Matches one quote-opened claim: "<quote>" <text> [<ref>]. Deliberately
# permissive about what sits between the closing quote and the ref (that's
# `text`, unconstrained here) - the grammar above is what makes the shape
# exact; this regex only needs to recover (quote, ref) pairs from it.
QUOTE_CLAIM_RE = re.compile(r'"([^"\n]+)"[^\[\]\n]*\[([1-9])\]')


def verify_quotes(answer: str, chunks: list[dict]) -> dict:
    """Is every claim's opening quotation actually in the extract it cites?

    Mechanical, like verify_citations: whitespace-collapsed substring, not a
    judged support call (blk_phase6_verifier_failure - the LLM-judge verifier
    could not separate right answers from wrong, so this stays a check a
    machine can decide). The refusal carries no claims and is neither
    verified nor failed - it is the answer this whole mode exists to make
    safe to fall back on.
    """
    if answer.strip() == REFUSAL:
        return {"quote_verified": False, "quote_failed": False, "n_quotes": 0}

    pairs = QUOTE_CLAIM_RE.findall(answer)
    n = len(pairs)

    def _claim_verified(quote: str, ref: str) -> bool:
        r = int(ref)
        if not (1 <= r <= len(chunks)):
            return False
        c = chunks[r - 1]
        hay = " ".join(f"{c.get('prefix', '')}{c['text']}".split())
        needle = " ".join(quote.split())
        return needle in hay

    quote_verified = n > 0 and all(_claim_verified(q, r) for q, r in pairs)
    return {"quote_verified": quote_verified, "quote_failed": not quote_verified,
            "n_quotes": n}
