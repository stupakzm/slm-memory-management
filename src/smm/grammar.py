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
TEMPLATE = r'''
root    ::= refusal | claims
refusal ::= "I don't know."
claims  ::= claim (" " claim){0,3}
claim   ::= text " [" ref "]"
ref     ::= REFS
text    ::= [^\[\]\n]+
'''


def cited_answer(n_extracts: int) -> str:
    """Grammar admitting only the refusal, or claims citing extracts 1..n."""
    n = max(1, min(n_extracts, 9))
    refs = " | ".join(f'"{i}"' for i in range(1, n + 1))
    # str.format is unusable here: GBNF's own `{0,3}` repetition is not a field.
    return TEMPLATE.replace("REFS", refs).strip() + "\n"


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
