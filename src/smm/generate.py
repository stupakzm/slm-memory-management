"""Chat client for the local llama-server running the answering model."""

from __future__ import annotations

import json
import urllib.request

from . import grammar

DEFAULT_URL = "http://127.0.0.1:8080"

# Phase 1 deliberately asks for abstention in the prompt and nothing else. The
# research says a 4B model cannot reliably self-assess evidence sufficiency, so
# this is here to *measure* how badly prompt-only abstention does - it is the
# number the phase 2 score gate has to beat.
SYSTEM = """You answer questions about a Linux system using only the manual page \
extracts provided. Follow these rules exactly:

- Use only the extracts. Never use knowledge from outside them.
- If the extracts do not contain the answer, reply exactly: I don't know.
- Prefer naming the exact flag, option or setting, spelled as the manual spells it.
- Cite the extract number you used, like [2].
- Be brief. No preamble."""

# Phase 15 R14: the reader's system prompt is one of the three instructions the
# system gives its models. SYSTEM stays today's text; SYSTEM_V2 widens it to both
# corpora and tells the reader the question may be misspelt or in everyday words.
# Selected by --reader-prompt v2 (cite mode only; quote mode stays on SYSTEM).
SYSTEM_V2 = """You answer questions about the software on this computer (Linux commands and \
configuration, and the GNU Emacs editor) using only the documentation extracts provided. \
The question may be misspelt, terse, or use everyday words instead of the documentation's \
own terms: answer what the user most plausibly means, if the extracts say it. Follow these \
rules exactly:

- Use only the extracts. Never use knowledge from outside them.
- If the extracts do not contain the answer, reply exactly: I don't know.
- Prefer naming the exact command, flag, option or setting, spelled as the documentation spells it.
- Cite the extract number you used, like [2].
- Be brief. No preamble."""

# Phase 10: the 30B reader gains correctness by answering from parametric
# knowledge, not by reading (blk_phase9_parametric_knowledge_failure) - the
# reranker gate cannot catch this, because a parametric answer still scores
# 0.94-0.999 (blk_phase9_30b_gate_threshold_failure). Requiring each claim to
# open with an exact quotation moves the check from the gate to the claim
# itself: a quote that isn't actually in the extract is mechanically
# detectable (smm.grammar.verify_quotes), the same way a bad citation is.
QUOTE_SYSTEM = SYSTEM + """
- Each claim must begin with an exact quotation, in double quotes, copied from the extract it cites.
- If no extract contains a sentence supporting the answer, reply exactly: I don't know."""


# Grammar for N alternative search-query rewrites, one per line, no numbering.
# Same house idiom as smm.grammar.cited_answer: a GBNF template with the one
# free parameter (here, exactly N lines) filled in by string replacement,
# because GBNF's own `{m,n}` repetition syntax collides with str.format.
REWRITE_TEMPLATE = r'''
root ::= line ("\n" line){REPS}
line ::= [^\n]+
'''


def _rewrite_grammar(n: int) -> str:
    """Grammar admitting exactly n non-empty lines, each on its own line."""
    n = max(1, n)
    return REWRITE_TEMPLATE.replace("REPS", f"{n - 1},{n - 1}").strip() + "\n"


# Grammar for the verifier's judge call (smm.verify.JudgeScorer): exactly one of
# two literal tokens, same house idiom as REWRITE_TEMPLATE/_rewrite_grammar
# above, just with no free parameter to fill in.
JUDGE_GRAMMAR = 'root ::= "yes" | "no"\n'

JUDGE_SYSTEM = """You check whether a manual-page extract explicitly supports a claim.
Answer with exactly one word: yes or no. Say yes only if the extract itself
states what the claim says - not because it sounds plausible or is true in
general."""


def build_rewrite_prompt(question: str, n: int, style: str = "man") -> list[dict]:
    """`style="man"` (default) is exactly today's prompt, byte-for-byte - phase 11
    R4b's docs-vocabulary style is opt-in only. `style="docs"` targets the R3
    finding that synonym/no-name/terse wordings lose answers mostly at
    RETRIEVAL, a vocabulary gap (blk phase11-results.md R3): it asks for
    documentation's own terminology instead, while still keeping any
    program/package/command name the user wrote exactly as written (the same
    verbatim-names guard as build_correct_prompt below -
    blk_r4a_spell_normalisation_fails_rule)."""
    if style == "docs":
        system = (
            f"Rewrite the user's request as {n} alternative search queries for a "
            "search engine over software documentation (Linux manual pages and the "
            "GNU Emacs manuals). Use the terminology that documentation itself would "
            "use for the commands, options and concepts, which is often different "
            "from everyday words. Keep any program, package or command name the user "
            "wrote exactly as written. Each on its own line, no numbering, no "
            "explanation. Keep them short."
        )
    else:
        system = (f"Rewrite the user's request as {n} alternative search queries for a "
                  "Linux manual-page search engine. Each on its own line, no numbering, "
                  "no explanation. Keep them short, name the likely command if you can, "
                  "and vary the wording.")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


# Phase 11 R4a': the model-based spelling corrector. R4a's corpus-vocabulary
# corrector (smm.normalize) failed its own safety rule by rewriting the names
# of uninstalled tools into installed ones (nmap -> mmap,
# blk_r4a_spell_normalisation_fails_rule); this prompt asks the model itself
# to fix spelling while holding every program/package/command/option/file
# name fixed, even ones it has never seen.
CORRECT_SYSTEM = (
    "Fix spelling mistakes in the user's question. Change nothing else: keep "
    "the wording, and keep every program, package, command, option and file "
    "name exactly as written, even unfamiliar ones. Output only the corrected "
    "question on one line."
)


def build_correct_prompt(question: str) -> list[dict]:
    return [
        {"role": "system", "content": CORRECT_SYSTEM},
        {"role": "user", "content": question},
    ]


def build_prompt(question: str, chunks: list[dict], system: str = SYSTEM,
                 header: str = "Manual page extracts") -> list[dict]:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] {c['doc_id']}\n{c['prefix']}{c['text']}")
    context = "\n\n".join(parts) if parts else "(no extracts found)"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{header}:\n\n{context}\n\nQuestion: {question}"},
    ]


class Generator:
    def __init__(self, url: str = DEFAULT_URL, timeout: int = 300):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def chat(self, messages: list[dict], max_tokens: int = 400, temperature: float = 0.0,
             grammar: str | None = None) -> str:
        payload = {
            "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "stream": False,
        }
        if grammar:
            payload["grammar"] = grammar
        req = urllib.request.Request(
            self.url + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.load(r)
        return out["choices"][0]["message"]["content"].strip()

    def answer(self, question: str, chunks: list[dict], cite_grammar: bool = False,
               mode: str = "cite", reader_prompt: str = "v1", **kw) -> str:
        """`cite_grammar` constrains decoding so every claim carries an in-range
        citation - see smm.grammar for what that does and does not guarantee.

        `mode="quote"` is the phase 10 opt-in: QUOTE_SYSTEM plus
        grammar.quoted_answer, so every claim must also open with an exact
        quotation from the extract it cites. Grammar is always on in quote
        mode, independent of `cite_grammar`. `mode="cite"` (the default) is
        exactly the pre-existing behaviour - same messages, same grammar.

        `reader_prompt="v2"` (phase 15 R14) swaps SYSTEM for SYSTEM_V2 and the
        extracts header for "Documentation extracts"; "v1" (default) is
        byte-identical to before. Cite mode only: quote mode is v1 only."""
        if reader_prompt not in ("v1", "v2"):
            raise ValueError(f"unknown reader_prompt {reader_prompt!r}")
        if reader_prompt == "v2" and mode == "quote":
            raise ValueError("reader_prompt v2 does not apply to quote mode")
        if mode == "quote":
            g = grammar.quoted_answer(len(chunks))
            return self.chat(build_prompt(question, chunks, system=QUOTE_SYSTEM), grammar=g, **kw)
        g = grammar.cited_answer(len(chunks)) if cite_grammar and chunks else None
        if reader_prompt == "v2":
            msgs = build_prompt(question, chunks, system=SYSTEM_V2, header="Documentation extracts")
        else:
            msgs = build_prompt(question, chunks)
        return self.chat(msgs, grammar=g, **kw)

    def rewrites(self, question: str, n: int = 1, max_tokens: int = 120,
                 style: str = "man") -> list[str]:
        """N grammar-constrained alternative search queries for `question`, one per
        line, no numbering. Grammar-constrained rather than filtered afterwards on
        purpose: the local 4B is sloppy at this (one run produced a 300-token blob
        of ORs) and rank fusion is what makes the fused-retrieval path robust to a
        bad variant, not a quality check here - see retrieve.fuse_variants.

        `style` is passed straight through to build_rewrite_prompt; "man"
        (default) reproduces today's behaviour exactly, "docs" is phase 11 R4b.
        """
        if n <= 0:
            return []
        text = self.chat(build_rewrite_prompt(question, n, style=style), max_tokens=max_tokens,
                         temperature=0.0, grammar=_rewrite_grammar(n))
        return [l.strip() for l in text.splitlines() if l.strip()][:n]

    def correct(self, question: str, max_tokens: int = 120) -> str:
        """Phase 11 R4a': ask the model to fix spelling in `question`, holding
        every program/package/command/option/file name fixed (build_correct_prompt).
        Grammar-constrained to a single non-empty line, same idiom as rewrites().

        Guard against the model answering the question instead of correcting it
        (blk_r4a_spell_normalisation_fails_rule's failure mode, generalised): an
        empty output, or one more than 2x the input's length plus 20 chars, is
        treated as a bad correction and `question` is returned unchanged."""
        text = self.chat(build_correct_prompt(question), max_tokens=max_tokens,
                         temperature=0.0, grammar=_rewrite_grammar(1)).strip()
        if not text or len(text) > 2 * len(question) + 20:
            return question
        return text

    def judge(self, claim: str, extract: str) -> float:
        """Does `extract` explicitly support `claim`? Grammar-constrained to
        exactly `yes` or `no` at temperature 0, so the verifier's judge
        mechanism (smm.verify.JudgeScorer) gets a clean 1.0/0.0 every time."""
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"Extract:\n{extract}\n\nClaim:\n{claim}"},
        ]
        text = self.chat(messages, max_tokens=5, temperature=0.0, grammar=JUDGE_GRAMMAR)
        return 1.0 if text.strip().lower() == "yes" else 0.0

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False
