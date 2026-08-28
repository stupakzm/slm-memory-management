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


def build_rewrite_prompt(question: str, n: int) -> list[dict]:
    system = (f"Rewrite the user's request as {n} alternative search queries for a "
              "Linux manual-page search engine. Each on its own line, no numbering, "
              "no explanation. Keep them short, name the likely command if you can, "
              "and vary the wording.")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def build_prompt(question: str, chunks: list[dict]) -> list[dict]:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] {c['doc_id']}\n{c['prefix']}{c['text']}")
    context = "\n\n".join(parts) if parts else "(no extracts found)"
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Manual page extracts:\n\n{context}\n\nQuestion: {question}"},
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

    def answer(self, question: str, chunks: list[dict], cite_grammar: bool = False, **kw) -> str:
        """`cite_grammar` constrains decoding so every claim carries an in-range
        citation - see smm.grammar for what that does and does not guarantee."""
        g = grammar.cited_answer(len(chunks)) if cite_grammar and chunks else None
        return self.chat(build_prompt(question, chunks), grammar=g, **kw)

    def rewrites(self, question: str, n: int = 1, max_tokens: int = 120) -> list[str]:
        """N grammar-constrained alternative search queries for `question`, one per
        line, no numbering. Grammar-constrained rather than filtered afterwards on
        purpose: the local 4B is sloppy at this (one run produced a 300-token blob
        of ORs) and rank fusion is what makes the fused-retrieval path robust to a
        bad variant, not a quality check here - see retrieve.fuse_variants.
        """
        if n <= 0:
            return []
        text = self.chat(build_rewrite_prompt(question, n), max_tokens=max_tokens,
                         temperature=0.0, grammar=_rewrite_grammar(n))
        return [l.strip() for l in text.splitlines() if l.strip()][:n]

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False
