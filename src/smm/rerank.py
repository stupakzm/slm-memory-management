"""Cross-encoder reranking against a local llama-server in `--reranking` mode.

The briefing calls the reranker non-optional, and phase 1 says why in two numbers:
59% of retrieval misses landed on a plausible wrong document, and 41% found the
right document but buried the answer at rank 11-17. A bi-encoder scores query and
chunk apart and cannot tell those cases from a hit; a cross-encoder reads both
together.

It also produces the score the abstention gate actually needs. Phase 1's raw dense
score put 37.5% of unanswerable questions above the answerable 10th percentile -
the two populations overlap too much to threshold. A reranker score is what
Finding 04 wants at the gate instead.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8082"

# Tokens reserved for the rerank template (query/document special tokens,
# separators) on top of the raw query+document token counts, when fitting a
# truncated document under the server's physical batch size.
TRUNCATE_MARGIN = 32

_TOO_LARGE_RE = re.compile(
    r"is too large to process.*current batch size:\s*(\d+)", re.DOTALL
)


class Reranker:
    def __init__(self, url: str = DEFAULT_URL, timeout: int = 300):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def _rerank_once(self, query: str, documents: list[str]) -> list[float]:
        """A single /v1/rerank request, no fallback. May raise HTTPError."""
        out = self._post("/v1/rerank", {"model": "reranker", "query": query,
                                        "documents": documents})
        got = [0.0] * len(documents)
        for r in out["results"]:
            got[r["index"]] = r["relevance_score"]
        return got

    @staticmethod
    def _too_large_batch_limit(e: urllib.error.HTTPError) -> int | None:
        """If `e` is the llama-server "input is too large to process /
        current batch size: B" 500, return B. Otherwise None (caller should
        re-raise: any other HTTP error propagates as before)."""
        if e.code != 500:
            return None
        try:
            body = json.loads(e.read())
        except Exception:
            return None
        message = body.get("error", {}).get("message", "")
        m = _TOO_LARGE_RE.search(message)
        return int(m.group(1)) if m else None

    def _tokenize(self, text: str) -> list:
        return self._post("/tokenize", {"content": text})["tokens"]

    def _detokenize(self, tokens: list) -> str:
        return self._post("/detokenize", {"tokens": tokens})["content"]

    def _truncated_score(self, query: str, document: str, limit: int) -> float | None:
        """Truncate `document` to fit under `limit` tokens and score it.
        Halves the kept token count on repeated too-large failures, up to 3
        times (4 attempts total). Returns None if it never fits."""
        q_tokens = self._tokenize(query)
        d_tokens = self._tokenize(document)
        keep = max(limit - len(q_tokens) - TRUNCATE_MARGIN, 0)
        for _ in range(4):
            if keep <= 0:
                return None
            truncated = self._detokenize(d_tokens[:keep])
            try:
                return self._rerank_once(query, [truncated])[0]
            except urllib.error.HTTPError as e:
                limit2 = self._too_large_batch_limit(e)
                if limit2 is None:
                    raise
                keep //= 2
        return None

    def _scores_with_fallback(self, query: str, documents: list[str],
                               limit: int) -> list[float]:
        """One document at a time (step 2), truncating whichever document
        still doesn't fit (step 3). A document that never fits gets the
        minimum score seen elsewhere in this batch minus 1.0 (or 0.0 if no
        document in the batch scored at all), so ordering stays total and
        finite without inventing a fake extreme like -inf."""
        scores: list[float | None] = []
        for document in documents:
            try:
                scores.append(self._rerank_once(query, [document])[0])
            except urllib.error.HTTPError as e:
                limit2 = self._too_large_batch_limit(e)
                if limit2 is None:
                    raise
                scores.append(self._truncated_score(query, document, limit2))
        fitted = [s for s in scores if s is not None]
        fallback = (min(fitted) - 1.0) if fitted else 0.0
        return [fallback if s is None else s for s in scores]

    def scores(self, query: str, documents: list[str]) -> list[float]:
        """Relevance score per document, in the order given.

        Normal path: one /v1/rerank request for the whole batch, exactly as
        before. If the query-time server's physical batch size rejects the
        batch (500, "is too large to process" / "current batch size: B"),
        fall back to scoring documents one at a time, truncating (and, if
        needed, further shrinking) whichever single document still doesn't
        fit. Any other HTTP error propagates unchanged.
        """
        if not documents:
            return []
        try:
            return self._rerank_once(query, documents)
        except urllib.error.HTTPError as e:
            limit = self._too_large_batch_limit(e)
            if limit is None:
                raise
            return self._scores_with_fallback(query, documents, limit)

    def rerank(self, query: str, chunks: list[dict], top_k: int | None = None,
               batch: int = 16) -> list[dict]:
        """Rescore chunks and return them best-first, each carrying `rerank_score`.

        Batched because the reranker context is 4096 tokens and a 50-candidate
        request would otherwise be one very large prompt.
        """
        if not chunks:
            return []
        docs = [f"{c.get('prefix','')}{c['text']}" for c in chunks]
        scored = []
        for i in range(0, len(docs), batch):
            scored.extend(self.scores(query, docs[i:i + batch]))
        out = [dict(c, rerank_score=s) for c, s in zip(chunks, scored)]
        out.sort(key=lambda c: -c["rerank_score"])
        return out[:top_k] if top_k else out

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False
