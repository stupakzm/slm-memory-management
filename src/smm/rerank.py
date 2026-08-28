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
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8082"


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

    def scores(self, query: str, documents: list[str]) -> list[float]:
        """Relevance score per document, in the order given."""
        if not documents:
            return []
        out = self._post("/v1/rerank", {"model": "reranker", "query": query,
                                        "documents": documents})
        got = [0.0] * len(documents)
        for r in out["results"]:
            got[r["index"]] = r["relevance_score"]
        return got

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
