"""Embedding client for a local llama-server running Qwen3-Embedding-0.6B.

Qwen3-Embedding is asymmetric: documents are embedded raw, queries are wrapped in
an instruction. Skipping that wrapper measurably costs retrieval quality, so it is
part of the client rather than left to callers.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8081"
TASK = "Given a question about using a Linux system, retrieve the manual page passage that answers it"


def query_text(q: str, task: str = TASK) -> str:
    return f"Instruct: {task}\nQuery: {q}"


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


class Embedder:
    def __init__(self, url: str = DEFAULT_URL, timeout: int = 600):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._dim: int | None = None

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = self._post("/v1/embeddings", {"input": texts, "model": "embedder"})
        vecs = [_normalize(d["embedding"]) for d in sorted(out["data"], key=lambda d: d["index"])]
        if vecs and self._dim is None:
            self._dim = len(vecs[0])
        return vecs

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_query(self, q: str, task: str = TASK) -> list[float]:
        return self.embed([query_text(q, task)])[0]

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.embed(["dimension probe"])
        return self._dim

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False
