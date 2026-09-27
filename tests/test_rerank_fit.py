"""Tests for tsk_20260927_rerankfit: Reranker.scores() falling back when a
batch (or a single document) is too large for the query-time llama-server
profile (scripts/servers.sh `reranker-query`: -c 1024, -b/-ub 768).

Hermetic by design (blk_test_env_constraints): no pytest, no real servers -
a stdlib http.server stands in for llama-server on a free loopback port,
implementing /v1/rerank, /tokenize and /detokenize with a deterministic,
whitespace-word token model ("tokens = whitespace-split words is fine"),
in the stub-server style of tests/test_cli.py and tests/test_rerank_select.py.
/v1/rerank returns the exact llama-server 500 body when a query+document pair
exceeds a configured batch-size limit B:
  {"error":{"code":500,"message":"input (N tokens) is too large to process.
  increase the physical batch size (current batch size: B)","type":"server_error"}}
"""

from __future__ import annotations

import http.server
import json
import sys
import threading
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm.rerank import Reranker  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def score_of(text: str) -> float:
    """Deterministic score, a function of the text alone (so a stub-recomputed
    expectation and the server's actual response always agree)."""
    return round(len(text) / 7.0, 4)


class RerankStub(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw)

    def _send_json(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _too_large(self, total: int):
        cfg = self.server.cfg
        if cfg.get("bad_message"):
            message = cfg["bad_message"]
        else:
            message = (f"input ({total} tokens) is too large to process. "
                       f"increase the physical batch size (current batch size: {cfg['B']})")
        self._send_json(500, {"error": {"code": 500, "message": message,
                                         "type": "server_error"}})

    def do_POST(self):
        payload = self._read_json()
        cfg = self.server.cfg
        self.server.log.append((self.path, payload))

        if self.path == "/v1/rerank":
            if cfg.get("always_bad_500"):
                self._too_large(0)
                return
            query = payload["query"]
            documents = payload["documents"]
            qn = len(query.split())
            B = cfg["B"]
            for doc in documents:
                total = qn + len(doc.split())
                if total > B:
                    self._too_large(total)
                    return
            self.server.accepted.append(qn + max((len(d.split()) for d in documents), default=0))
            results = [{"index": i, "relevance_score": score_of(d)}
                       for i, d in enumerate(documents)]
            self._send_json(200, {"results": results})
        elif self.path == "/tokenize":
            self._send_json(200, {"tokens": payload["content"].split()})
        elif self.path == "/detokenize":
            self._send_json(200, {"content": " ".join(payload["tokens"])})
        else:
            self._send_json(404, {"error": "not found"})

    def log_message(self, *a):
        pass


def start_stub(B: int = 1_000_000, bad_message: str | None = None,
               always_bad_500: bool = False):
    server = http.server.HTTPServer(("127.0.0.1", 0), RerankStub)
    server.cfg = {"B": B, "bad_message": bad_message, "always_bad_500": always_bad_500}
    server.log = []
    server.accepted = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    return server, thread, url


def stop_stub(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


# --------------------------------------------------------------------------


def test_fitting_batch_is_one_request_same_scores():
    server, thread, url = start_stub(B=1_000_000)
    try:
        r = Reranker(url=url)
        docs = ["short document one", "a slightly longer document here", "doc three"]
        got = r.scores("what is this", docs)
        rerank_requests = [p for path, p in server.log if path == "/v1/rerank"]
        check(len(rerank_requests) == 1, f"expected exactly one /v1/rerank request, got {len(rerank_requests)}")
        check(got == [score_of(d) for d in docs], f"scores changed for a fitting batch: {got}")
    finally:
        stop_stub(server, thread)


def test_one_oversized_document_falls_back_others_unchanged():
    # query is 2 words; B=10 so anything with doc word-count > 8 is too large.
    B = 10
    server, thread, url = start_stub(B=B)
    try:
        r = Reranker(url=url)
        normal_a = "short doc a"
        oversized = words(50)  # 50 words: 2 + 50 = 52 >> B, forces fallback + truncation
        normal_b = "short doc b here"
        docs = [normal_a, oversized, normal_b]
        got = r.scores("q1 q2", docs)

        check(len(got) == 3, f"result length must match input length: {got}")
        check(got[0] == score_of(normal_a), f"normal doc 0 score changed: {got[0]}")
        check(got[2] == score_of(normal_b), f"normal doc 2 score changed: {got[2]}")
        check(isinstance(got[1], float), f"oversized doc must still get a float score: {got[1]!r}")

        rerank_requests = [p for path, p in server.log if path == "/v1/rerank"]
        # 1 failed full-batch request + 3 one-document retries (one truncated)
        check(len(rerank_requests) >= 4,
              f"expected the full-batch attempt plus one request per document, got {len(rerank_requests)}")
        first = rerank_requests[0]
        check(first["documents"] == docs, "first request should be the untouched full batch")
    finally:
        stop_stub(server, thread)


def test_truncation_respects_batch_limit():
    B = 100
    server, thread, url = start_stub(B=B)
    try:
        r = Reranker(url=url)
        query = "what is this about"  # 4 words
        oversized = words(5000)
        got = r.scores(query, [oversized])
        check(len(got) == 1 and isinstance(got[0], float), f"expected one float score: {got}")
        check(server.accepted, "no request was ever accepted by the stub")
        check(all(n <= B for n in server.accepted),
              f"a request over the stub's limit B={B} was accepted: {server.accepted}")
    finally:
        stop_stub(server, thread)


def test_other_500_propagates():
    server, thread, url = start_stub(always_bad_500=True, bad_message="internal server error, no batch here")
    try:
        r = Reranker(url=url)
        raised = False
        try:
            r.scores("q", ["doc one", "doc two"])
        except urllib.error.HTTPError as e:
            raised = True
            check(e.code == 500, f"expected the stub's 500 to propagate, got {e.code}")
        check(raised, "a non-'too large' 500 must propagate as an HTTPError")
    finally:
        stop_stub(server, thread)


def test_rerank_end_to_end_mixed_batch():
    B = 20
    server, thread, url = start_stub(B=B)
    try:
        r = Reranker(url=url)
        chunks = [
            {"id": "a", "text": "short chunk a"},
            {"id": "b", "text": words(200)},  # oversized, forces fallback path
            {"id": "c", "text": "short chunk c is here"},
        ]
        out = r.rerank("q1 q2", chunks, batch=16)

        check(len(out) == len(chunks), f"rerank must preserve length: {len(out)} vs {len(chunks)}")
        check(all("rerank_score" in c for c in out), "every chunk must carry rerank_score")
        scores = [c["rerank_score"] for c in out]
        check(scores == sorted(scores, reverse=True), f"rerank() must return best-first: {scores}")
        ids = {c["id"] for c in out}
        check(ids == {"a", "b", "c"}, f"all original chunks must be present: {ids}")
    finally:
        stop_stub(server, thread)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  pass  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001 - a crashing test is still a failure to report
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
