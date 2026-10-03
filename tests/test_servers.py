"""Tests for scripts/servers.sh: status honours SMM_*_PORT, and wait_ready
fails fast on a dead server but still waits for a slow one.

Everything runs against stub llama-servers on scratch ports with a scratch
SMM_RUN, never the real ports, pidfiles or servers.
"""

import http.server
import os
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVERS = ROOT / "scripts" / "servers.sh"

STUB = """#!/usr/bin/env python3
import http.server, socketserver, sys, time
port = 0
argv = sys.argv[1:]
for i, a in enumerate(argv):
    if a == "--port" and i + 1 < len(argv):
        port = int(argv[i + 1])
%s
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *a):
        pass
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", port), H) as httpd:
    httpd.serve_forever()
"""


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_env(tmp: Path, prelude: str) -> dict:
    bindir = tmp / "bin"
    bindir.mkdir()
    exe = bindir / "llama-server"
    exe.write_text(STUB % prelude)
    exe.chmod(0o755)
    env = dict(os.environ)
    env["SMM_RUN"] = str(tmp / "run")
    env["LLAMA_CPP"] = str(bindir)
    env["SMM_EMBED_PORT"] = str(free_port())
    env["SMM_RERANK_PORT"] = str(free_port())
    env["SMM_GEN_PORT"] = str(free_port())
    return env


def servers(env, *args, timeout=60):
    return subprocess.run(["bash", str(SERVERS), *args], env=env,
                          capture_output=True, text=True, timeout=timeout)


def test_status_uses_configured_ports():
    with tempfile.TemporaryDirectory() as d:
        env = make_env(Path(d), "")
        port = int(env["SMM_GEN_PORT"])

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok")

            def log_message(self, *a):
                pass

        httpd = socketserver.TCPServer(("127.0.0.1", port), H)
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            out = servers(env, "status").stdout
        finally:
            httpd.shutdown()
            httpd.server_close()
        check(f"generator  UP    :{port}" in out, f"generator should be UP on :{port}: {out!r}")
        check(f":{env['SMM_EMBED_PORT']}" in out and f":{env['SMM_RERANK_PORT']}" in out,
              f"embedder/reranker ports should be reported: {out!r}")
        check(":8080" not in out, f"status must not report the hard-coded 8080: {out!r}")


def test_wait_ready_fails_fast_on_dead_server():
    with tempfile.TemporaryDirectory() as d:
        env = make_env(Path(d), "sys.exit(1)")
        t0 = time.time()
        try:
            r = servers(env, "start", "generator", timeout=60)
        finally:
            servers(env, "stop")
        elapsed = time.time() - t0
        check(r.returncode != 0, f"start should fail for a dead server (rc={r.returncode})")
        check(elapsed < 10, f"should fail fast, took {elapsed:.1f}s")
        check("did not come up" in r.stderr, f"missing message: {r.stderr!r}")


def test_wait_ready_still_waits_for_slow_start():
    with tempfile.TemporaryDirectory() as d:
        env = make_env(Path(d), "time.sleep(3)")
        try:
            r = servers(env, "start", "generator", timeout=60)
            check(r.returncode == 0, f"slow start should succeed: rc={r.returncode} {r.stderr!r}")
            check("generator ready" in r.stdout, f"stdout: {r.stdout!r}")
        finally:
            servers(env, "stop")


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)
