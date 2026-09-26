"""Tests for the selectable reranker model (tsk_20260926_213d599f):
`scripts/fetch_models.py`'s opt-in reranker-4b role and sha256 verification,
and `scripts/servers.sh`'s SMM_RERANK_MODEL override / missing-file failure.

Hermetic by design (blk_test_env_constraints): no network, no real servers.
Every test that starts a "server" reuses the stub-llama-server idea from
tests/test_cli.py (a stdlib http.server standing in for llama-server) on
scratch ports, a scratch SMM_RUN, and a scratch SMM_MODELS directory, so this
never looks at, starts, or kills a real llama-server this machine may already
have running on 8081/8082 - see SMM_RUN/SMM_*_PORT/SMM_MODELS/SMM_RERANK_MODEL
in scripts/servers.sh. The stub additionally logs its argv (via
STUB_ARGV_LOG) so a test can see which -m path it was launched with.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import socket
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVERS_SH = ROOT / "scripts" / "servers.sh"

_spec = importlib.util.spec_from_file_location("fetch_models", ROOT / "scripts" / "fetch_models.py")
fetch_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_models)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ensure_stub_llama_server(stub_dir: Path) -> Path:
    """A fake `llama-server`: binds --port, answers 200 to any GET, and (unlike
    tests/test_cli.py's stub) logs its full argv to $STUB_ARGV_LOG so a test
    can check which -m path it was launched with."""
    stub_dir.mkdir(parents=True, exist_ok=True)
    exe = stub_dir / "llama-server"
    exe.write_text("""#!/usr/bin/env python3
import http.server
import os
import socketserver
import sys

argv = sys.argv[1:]
port = 0
for i, a in enumerate(argv):
    if a == "--port" and i + 1 < len(argv):
        port = int(argv[i + 1])

log = os.environ.get("STUB_ARGV_LOG")
if log:
    with open(log, "a") as f:
        f.write(" ".join(argv) + "\\n")

class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", port), Health) as httpd:
    httpd.serve_forever()
""")
    exe.chmod(0o755)
    return exe


def base_env(tmp: Path, stub: Path) -> dict:
    run_dir = tmp / "run"
    run_dir.mkdir(exist_ok=True)
    env = dict(os.environ)
    env["SMM_RUN"] = str(run_dir)
    env["LLAMA_CPP"] = str(stub.parent)
    env["SMM_EMBED_PORT"] = str(free_port())
    env["SMM_RERANK_PORT"] = str(free_port())
    env["SMM_GEN_PORT"] = str(free_port())
    env.pop("SMM_RERANK_MODEL", None)
    return env


def stop(env: dict):
    subprocess.run(["bash", str(SERVERS_SH), "stop"], env=env, capture_output=True)


# --------------------------------------------------------------------------


def test_default_roles_unchanged():
    check(fetch_models.DEFAULT_ROLES == ("embedder", "generator", "reranker"),
          f"default roles must be exactly embedder, generator, reranker: {fetch_models.DEFAULT_ROLES}")
    check("reranker-4b" in fetch_models.MODELS, "reranker-4b role should exist")
    repo, filename, sha = fetch_models.MODELS["reranker-4b"]
    check(bool(repo) and bool(filename), f"reranker-4b entry incomplete: {fetch_models.MODELS['reranker-4b']}")
    check(bool(sha) and re.fullmatch(r"[0-9a-f]{64}", sha), f"reranker-4b sha256 should be 64 hex chars: {sha!r}")


def test_verify_sha256():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "f.bin"
        data = b"hello world"
        p.write_bytes(data)
        real = hashlib.sha256(data).hexdigest()

        fetch_models.verify_sha256(p, real)  # must not raise
        check(p.exists(), "file should still exist after a passing verify")

        raised = False
        try:
            fetch_models.verify_sha256(p, "0" * 64)
        except SystemExit:
            raised = True
        check(raised, "verify_sha256 must raise SystemExit on a hash mismatch")
        check(p.exists(), "file must still exist after a failing verify (never deleted)")


def test_servers_default_reranker_model():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        model_name = "qwen3-reranker-0.6b-q8_0.gguf"
        (models_dir / model_name).write_bytes(b"stub-weights")
        stub = ensure_stub_llama_server(tmp / "stub-bin")
        argv_log = tmp / "argv.log"

        env = base_env(tmp, stub)
        env["SMM_MODELS"] = str(models_dir)
        env["STUB_ARGV_LOG"] = str(argv_log)

        try:
            proc = subprocess.run(["bash", str(SERVERS_SH), "start", "reranker"],
                                   env=env, capture_output=True, text=True, timeout=30)
            check(proc.returncode == 0, f"start reranker should succeed: {proc.stderr}")
            check(argv_log.exists(), "stub should have logged its argv")
            logged = argv_log.read_text()
            expected = f"-m {models_dir}/{model_name}"
            check(expected in logged, f"expected {expected!r} in logged argv {logged!r}")
        finally:
            stop(env)


def test_servers_rerank_model_override():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        model_name = "x.gguf"
        (models_dir / model_name).write_bytes(b"stub-weights")
        stub = ensure_stub_llama_server(tmp / "stub-bin")
        argv_log = tmp / "argv.log"

        env = base_env(tmp, stub)
        env["SMM_MODELS"] = str(models_dir)
        env["STUB_ARGV_LOG"] = str(argv_log)
        env["SMM_RERANK_MODEL"] = model_name

        try:
            proc = subprocess.run(["bash", str(SERVERS_SH), "start", "reranker"],
                                   env=env, capture_output=True, text=True, timeout=30)
            check(proc.returncode == 0, f"start reranker with override should succeed: {proc.stderr}")
            logged = argv_log.read_text() if argv_log.exists() else ""
            expected = f"-m {models_dir}/{model_name}"
            check(expected in logged, f"expected {expected!r} in logged argv {logged!r}")
        finally:
            stop(env)

        # Now the override points at a file that does not exist: must fail
        # non-zero and never launch the stub.
        argv_log.write_text("")
        env["SMM_RERANK_PORT"] = str(free_port())
        env["SMM_RERANK_MODEL"] = "missing-model.gguf"
        try:
            proc2 = subprocess.run(["bash", str(SERVERS_SH), "start", "reranker"],
                                    env=env, capture_output=True, text=True, timeout=30)
            check(proc2.returncode != 0, f"start with a missing model must fail: rc={proc2.returncode} out={proc2.stdout}")
            check(not argv_log.read_text().strip(),
                  f"the stub must never have been launched for a missing model: {argv_log.read_text()!r}")
            check(not (Path(env["SMM_RUN"]) / "reranker.pid").exists(),
                  "no pidfile should exist after a failed start")
        finally:
            stop(env)


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
