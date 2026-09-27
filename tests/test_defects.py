"""Tests for the three tsk_20260927_defects fixes:

  (a) src/smm/daemon.py's start() must return False immediately when
      scripts/servers.sh exits non-zero, instead of polling /health for the
      full `wait` window regardless.
  (b) scripts/eval_verifier.py's "answered_with_no_evidence" denominator must
      be the no-evidence subset (answerable records whose evidence_retrieved
      is false - the same subset scripts/eval_answers.py:301-302 uses), not
      every answerable record.
  (c) scripts/servers.sh's start_generator must support SMM_GEN_MODEL (which
      weight to load) and SMM_GEN_ARGS (extra llama-server args, appended),
      mirroring SMM_RERANK_MODEL exactly, while leaving the unset-both argv
      byte-identical to before.

Hermetic by design (blk_test_env_constraints): stdlib only, no pytest, no
network, no real servers. Daemon tests import src/smm/daemon.py directly (it
is not part of the smm package - see its own module docstring) and monkeypatch
its module attributes (RUN, SERVERS_SH, PORT_OF) to scratch values, the same
idea tests/test_cli.py uses via subprocess. servers.sh tests reuse the
stub-llama-server-that-logs-its-argv idea from tests/test_rerank_select.py, on
scratch ports, a scratch SMM_RUN, and a scratch SMM_MODELS - never the real
ports 8080-8083 or the real .run/.
"""

from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAEMON_PATH = ROOT / "src" / "smm" / "daemon.py"
EVAL_VERIFIER_PATH = ROOT / "scripts" / "eval_verifier.py"
SERVERS_SH = ROOT / "scripts" / "servers.sh"


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def write_exec(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(0o755)
    return path


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_daemon():
    """A fresh import each call, so each test monkeypatches its own module
    object (RUN/SERVERS_SH/PORT_OF) without leaking into another test."""
    return load_module("daemon_under_test", DAEMON_PATH)


# --------------------------------------------------------------------------
# (a) daemon.start() must not poll the full `wait` window on a hard failure.
# --------------------------------------------------------------------------


def test_daemon_start_fails_fast_on_nonzero_servers_sh():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        stub = write_exec(tmp / "stub-fail.sh", "#!/usr/bin/env bash\nexit 1\n")
        daemon = load_daemon()
        daemon.RUN = tmp / "run"
        daemon.SERVERS_SH = stub
        daemon.PORT_OF["embedder"] = free_port()  # never actually served

        t0 = time.time()
        ok = daemon.start("embedder", wait=90.0)
        elapsed = time.time() - t0

        check(ok is False, "start() must return False when servers.sh exits non-zero")
        check(elapsed < 5.0,
              f"start() must fail fast on a non-zero exit, not poll for the full wait: {elapsed:.1f}s")


FAKE_HEALTH_SERVER = """#!/usr/bin/env python3
import http.server
import socketserver
import sys

port = int(sys.argv[1])


class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", port), Health) as httpd:
    httpd.serve_forever()
"""


def test_daemon_start_succeeds_when_stub_exits_zero_and_health_answers():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        port = free_port()
        health_srv = write_exec(tmp / "fake_health_server.py", FAKE_HEALTH_SERVER)
        stub = write_exec(tmp / "stub-ok.sh", f"""#!/usr/bin/env bash
{sys.executable} {health_srv} {port} > {tmp}/health.log 2>&1 &
echo $! > {tmp}/health.pid
disown
exit 0
""")
        daemon = load_daemon()
        daemon.RUN = tmp / "run"
        daemon.SERVERS_SH = stub
        daemon.PORT_OF["embedder"] = port

        try:
            ok = daemon.start("embedder", wait=5.0)
            check(ok is True,
                  "start() must return True once servers.sh exits 0 and /health answers")
        finally:
            pidfile = tmp / "health.pid"
            if pidfile.exists():
                try:
                    os.kill(int(pidfile.read_text().strip()), 15)
                except (OSError, ValueError):
                    pass


# --------------------------------------------------------------------------
# (b) the no-evidence denominator is the no-evidence subset, not every
#     answerable record.
# --------------------------------------------------------------------------


def test_no_evidence_denominator_is_the_no_evidence_subset():
    ev = load_module("eval_verifier_under_test", EVAL_VERIFIER_PATH)

    ans_recs = [
        {"qid": "a1", "evidence_retrieved": True},
        {"qid": "a2", "evidence_retrieved": False},
        {"qid": "a3", "evidence_retrieved": False},
        {"qid": "a4", "evidence_retrieved": False},
    ]
    # a3 is removed/abstained after replay; a2 and a4 are still answered.
    removed = {"a3"}

    def still_answered(qid: str) -> bool:
        return qid not in removed

    answered, total = ev.no_evidence_answered(ans_recs, still_answered)

    check(total == 3,
          f"denominator must be the no-evidence subset (a2,a3,a4 = 3), got {total}")
    check(total != len(ans_recs),
          "denominator must not be every answerable record (the old bug)")
    check(answered == 2, f"numerator must count still-answered no-evidence records, got {answered}")


# --------------------------------------------------------------------------
# (c) scripts/servers.sh start_generator: SMM_GEN_MODEL / SMM_GEN_ARGS.
# --------------------------------------------------------------------------


def ensure_stub_llama_server(stub_dir: Path) -> Path:
    """A fake `llama-server`: binds --port, answers 200 to any GET, and logs
    its full argv to $STUB_ARGV_LOG (mirrors tests/test_rerank_select.py's
    ensure_stub_llama_server, duplicated here to keep this file hermetic and
    self-contained rather than importing another test module)."""
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
    env.pop("SMM_GEN_MODEL", None)
    env.pop("SMM_GEN_ARGS", None)
    return env


def stop(env: dict):
    subprocess.run(["bash", str(SERVERS_SH), "stop"], env=env, capture_output=True)


def test_servers_gen_model_override():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        model_name = "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
        (models_dir / model_name).write_bytes(b"stub-weights")
        stub = ensure_stub_llama_server(tmp / "stub-bin")
        argv_log = tmp / "argv.log"

        env = base_env(tmp, stub)
        env["SMM_MODELS"] = str(models_dir)
        env["STUB_ARGV_LOG"] = str(argv_log)
        env["SMM_GEN_MODEL"] = model_name

        try:
            proc = subprocess.run(["bash", str(SERVERS_SH), "start", "generator"],
                                   env=env, capture_output=True, text=True, timeout=30)
            check(proc.returncode == 0, f"start generator with override should succeed: {proc.stderr}")
            logged = argv_log.read_text() if argv_log.exists() else ""
            expected = f"-m {models_dir}/{model_name}"
            check(expected in logged, f"expected {expected!r} in logged argv {logged!r}")
        finally:
            stop(env)


def test_servers_gen_args_appended():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        model_name = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
        (models_dir / model_name).write_bytes(b"stub-weights")
        stub = ensure_stub_llama_server(tmp / "stub-bin")
        argv_log = tmp / "argv.log"

        env = base_env(tmp, stub)
        env["SMM_MODELS"] = str(models_dir)
        env["STUB_ARGV_LOG"] = str(argv_log)
        env["SMM_GEN_ARGS"] = "--cpu-moe -t 12"

        try:
            proc = subprocess.run(["bash", str(SERVERS_SH), "start", "generator"],
                                   env=env, capture_output=True, text=True, timeout=30)
            check(proc.returncode == 0, f"start generator with SMM_GEN_ARGS should succeed: {proc.stderr}")
            logged = argv_log.read_text() if argv_log.exists() else ""
            check(logged.strip().endswith("--cpu-moe -t 12"),
                  f"SMM_GEN_ARGS should be appended (word-split) to the argv: {logged!r}")
        finally:
            stop(env)


def test_servers_gen_model_missing_file_fails_without_launching():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        stub = ensure_stub_llama_server(tmp / "stub-bin")
        argv_log = tmp / "argv.log"

        env = base_env(tmp, stub)
        env["SMM_MODELS"] = str(models_dir)
        env["STUB_ARGV_LOG"] = str(argv_log)
        env["SMM_GEN_MODEL"] = "missing-model.gguf"

        try:
            proc = subprocess.run(["bash", str(SERVERS_SH), "start", "generator"],
                                   env=env, capture_output=True, text=True, timeout=30)
            check(proc.returncode != 0,
                  f"start with a missing SMM_GEN_MODEL must fail: rc={proc.returncode} out={proc.stdout}")
            check(not argv_log.read_text().strip() if argv_log.exists() else True,
                  f"the stub must never have been launched for a missing model: "
                  f"{argv_log.read_text() if argv_log.exists() else ''!r}")
            check(not (Path(env["SMM_RUN"]) / "generator.pid").exists(),
                  "no pidfile should exist after a failed start")
        finally:
            stop(env)


def test_servers_gen_unset_argv_byte_identical_to_before():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        models_dir = tmp / "models"
        models_dir.mkdir()
        model_name = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
        (models_dir / model_name).write_bytes(b"stub-weights")
        stub = ensure_stub_llama_server(tmp / "stub-bin")
        argv_log = tmp / "argv.log"

        env = base_env(tmp, stub)
        env["SMM_MODELS"] = str(models_dir)
        env["STUB_ARGV_LOG"] = str(argv_log)
        # SMM_GEN_MODEL / SMM_GEN_ARGS both unset (base_env already pops them).

        try:
            proc = subprocess.run(["bash", str(SERVERS_SH), "start", "generator"],
                                   env=env, capture_output=True, text=True, timeout=30)
            check(proc.returncode == 0, f"start generator should succeed: {proc.stderr}")
            logged = argv_log.read_text().strip() if argv_log.exists() else ""
            expected = (f"-m {models_dir}/{model_name} -c 8192 -ngl 99 --temp 0.0 "
                        f"--host 127.0.0.1 --port {env['SMM_GEN_PORT']}")
            check(logged == expected,
                  f"argv must be byte-identical to before this change:\n  got:      {logged!r}\n  expected: {expected!r}")
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
