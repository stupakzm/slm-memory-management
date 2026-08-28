"""Tests for the `asq`/`smm` launcher: lazy server start, the idle reaper, and
the renameable-symlink mechanics - the parts a retrieval eval would never touch.

Every test that starts a "server" uses a stub (a stdlib http.server standing
in for llama-server) on scratch ports and a scratch .run directory, so this
never looks at, starts, or kills a real llama-server this machine may already
have running - see SMM_RUN/SMM_*_PORT in scripts/servers.sh and src/smm/daemon.py.
"""

import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN_SMM = ROOT / "bin" / "smm"
DAEMON = ROOT / "src" / "smm" / "daemon.py"
STUB_DIR = ROOT / ".run" / "test-stub-bin"  # scratch, gitignored, not scope.paths

# bin/smm insists on $ROOT/.venv/bin/python (never the ambient interpreter -
# that is the point of goal #1). This worktree has no .venv of its own (it's
# gitignored - see task env facts), so point it at whatever venv is actually
# running this test file, which for `.venv/bin/python tests/test_cli.py` is
# already the real one.
VENV_PY = Path(sys.executable).resolve()


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def ensure_repo_venv():
    """bin/smm resolves ROOT/.venv/bin/python relative to its own real path.
    This worktree doesn't ship a .venv (gitignored); point at the venv this
    test is actually running under so bin/smm subprocess calls work."""
    venv_link = ROOT / ".venv"
    if venv_link.exists():
        return
    real_venv_root = VENV_PY.parent.parent
    venv_link.symlink_to(real_venv_root)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def write_exec(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(0o755)
    return path


def ensure_stub_llama_server() -> Path:
    """A fake `llama-server`: binds --port, answers 200 to any GET. Ignores
    every other flag (including -m, whose .gguf never exists in this worktree)."""
    STUB_DIR.mkdir(parents=True, exist_ok=True)
    exe = STUB_DIR / "llama-server"
    if not exe.exists():
        write_exec(exe, """#!/usr/bin/env python3
import http.server
import socketserver
import sys

port = 0
argv = sys.argv[1:]
for i, a in enumerate(argv):
    if a == "--port" and i + 1 < len(argv):
        port = int(argv[i + 1])

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
    return exe


def write_fake_ask(tmp: Path) -> Path:
    """Stands in for scripts/ask.py: same ordered health checks, same exit-2
    "X not running" messages (scripts/ask.py:72,78,106), same --retrieve-only
    / --no-rerank short-circuits - so daemon.run_wrapped's retry logic is
    exercised exactly as it would be against the real script."""
    return write_exec(tmp / "fake_ask.py", """#!/usr/bin/env python3
import os, sys, urllib.request

def health(port):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
        return True
    except Exception:
        return False

def main():
    args = sys.argv[1:]
    emb, rer, gen = (int(os.environ[k]) for k in
                      ("SMM_EMBED_PORT", "SMM_RERANK_PORT", "SMM_GEN_PORT"))
    no_rerank = "--no-rerank" in args
    retrieve_only = "--retrieve-only" in args
    if not health(emb):
        print("embedder not running: ./scripts/servers.sh start embedder", file=sys.stderr)
        return 2
    if not no_rerank and not health(rer):
        print("reranker not running: ./scripts/servers.sh start reranker", file=sys.stderr)
        return 2
    if retrieve_only:
        print("[1] 0.9000 fake-chunk  fake retrieval hit")
        return 0
    if not health(gen):
        print("generator not running: ./scripts/servers.sh start generator", file=sys.stderr)
        return 2
    print("fake answer")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
""")


def write_fake_ok(tmp: Path) -> Path:
    """Needs nothing, always succeeds - used to exercise touch()/ensure_reaper()
    in isolation, without pulling a server into the reaper tests."""
    return write_exec(tmp / "fake_ok.py", "#!/usr/bin/env python3\nprint('ok')\n")


def isolated_env(tmp: Path, idle=None) -> dict:
    stub = ensure_stub_llama_server()
    env = dict(os.environ)
    env["SMM_RUN"] = str(tmp / "run")
    env["LLAMA_CPP"] = str(stub.parent)
    env["SMM_EMBED_PORT"] = str(free_port())
    env["SMM_RERANK_PORT"] = str(free_port())
    env["SMM_GEN_PORT"] = str(free_port())
    if idle is not None:
        env["SMM_IDLE"] = str(idle)
    Path(env["SMM_RUN"]).mkdir(parents=True, exist_ok=True)
    return env


def health(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
            return r.status == 200
    except Exception:
        return False


def daemon_run(env: dict, script: Path, args: list) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(VENV_PY), str(DAEMON), "run", str(VENV_PY), str(script), "--", *args],
        env=env, capture_output=True, text=True, timeout=60,
    )


def cleanup(env: dict):
    """Stop anything the test started and kill a still-alive reaper directly
    (belt and braces - reap_loop only self-removes its pidfile on the path
    where it actually reaped)."""
    subprocess.run(["bash", str(ROOT / "scripts" / "servers.sh"), "stop"],
                    env=env, capture_output=True)
    reaper_pid_file = Path(env["SMM_RUN"]) / "reaper.pid"
    if reaper_pid_file.exists():
        try:
            pid = int(reaper_pid_file.read_text().strip())
            os.kill(pid, 15)
        except (ValueError, OSError):
            pass


def wait_until(cond, timeout=15.0, interval=0.2):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return cond()


# --------------------------------------------------------------------------


def test_status_starts_nothing():
    proc = subprocess.run([str(BIN_SMM), "status"], cwd="/tmp",
                           capture_output=True, text=True, timeout=30)
    check(proc.returncode == 0, f"status should exit 0: {proc.stderr}")
    for name in ("embedder", "reranker", "generator"):
        check(name in proc.stdout, f"status should report {name}: {proc.stdout}")
    check("down" in proc.stdout or "UP" in proc.stdout, "status should say up/down")


def test_name_from_argv0():
    """Usage text reads back the name the tool was actually invoked as -
    renaming must never require a code edit (bin/smm derives it from $0)."""
    tmp = ROOT / ".run" / "test-argv0"
    tmp.mkdir(parents=True, exist_ok=True)
    link = tmp / "zsm"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(BIN_SMM.resolve())
    try:
        proc = subprocess.run([str(link)], capture_output=True, text=True, timeout=30)
        check(proc.returncode == 1, "bare invocation with no question should fail usage")
        check("zsm" in proc.stderr, f"usage should read back invoked name: {proc.stderr}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_install_default_name_and_path_warning():
    tmp = ROOT / ".run" / "test-install"
    tmp.mkdir(parents=True, exist_ok=True)
    prefix = tmp / "bin"
    try:
        env = dict(os.environ)
        env["PATH"] = "/usr/bin:/bin"  # guaranteed not to contain our scratch prefix
        proc = subprocess.run([str(BIN_SMM), "install", "--prefix", str(prefix)],
                               capture_output=True, text=True, env=env, timeout=30)
        check(proc.returncode == 0, f"install should exit 0: {proc.stderr}")
        installed = prefix / "asq"
        check(installed.is_symlink(), "install should default the link name to asq")
        check(Path(os.path.realpath(installed)) == BIN_SMM.resolve(),
              "asq should resolve back to bin/smm")
        check("not on" in proc.stderr and "PATH" in proc.stderr,
              f"install should warn when prefix is off PATH: {proc.stderr}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_works_from_any_cwd_via_symlink():
    ensure_repo_venv()
    tmp = ROOT / ".run" / "test-anycwd"
    tmp.mkdir(parents=True, exist_ok=True)
    prefix = tmp / "bin"
    try:
        install = subprocess.run([str(BIN_SMM), "install", "--name", "foo", "--prefix", str(prefix)],
                                  capture_output=True, text=True, timeout=30)
        check(install.returncode == 0, f"install failed: {install.stderr}")
        proc = subprocess.run([str(prefix / "foo"), "status"], cwd="/tmp",
                               capture_output=True, text=True, timeout=30)
        check(proc.returncode == 0,
              f"invoking the symlink from /tmp should still resolve ROOT: {proc.stderr}")
        check("embedder" in proc.stdout, f"should reach real daemon output: {proc.stdout}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_lifecycle_retrieve_only_skips_generator():
    """Goal #2, hard requirement: --retrieve-only must never start the generator."""
    tmp = ROOT / ".run" / "test-lifecycle-ro"
    tmp.mkdir(parents=True, exist_ok=True)
    env = isolated_env(tmp)
    try:
        fake_ask = write_fake_ask(tmp)
        proc = daemon_run(env, fake_ask, ["--retrieve-only", "how do I x"])
        check(proc.returncode == 0, f"retrieve-only run should succeed: {proc.stderr}")
        check(health(int(env["SMM_EMBED_PORT"])), "embedder should have been started")
        check(health(int(env["SMM_RERANK_PORT"])), "reranker should have been started")
        check(not health(int(env["SMM_GEN_PORT"])),
              "generator must NEVER start for --retrieve-only")
        check(not (Path(env["SMM_RUN"]) / "generator.pid").exists(),
              "no generator pidfile should exist for --retrieve-only")
    finally:
        cleanup(env)
        shutil.rmtree(tmp, ignore_errors=True)


def test_lifecycle_no_rerank_skips_reranker():
    """Goal #2, hard requirement: --no-rerank must never start the reranker."""
    tmp = ROOT / ".run" / "test-lifecycle-nr"
    tmp.mkdir(parents=True, exist_ok=True)
    env = isolated_env(tmp)
    try:
        fake_ask = write_fake_ask(tmp)
        proc = daemon_run(env, fake_ask, ["--no-rerank", "how do I x"])
        check(proc.returncode == 0, f"no-rerank run should succeed: {proc.stderr}")
        check(health(int(env["SMM_EMBED_PORT"])), "embedder should have been started")
        check(not health(int(env["SMM_RERANK_PORT"])),
              "reranker must NEVER start for --no-rerank")
        check(health(int(env["SMM_GEN_PORT"])), "generator should have been started")
    finally:
        cleanup(env)
        shutil.rmtree(tmp, ignore_errors=True)


def test_lifecycle_reuses_healthy_server():
    """A server already healthy must be reused, not restarted."""
    tmp = ROOT / ".run" / "test-lifecycle-reuse"
    tmp.mkdir(parents=True, exist_ok=True)
    env = isolated_env(tmp)
    try:
        fake_ask = write_fake_ask(tmp)
        p1 = daemon_run(env, fake_ask, ["--retrieve-only", "q1"])
        check(p1.returncode == 0, f"first run failed: {p1.stderr}")
        pidfile = Path(env["SMM_RUN"]) / "embedder.pid"
        check(pidfile.exists(), "embedder pidfile should exist after first run")
        pid1 = pidfile.read_text().strip()

        p2 = daemon_run(env, fake_ask, ["--retrieve-only", "q2"])
        check(p2.returncode == 0, f"second run failed: {p2.stderr}")
        pid2 = pidfile.read_text().strip()
        check(pid1 == pid2, f"embedder should be reused, not restarted: {pid1} != {pid2}")
    finally:
        cleanup(env)
        shutil.rmtree(tmp, ignore_errors=True)


def test_reaper_single_instance_pidfile_liveness():
    """Exactly one reaper at a time: a live one is left alone, a stale pidfile
    (dead pid) is replaced by a fresh one."""
    tmp = ROOT / ".run" / "test-reaper-single"
    tmp.mkdir(parents=True, exist_ok=True)
    env = isolated_env(tmp, idle=600)  # long idle: this test only checks spawning, not reaping
    try:
        fake_ok = write_fake_ok(tmp)
        p1 = daemon_run(env, fake_ok, [])
        check(p1.returncode == 0, f"first run failed: {p1.stderr}")
        reaper_pid_file = Path(env["SMM_RUN"]) / "reaper.pid"
        check(reaper_pid_file.exists(), "a reaper should have been spawned")
        pid1 = reaper_pid_file.read_text().strip()
        check(os.path.exists(f"/proc/{pid1}"), "reaper should be alive")

        p2 = daemon_run(env, fake_ok, [])
        check(p2.returncode == 0, f"second run failed: {p2.stderr}")
        pid2 = reaper_pid_file.read_text().strip()
        check(pid1 == pid2, "a second invocation must not spawn a second reaper")

        # simulate a stale pidfile: a pid that is guaranteed dead. pid1 is
        # still genuinely alive at this point (idle=600, nothing reaps it) -
        # overwriting the pidfile only orphans our *reference* to it, so it
        # must be killed explicitly rather than left to the pidfile-based
        # cleanup below, which will only ever see whatever pid3 turns out to be.
        dead = subprocess.Popen(["true"])
        dead.wait()
        reaper_pid_file.write_text(str(dead.pid))
        p3 = daemon_run(env, fake_ok, [])
        check(p3.returncode == 0, f"third run failed: {p3.stderr}")
        pid3 = reaper_pid_file.read_text().strip()
        check(pid3 != str(dead.pid), "a dead pidfile must be replaced by a fresh reaper")
        check(os.path.exists(f"/proc/{pid3}"), "the replacement reaper should be alive")
    finally:
        try:
            os.kill(int(pid1), 15)
        except (NameError, ValueError, OSError):
            pass
        cleanup(env)
        shutil.rmtree(tmp, ignore_errors=True)


def test_reaper_refresh_then_idle_reaps():
    """The core auto-cleanup contract: a touch inside the window cancels the
    pending reap (the reaper re-checks the stamp on wake, it does not act on
    a deadline computed before the refresh); once truly idle for SMM_IDLE
    seconds it stops every server and removes every pidfile."""
    tmp = ROOT / ".run" / "test-reaper-idle"
    tmp.mkdir(parents=True, exist_ok=True)
    # idle=5s: small enough to keep the test quick, large enough to clear the
    # ~1s polling granularity of scripts/servers.sh's own wait_ready loop, so
    # normal stub startup latency can't itself trip the reap.
    env = isolated_env(tmp, idle=5)
    try:
        fake_ask = write_fake_ask(tmp)
        p1 = daemon_run(env, fake_ask, ["--retrieve-only", "q"])
        check(p1.returncode == 0, f"initial run failed: {p1.stderr}")
        check(health(int(env["SMM_EMBED_PORT"])), "embedder should be up after the first run")

        # Refresh the stamp partway through the idle window - this is "a
        # fresh invocation inside the window", the case that must survive.
        time.sleep(1.5)
        p2 = daemon_run(env, fake_ask, ["--retrieve-only", "q2"])
        check(p2.returncode == 0, f"refresh run failed: {p2.stderr}")

        # 3.5s after the refresh: less than SMM_IDLE=5s since the refresh,
        # but more than 5s since the *original* touch (1.5 + 3.5 = 5.0+).
        # A reaper acting on the stale deadline would have killed the
        # servers by now.
        time.sleep(3.5)
        check(health(int(env["SMM_EMBED_PORT"])),
              "a fresh invocation inside the window must not be reaped")

        # Now let it actually go idle past SMM_IDLE since the last touch.
        reaped = wait_until(lambda: not health(int(env["SMM_EMBED_PORT"])), timeout=10.0)
        check(reaped, "servers should be stopped once truly idle for SMM_IDLE seconds")
        check(not health(int(env["SMM_RERANK_PORT"])), "reranker should be stopped too")

        run_dir = Path(env["SMM_RUN"])
        wait_until(lambda: not (run_dir / "embedder.pid").exists(), timeout=5.0)
        check(not (run_dir / "embedder.pid").exists(), "embedder pidfile should be removed")
        check(not (run_dir / "reranker.pid").exists(), "reranker pidfile should be removed")
        wait_until(lambda: not (run_dir / "reaper.pid").exists(), timeout=5.0)
        check(not (run_dir / "reaper.pid").exists(),
              "the reaper should remove its own pidfile once it has reaped")
    finally:
        cleanup(env)
        shutil.rmtree(tmp, ignore_errors=True)


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
