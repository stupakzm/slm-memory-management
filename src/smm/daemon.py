"""Server lifecycle for the `asq`/`smm` launcher: health checks, lazy
start-on-demand, an activity stamp, and the idle reaper that frees VRAM once
nobody has touched the stamp for `SMM_IDLE` seconds (default 600).

Deliberately dependency-free (stdlib only): urllib for health checks,
subprocess to shell out, os/time for the reaper loop. The actual llama-server
flags and VRAM sizing live in scripts/servers.sh and are reused as-is via its
`start <target>` / `stop` CLI — this module only decides *when* to call them.

This file is run directly (`python daemon.py <cmd>`), not imported as part of
the `smm` package, so it has no relative imports and no import-time side
effects beyond stdlib.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
# SMM_RUN/SMM_*_PORT mirror the same-named overrides in scripts/servers.sh -
# unset, this is exactly the real pidfile directory and the real ports.
# tests/test_cli.py sets them so it never touches a real, already-running
# server or its pidfiles.
RUN = Path(os.environ.get("SMM_RUN", str(ROOT / ".run")))
SERVERS_SH = ROOT / "scripts" / "servers.sh"

# name -> (port, servers.sh start target). Order is embedder -> reranker ->
# generator: that is the order ask.py itself checks health in (it needs the
# embedder to embed the query before it can rerank, and the gate can refuse
# before the generator is ever touched), so a wrapped run that retries on
# "not running" naturally brings servers up in this order and no other.
MODELS = [
    ("embedder", int(os.environ.get("SMM_EMBED_PORT", 8081)), "embedder-lean"),
    ("reranker", int(os.environ.get("SMM_RERANK_PORT", 8082)), "reranker-query"),
    ("generator", int(os.environ.get("SMM_GEN_PORT", 8080)), "generator-query"),
]
PORT_OF = {name: port for name, port, _ in MODELS}
TARGET_OF = {name: target for name, _, target in MODELS}

IDLE_DEFAULT = 600
STAMP = RUN / "last-use"
REAPER_PID = RUN / "reaper.pid"
REAPER_LOG = RUN / "reaper.log"


def idle_seconds() -> float:
    return float(os.environ.get("SMM_IDLE", IDLE_DEFAULT))


def health(port: int, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def status() -> int:
    for name, port, _ in MODELS:
        up = health(port)
        print(f"  {name:9} {'UP' if up else 'down':5} :{port}")
    pid = _reaper_alive()
    print(f"  reaper    {'UP   ' if pid else 'down '}" + (f" pid {pid}" if pid else ""))
    return 0


def start(name: str, wait: float = 90.0) -> bool:
    """Start one server via servers.sh and block until it answers /health."""
    target = TARGET_OF[name]
    port = PORT_OF[name]
    RUN.mkdir(parents=True, exist_ok=True)
    with open(RUN / f"{name}.launch.log", "ab") as log:
        subprocess.run(["bash", str(SERVERS_SH), "start", target],
                        stdout=log, stderr=log, check=False)
    deadline = time.time() + wait
    while time.time() < deadline:
        if health(port):
            return True
        time.sleep(0.5)
    return False


def ensure_for(names: list[str]) -> None:
    for name in names:
        if not health(PORT_OF[name]):
            start(name)


def touch() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    STAMP.write_text(str(time.time()))


def _reaper_alive() -> int | None:
    if not REAPER_PID.exists():
        return None
    try:
        pid = int(REAPER_PID.read_text().strip())
    except ValueError:
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    return pid


def ensure_reaper() -> None:
    """Make sure exactly one reaper is watching the stamp (pidfile + liveness
    check). Spawned detached, in its own session, so it survives this process
    exiting and never holds the terminal open."""
    if _reaper_alive():
        return
    RUN.mkdir(parents=True, exist_ok=True)
    log = open(REAPER_LOG, "ab")
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "reap"],
        stdin=subprocess.DEVNULL, stdout=log, stderr=log,
        start_new_session=True, cwd=str(ROOT),
    )
    REAPER_PID.write_text(str(proc.pid))


def _stop_all() -> None:
    subprocess.run(["bash", str(SERVERS_SH), "stop"], check=False)


def reap_loop() -> None:
    """The detached reaper's body. Re-reads the stamp every time it wakes, so
    a fresh invocation that lands mid-sleep pushes the deadline out instead of
    being killed for a deadline that is now stale. Sleeps in short slices
    (capped at 5s) purely so it wakes often enough to notice that refresh —
    it never acts on anything but the stamp it just read."""
    while True:
        idle = idle_seconds()
        try:
            stamp = float(STAMP.read_text())
        except (OSError, ValueError):
            stamp = time.time()
        remaining = idle - (time.time() - stamp)
        if remaining <= 0:
            _stop_all()
            try:
                REAPER_PID.unlink()
            except OSError:
                pass
            return
        time.sleep(min(remaining, 5.0))


# Exit code 2 is what ask.py/ingest.py use for "a server this run needs is not
# up" (and only for that - see scripts/ask.py:73,79,107 and
# scripts/ingest.py:91). Matching the server's name in stderr is enough to
# tell which one, without parsing the exact wording either script chose.
def _missing_server(stderr_text: str, already_tried: set[str]) -> str | None:
    for name, _, _ in MODELS:
        if name in stderr_text and name not in already_tried:
            return name
    return None


def run_wrapped(python: str, script: str, argv: list[str]) -> int:
    """Run `python script *argv`, lazily starting whichever server it reports
    missing and retrying, in embedder -> reranker -> generator order. A
    healthy server already up is reused, never restarted. If the gate refuses
    before ever checking the generator, this loop never starts one.

    Touches the stamp again after each server actually comes up (not just
    once at the top): bringing three models up from cold can itself take
    longer than a short idle window, and an invocation still busy starting
    what it needs must not be reaped out from under itself."""
    touch()
    ensure_reaper()
    tried: set[str] = set()
    proc = None
    for _ in range(len(MODELS) + 1):
        proc = subprocess.run([python, script, *argv], stderr=subprocess.PIPE)
        if proc.returncode != 2:
            if proc.stderr:
                sys.stderr.buffer.write(proc.stderr)
            touch()
            return proc.returncode
        err = proc.stderr.decode(errors="replace")
        missing = _missing_server(err, tried)
        if missing is None:
            sys.stderr.buffer.write(proc.stderr)
            touch()
            return proc.returncode
        tried.add(missing)
        if not start(missing):
            sys.stderr.write(f"asq: {missing} failed to start; see .run/{missing}.launch.log\n")
            sys.stderr.buffer.write(proc.stderr)
            touch()
            return 2
        touch()
    if proc is not None and proc.stderr:
        sys.stderr.buffer.write(proc.stderr)
    return proc.returncode if proc is not None else 1


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "status"
    if cmd == "status":
        return status()
    if cmd == "stop":
        _stop_all()
        pid = _reaper_alive()
        if pid:
            try:
                os.kill(pid, 15)
            except OSError:
                pass
            try:
                REAPER_PID.unlink()
            except OSError:
                pass
        return 0
    if cmd == "touch":
        touch()
        return 0
    if cmd == "ensure":
        ensure_for(argv[1:])
        return 0
    if cmd == "reap":
        reap_loop()
        return 0
    if cmd == "run":
        # run <python> <script> -- <args...>
        rest = argv[1:]
        if "--" not in rest:
            print("usage: daemon.py run <python> <script> -- <args...>", file=sys.stderr)
            return 2
        sep = rest.index("--")
        python, script = rest[0], rest[1]
        return run_wrapped(python, script, rest[sep + 1:])
    print(f"unknown daemon command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
