"""Tests for the part of the system that decides whether anything runs.

The phase 4 exit gate is "zero unconfirmed destructive executions across the full
eval run". The eval demonstrates that on 80 questions; these assert it on the cases
an eval set would not think to contain.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smm.tools import TOOLS, Call, classify, execute, grammar, parse_call  # noqa: E402

PAGES = {"tar.1", "mount.8", "signal.7", "sshd_config.5"}


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_classify():
    for cmd in ["rm -rf /tmp/build", "dd if=x of=/dev/sdb", "mke2fs -L backup /dev/sdb1",
                "chmod -R 777 /var/www", "chown -R www-data /srv", "userdel -r olduser",
                "systemctl stop nginx", "shred -u secret.txt", "truncate -s 0 app.log"]:
        check(classify(cmd) == "destructive", f"should be destructive: {cmd}")
    for cmd in ["ls -lh /var/log", "df -i /home", "ps aux", "ss -ltn",
                "systemctl list-unit-files", "journalctl -b", "stat -c %s f.pdf"]:
        check(classify(cmd) == "read-only", f"should be read-only: {cmd}")
    # Anything that can hide a second command is never auto-run, whatever it starts as.
    for cmd in ["ls; rm -rf /", "ls && rm x", "cat f | sh", "echo $(rm -rf /)",
                "ls > /dev/sda", "sudo ls"]:
        check(classify(cmd) == "destructive", f"shell metacharacter must block: {cmd}")
    check(classify("frobnicate --wat") == "unknown", "unknown binary")


def test_parse_rejects_malformed():
    n = 5
    bad = [
        ('not json', "not JSON"),
        ('{"tool": "rm_rf", "cite": 1}', "unknown tool"),
        ('{"tool": "propose_command", "command": "ls", "cite": 1}', "missing argument"),
        ('{"tool": "propose_command", "command": "", "explanation": "x", "cite": 1}',
         "empty string"),
        ('{"tool": "answer", "text": "hi", "cite": 9}', "cite out of range"),
        ('{"tool": "show_manpage", "page": "tar", "section": "3"}', "bad enum"),
        ('{"tool": "show_manpage", "page": "rm -rf /", "section": "1"}', "page injection"),
        ('{"tool": "show_manpage", "page": "nosuch", "section": "1"}', "page not here"),
    ]
    for text, why in bad:
        c = parse_call(text, TOOLS, n, known_pages=PAGES)
        check(not c.valid, f"should have been rejected ({why}): {text}")


def test_parse_accepts_good():
    c = parse_call('{"tool": "propose_command", "command": "ls -lh /var/log",'
                   ' "explanation": "list", "cite": 2}', TOOLS, 5, known_pages=PAGES)
    check(c.valid and c.tool == "propose_command" and c.cite == 2, "valid call rejected")
    check(c.risk == "read-only", "ls should be read-only")
    c = parse_call('{"tool": "show_manpage", "page": "tar", "section": "1"}',
                   TOOLS, 5, known_pages=PAGES)
    check(c.valid and c.risk == "read-only", "manpage call rejected")


def test_propose_command_never_runs():
    """The gate. A proposed command does not run - confirmed, allowed, or read-only."""
    for cmd in ["rm -rf /tmp/build", "ls -lh"]:
        c = parse_call(f'{{"tool": "propose_command", "command": "{cmd}",'
                       f' "explanation": "x", "cite": 1}}', TOOLS, 5, known_pages=PAGES)
        for confirm in (None, lambda _c: True):
            r = execute(c, confirm=confirm, allow_execution=True)
            check(r["ran"] is False, f"propose_command must never run: {cmd} {r}")


def test_destructive_needs_confirmation():
    c = parse_call('{"tool": "propose_command", "command": "rm -rf /tmp/x",'
                   ' "explanation": "x", "cite": 1}', TOOLS, 5, known_pages=PAGES)
    check(c.needs_confirmation, "destructive call must need confirmation")
    r = execute(c, confirm=lambda _c: False, allow_execution=True)
    check(r["ran"] is False and r["reason"] == "not confirmed", f"unexpected: {r}")


def test_execution_off_by_default():
    c = parse_call('{"tool": "show_manpage", "page": "tar", "section": "1"}',
                   TOOLS, 5, known_pages=PAGES)
    check(execute(c)["ran"] is False, "nothing runs unless execution is enabled")
    check(execute(c, allow_execution=True)["ran"] is True, "read-only tool should run")


def test_invalid_call_never_runs():
    c = Call("show_manpage", {"page": "tar", "section": "1"}, problems=["synthetic"])
    check(execute(c, allow_execution=True)["ran"] is False, "invalid call must not run")


def test_grammar_shape():
    g = grammar(TOOLS, 5)
    check("root ::=" in g and g.count("::=") >= len(TOOLS) + 1, "grammar incomplete")
    check('("\\"1\\"" | "\\"5\\"" | "\\"7\\"" | "\\"8\\"")' in g,
          "enum alternation must be parenthesised")


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
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
