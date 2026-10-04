"""Drives emacs --batch against a stub asq; never runs the real asq."""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EL = os.path.join(ROOT, "emacs", "asq.el")
EMACS = "/usr/bin/emacs"
QUESTION = 'how do I "quote" $HOME; rm -rf *  now'


def _stub(d):
    path = os.path.join(d, "stub-asq")
    with open(path, "w") as f:
        f.write('#!/bin/sh\nfor a in "$@"; do printf "%s\\n" "$a"; done\n')
    os.chmod(path, 0o755)
    return path


def _run_asq(prefix):
    """Return the lines of *asq* after the 'Q:' header, as argv elements."""
    assert os.access(EMACS, os.X_OK), "emacs missing: " + EMACS
    with tempfile.TemporaryDirectory() as d:
        stub = _stub(d)
        out = os.path.join(d, "out.txt")
        expr = (
            '(progn (setq asq-program "%s") (asq %s %s)'
            ' (while (process-live-p (get-buffer-process (get-buffer "*asq*")))'
            '   (accept-process-output nil 0.05))'
            ' (accept-process-output nil 0.2)'
            ' (with-temp-file "%s" (insert (with-current-buffer "*asq*" (buffer-string)))))'
        ) % (stub, '"%s"' % QUESTION.replace("\\", "\\\\").replace('"', '\\"'),
             "t" if prefix else "nil", out)
        r = subprocess.run(
            [EMACS, "--batch", "-Q", "-l", EL, "--eval", expr],
            capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        with open(out) as f:
            text = f.read()
    head = "Q: " + QUESTION + "\n\n"
    assert text.startswith(head), text
    return text[len(head):].split("\n")[:-1]


def test_asq_passes_db_and_domain():
    argv = _run_asq(False)
    assert argv == ["--db", "data/index/phase11.db", "--domain", "emacs", QUESTION], argv


def test_asq_prefix_searches_everything():
    argv = _run_asq(True)
    assert "--domain" not in argv, argv
    assert argv == ["--db", "data/index/phase11.db", QUESTION], argv


def test_asq_question_is_one_argument():
    argv = _run_asq(False)
    assert argv[-1] == QUESTION and len(argv) == 5, argv


def test_asq_byte_compiles_cleanly():
    assert os.access(EMACS, os.X_OK), "emacs missing: " + EMACS
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "asq.el")
        with open(EL) as f, open(src, "w") as g:
            g.write(f.read())
        r = subprocess.run(
            [EMACS, "--batch", "-Q", "--eval", "(setq byte-compile-error-on-warn t)",
             "-f", "batch-byte-compile", src],
            capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        assert os.path.exists(os.path.join(d, "asq.elc"))


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("pass", name)
            except Exception as e:
                failed += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if failed else 0)
