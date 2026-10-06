"""Tests for --reader-prompt v3 (tsk_20261006_readerv3): SYSTEM_V2 followed by three
worked examples. v1 and v2 messages must stay byte-identical.

Hermetic (blk_test_env_constraints): stdlib only; sqlite_vec is stubbed, the
helpers come from tests/test_instructions.py.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_instructions as ti  # noqa: E402  (stubs sqlite_vec, loads eval_answers)

generate = ti.generate
check = ti.check
CHUNKS = ti.CHUNKS
EXPECTED_SHA = "22333b33f4b7bfa8bfacb9b0fb83fe270ca0bde60155ccfec8324b34d5087d42"


def test_v3_system_starts_with_v2_and_adds_three_examples():
    s = generate.SYSTEM_V3
    check(s == generate.SYSTEM_V2 + generate.EXAMPLES_V3, "SYSTEM_V3 = SYSTEM_V2 + EXAMPLES_V3")
    check(s.startswith(generate.SYSTEM_V2), "starts with SYSTEM_V2")
    check(generate.EXAMPLES_V3.startswith("\nWorked examples."), repr(generate.EXAMPLES_V3[:30]))
    for n in (1, 2, 3):
        check(s.count(f"Example {n} - ") == 1, f"example {n}")
    check("Example 4" not in s, "exactly three examples")


def test_v3_examples_text_is_pinned():
    e = generate.EXAMPLES_V3
    got = hashlib.sha256(e.encode()).hexdigest()
    check(got == EXPECTED_SHA, got)
    check("Answer: I don't know." in e, "refusal sentinel")
    check("[3]" in e.split("Example 2")[1].split("Example 3")[0], "example 2 cites [3]")
    check("firts" in e, "misspelling sentinel")


def test_v3_messages_use_documentation_header():
    g = ti._Chat()
    g.answer("q?", CHUNKS, cite_grammar=True, reader_prompt="v3")
    msgs, grammar = g.calls[0]
    check(msgs[0] == {"role": "system", "content": generate.SYSTEM_V3}, msgs[0])
    check(msgs[1]["content"] == "Documentation extracts:\n\n[1] d.1\np t\n\nQuestion: q?",
          msgs[1]["content"])
    check(grammar is not None, "cite grammar is still applied under v3")


def test_v1_and_v2_messages_unchanged():
    g = ti._Chat()
    g.answer("q?", CHUNKS, reader_prompt="v1")
    g.answer("q?", CHUNKS, reader_prompt="v2")
    check(g.calls[0][0] == generate.build_prompt("q?", CHUNKS), "v1 messages")
    check(g.calls[1][0] == generate.build_prompt("q?", CHUNKS, system=ti.V2,
                                                 header="Documentation extracts"), "v2 messages")
    check(generate.SYSTEM == ti.V1 and generate.SYSTEM_V2 == ti.V2, "v1/v2 system text")


def test_v3_rejects_quote_mode():
    try:
        ti._Chat().answer("q?", CHUNKS, reader_prompt="v3", mode="quote")
    except ValueError:
        return
    raise AssertionError("v3 with quote mode must raise ValueError")


def test_unknown_reader_prompt_rejected():
    for bad in ("v4", "V3", ""):
        try:
            ti._Chat().answer("q?", CHUNKS, reader_prompt=bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} must raise ValueError")


def test_eval_answers_accepts_v3_and_records_it():
    code, out, _, ans_kw = ti._eval_run(["--reader-prompt", "v3"])
    check(code == 0, f"exit {code}")
    check(ans_kw == [{"reader_prompt": "v3"}], f"v3 reaches answer(): {ans_kw}")
    check(out["config"].get("reader_prompt") == "v3", out["config"])
    code, out, _, ans_kw = ti._eval_run(["--reader-prompt", "v3", "--answer-mode", "quote"])
    check(code == 2 and out is None and not ans_kw, f"v3 with quote must exit 2: {code}")


def test_eval_answers_default_config_has_no_reader_prompt():
    code, out, _, ans_kw = ti._eval_run([])
    check(code == 0 and ans_kw == [{}], f"{code} {ans_kw}")
    check("reader_prompt" not in out["config"], out["config"])


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  pass  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
