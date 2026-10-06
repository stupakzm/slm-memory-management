"""Tests for the pure parts of scripts/train_reader.py (reader LoRA fine-tune on RAFT data).

Hermetic (blk_test_env_constraints): /usr/bin/python3, no pytest, no torch, no GPU, no
data/ or models/. The tokenizer is a small fake that renders the chat template the way
Qwen's does and splits text into word / whitespace / punctuation / special-token pieces.
NOT covered here (needs torch and weights): the LoRA wrapper and merge, the model loss,
checkpoint save/load, generation, save_merged, and the real tokenizer's chat template.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tr = _load("train_reader")
from smm import grammar  # noqa: E402


def check(cond, msg=""):
    if not cond:
        raise AssertionError(msg)


class FakeTok:
    """Pieces: <|special|> tokens, words, single whitespace chars, single punctuation."""
    piece = re.compile(r"<\|[a-z_]+\|>|\w+|\s|[^\w\s]")

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.vocab.setdefault(p, len(self.vocab))
                              for p in self.piece.findall(text)]}

    def decode(self, ids):
        inv = {i: p for p, i in self.vocab.items()}
        return "".join(inv[i] for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")


class GluedTok(FakeTok):
    """Whitespace sticks to the word after it, so tokenising prompt+answer jointly would
    differ from tokenising them apart at the boundary."""
    piece = re.compile(r"<\|[a-z_]+\|>|\s*\w+|\s|[^\w\s]")


def msgs(answer, user="how do I list files?"):
    return [{"role": "system", "content": "Answer from the extracts."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer}]


def test_split_is_deterministic_by_example_id():
    ids = [f"ex-{i:05d}" for i in range(5000)]
    a = {i: tr.split_of(i, 0.02) for i in ids}
    check(a == {i: tr.split_of(i, 0.02) for i in reversed(ids)}, "independent of order")
    check(set(a.values()) == {"dev", "train"}, set(a.values()))
    n_dev = sum(1 for v in a.values() if v == "dev")
    check(60 < n_dev < 140, n_dev)  # 2% of 5000 = 100
    for i in ids[:50]:  # the rule itself
        want = int(hashlib.sha256(i.encode()).hexdigest(), 16) % 10000 < 0.02 * 10000
        check((a[i] == "dev") == want, i)
    edge = {}  # ids whose hash bucket is exactly 199 / 200: the threshold is "< 200", not "<= 200"
    for n in range(400000):
        h = int(hashlib.sha256(f"edge-{n}".encode()).hexdigest(), 16) % 10000
        if h in (199, 200):
            edge.setdefault(h, f"edge-{n}")
        if len(edge) == 2:
            break
    check(len(edge) == 2, "found boundary ids")
    check(tr.split_of(edge[199], 0.02) == "dev" and tr.split_of(edge[200], 0.02) == "train", edge)
    check(all(tr.split_of(i, 0.0) == "train" for i in ids[:200]), "dev_frac 0: no dev")
    check(all(tr.split_of(i, 1.0) == "dev" for i in ids[:200]), "dev_frac 1: all dev")
    check(all(tr.split_of(i, 0.5) == "dev" for i in ids if a[i] == "dev"),
          "a larger dev_frac only adds examples")
    rows = [{"id": i, "messages": msgs("a")} for i in ids[:300]]
    got = tr.prepare(FakeTok(), rows, 4096, 0.02)
    check({e["id"] for e in got["dev"]} == {i for i in ids[:300] if a[i] == "dev"}, "prepare splits by split_of")
    check(len(got["train"]) + len(got["dev"]) == 300, "every example in exactly one side")


def test_loss_mask_covers_only_the_assistant_answer():
    tok = FakeTok()
    answer = "Use ls -l [1]."
    ids, labels = tr.build_example(tok, msgs(answer), 4096)
    check(len(ids) == len(labels), "same length")
    n_prompt = sum(1 for x in labels if x == tr.IGNORE)
    check(all(x == tr.IGNORE for x in labels[:n_prompt]), "the prompt is a contiguous masked prefix")
    check(all(l == i for l, i in zip(labels[n_prompt:], ids[n_prompt:])), "answer labels equal the ids")
    trained = tok.decode([x for x in labels if x != tr.IGNORE])
    check(trained == answer + "<|im_end|>", repr(trained))  # the end marker is trained, nothing else
    for foreign in ("Answer from the extracts.", "how do I list files?", "system", "user", "<|im_start|>"):
        check(foreign not in trained, f"{foreign!r} leaked into the loss")
    check(tr.n_target_tokens(labels) == len(labels) - n_prompt, "targets = answer tokens")
    # an assistant turn that is not last is not an example
    try:
        tr.build_example(tok, msgs(answer)[:-1], 4096)
        check(False, "no assistant answer should raise")
    except ValueError:
        pass


def test_prompt_and_answer_token_boundary_is_exact():
    for tok in (FakeTok(), GluedTok()):
        m = msgs("I don't know.")
        prompt = tok.apply_chat_template(m[:-1], tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        ids, labels = tr.build_example(tok, m, 4096)
        n = len(prompt_ids)
        check(ids[:n] == prompt_ids, "the prompt tokens are exactly what inference feeds the model")
        check(tok.decode(ids[:n]).endswith("<|im_start|>assistant\n"), "generation prompt is in the prompt")
        check(labels[n - 1] == tr.IGNORE and labels[n] == ids[n] != tr.IGNORE,
              "the first answer token is the first labelled one")
        check(tok.decode(ids[n:]) == "I don't know.<|im_end|>", tok.decode(ids[n:]))
        check(tok.decode(ids) == prompt + "I don't know.<|im_end|>", "nothing added or lost")
    # the template, not the raw messages, builds the prompt: without the generation prompt
    # the boundary would sit before "<|im_start|>assistant"
    bare = FakeTok().apply_chat_template(msgs("x")[:-1], tokenize=False, add_generation_prompt=False)
    check(not bare.endswith("assistant\n"), "fake template honours add_generation_prompt")


def test_over_long_examples_are_dropped_and_counted():
    tok = FakeTok()
    short = msgs("ok")
    long_ = msgs("word " * 200)
    n_short = len(tr.build_example(tok, short, 10**6)[0])
    check(tr.build_example(tok, short, n_short) is not None, "exactly max_len is kept")
    check(tr.build_example(tok, short, n_short - 1) is None, "one over is dropped")
    rows = [{"id": "a", "messages": short}, {"id": "b", "messages": long_},
            {"id": "c", "messages": short}, {"id": "d", "messages": long_}]
    got = tr.prepare(tok, rows, n_short + 5, 0.5)
    check(got["dropped"] == ["b", "d"], got["dropped"])
    kept = got["train"] + got["dev"]
    check(sorted(e["id"] for e in kept) == ["a", "c"], "only the short ones survive")
    check(all(len(e["input_ids"]) == n_short for e in kept), "kept examples are whole, never truncated")
    check(tr.prepare(tok, rows, 10**6, 0.5)["dropped"] == [], "nothing dropped when all fit")


def test_is_refusal_matches_the_grammar_refusal_only():
    check(tr.is_refusal(grammar.REFUSAL), "the grammar's own refusal")
    check(tr.is_refusal("  " + grammar.REFUSAL + "\n"), "strip is applied")
    for no in ("I don't know", "I do not know.", "i don't know.", "I don't know. [1]",
               "Sure. I don't know.", "I don't know.\nI don't know.", "", "   ", None,
               "Use ls [1]."):
        check(not tr.is_refusal(no), repr(no))


def test_refusal_metrics_precision_and_recall():
    # gold, predicted: 2 true positives, 1 false positive, 1 false negative, 3 true negatives
    pairs = [(True, True)] * 2 + [(False, True)] + [(True, False)] + [(False, False)] * 3
    m = tr.refusal_metrics(pairs)
    check(m["n"] == 7 and abs(m["precision"] - 2 / 3) < 1e-12 and abs(m["recall"] - 2 / 3) < 1e-12, m)
    m = tr.refusal_metrics([(True, True)] + [(False, True)] * 3)
    check(m["precision"] == 0.25 and m["recall"] == 1.0, m)  # precision and recall differ
    m = tr.refusal_metrics([(True, False)] * 2 + [(False, False)])
    check(m["precision"] is None and m["recall"] == 0.0 and m["n"] == 3, m)  # nothing predicted
    m = tr.refusal_metrics([(False, True), (False, False)])
    check(m["precision"] == 0.0 and m["recall"] is None, m)  # no gold refusals
    check(tr.refusal_metrics([]) == {"precision": None, "recall": None, "n": 0}, "empty")
    check(tr.refusal_metrics(iter([(True, True)]))["n"] == 1, "accepts an iterator")


def test_lora_targets_are_all_attention_and_mlp_projections():
    attn = ["q_proj", "k_proj", "v_proj", "o_proj"]
    mlp = ["gate_proj", "up_proj", "down_proj"]
    names = ["lm_head", "model.embed_tokens"]
    want = []
    for layer in range(2):
        p = f"model.layers.{layer}"
        names += [f"{p}.self_attn.{n}" for n in attn] + [f"{p}.mlp.{n}" for n in mlp]
        names += [f"{p}.self_attn", f"{p}.mlp", f"{p}.self_attn.q_norm", f"{p}.input_layernorm"]
        want += [f"{p}.self_attn.{n}" for n in attn] + [f"{p}.mlp.{n}" for n in mlp]
    got = tr.lora_target_names(names)
    check(got == want, got)
    check(len(got) == 14 and not any("lm_head" in n for n in got), "7 per layer, never lm_head")
    check(set(tr.LORA_TARGETS) == set(attn + mlp), tr.LORA_TARGETS)
    check(tr.lora_target_names(iter(["a.q_proj"])) == ["a.q_proj"], "accepts an iterator")
    check(tr.lora_target_names(["model.q_projection", "x_proj", "proj"]) == [], "exact last-component match")


def test_resume_state_roundtrip():
    plan = {"n_train": 100, "grad_accum": 16, "epochs": 2, "seed": 15, "total_steps": 14}
    sha = hashlib.sha256(b"data").hexdigest()
    state = tr.make_resume_state(10, plan, sha)
    check(state["step"] == 10 and state["epoch"] == 1, state)
    check(state["rng_position"] == {"seed": 15, "epoch": 1, "step_in_epoch": 3}, state["rng_position"])
    check(state["data_sha256"] == sha and state["plan"] == plan, state)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "out" / tr.STATE_NAME
        tr.save_resume_state(p, state)
        check(not list(p.parent.glob("*.tmp")), "atomic write leaves no temp file")
        check(tr.load_resume_state(p, sha, plan) == state, "roundtrip")
        for bad_sha, bad_plan, what in (("0" * 64, plan, "different data"),
                                        (sha, {**plan, "epochs": 3}, "different plan")):
            try:
                tr.load_resume_state(p, bad_sha, bad_plan)
                check(False, f"{what} must be refused")
            except tr.ResumeError:
                pass
        f = Path(td) / "data.jsonl"
        f.write_bytes(b"data")
        check(tr.file_sha256(f) == sha, "file_sha256 is the sha256 of the bytes")
        f.write_bytes(b"data2")
        check(tr.file_sha256(f) != sha, "and changes with them")


def test_import_works_without_torch():
    code = (
        "import sys, importlib.abc\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('torch', 'transformers', 'peft', 'trl', 'safetensors'):\n"
        "            raise ImportError('blocked: ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "try:\n"
        "    import torch\n"
        "    raise SystemExit('the blocker does not work')\n"
        "except ImportError:\n"
        "    pass\n"
        f"sys.path.insert(0, r'{ROOT / 'scripts'}')\n"
        "import train_reader\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('torch', 'transformers', 'peft', 'safetensors')]\n"
        "print(len(bad), train_reader.split_of('x', 0.5) in ('dev', 'train'))\n"
    )
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    check(r.returncode == 0 and r.stdout.strip() == "0 True", (r.stdout + r.stderr)[-500:])
    check("torch" not in sys.modules, "torch leaked into this process")


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeTrainer:
    def __init__(self, clock, seconds_per_step, checkpoint_at=()):
        self.clock, self.dt, self.checkpoint_at = clock, seconds_per_step, set(checkpoint_at)
        self.ran: list[int] = []
        self.saved: list[int] = []

    def step(self, i):
        self.clock.now += self.dt
        self.ran.append(i)
        return 1.0 / (i + 1)

    def save(self, step):
        self.saved.append(step)

    def checkpoint(self, step):
        return step in self.checkpoint_at


def test_stop_after_seconds_saves_and_returns_zero_code():
    clock = FakeClock()
    t = FakeTrainer(clock, 10.0)
    msgs_out: list[str] = []
    r = tr.train_loop(t, 0, 10, stop_after_s=25, clock=clock, log=msgs_out.append)
    check(r["status"] == "paused" and r["step"] == 3 and r["exit_code"] == 0, r)
    check(t.ran == [0, 1, 2], "the step that crosses the cap is finished, no more")
    check(t.saved == [3], f"saved once at the pause: {t.saved}")
    check([s for s, _ in r["losses"]] == [1, 2, 3] and msgs_out, "losses and a log line")
    # a checkpoint at the very step that pauses is not saved twice
    clock = FakeClock()
    t = FakeTrainer(clock, 10.0, checkpoint_at={3})
    r = tr.train_loop(t, 0, 10, stop_after_s=25, clock=clock, log=lambda s: None)
    check(r["status"] == "paused" and t.saved.count(3) == 1, t.saved)
    # resume: continues from the saved step and runs to the end under no cap
    clock2 = FakeClock()
    t2 = FakeTrainer(clock2, 10.0, checkpoint_at={5})
    r2 = tr.train_loop(t2, 3, 10, clock=clock2, log=lambda s: None)
    check(r2["status"] == "complete" and r2["step"] == 10 and r2["exit_code"] == 0, r2)
    check(t2.ran == list(range(3, 10)) and t2.saved == [5], (t2.ran, t2.saved))
    # the cap never pauses at the last step: that is a completed run
    clock3 = FakeClock()
    t3 = FakeTrainer(clock3, 10.0)
    r3 = tr.train_loop(t3, 0, 3, stop_after_s=5, clock=clock3, log=lambda s: None)
    check(r3["status"] == "paused" and r3["step"] == 1, r3)
    clock4 = FakeClock()
    t4 = FakeTrainer(clock4, 10.0)
    r4 = tr.train_loop(t4, 2, 3, stop_after_s=5, clock=clock4, log=lambda s: None)
    check(r4["status"] == "complete" and t4.saved == [], (r4, t4.saved))


def test_step_plan_and_schedule():
    plan = tr.step_plan(37, 16, 2, 15)
    check(len(plan) == 2 * 3 and [len(p) for p in plan] == [16, 16, 5] * 2, [len(p) for p in plan])
    for e in range(2):
        check(sorted(i for p in plan[3 * e:3 * e + 3] for i in p) == list(range(37)), "each epoch covers all")
    check(plan == tr.step_plan(37, 16, 2, 15) and plan != tr.step_plan(37, 16, 2, 16), "seeded")
    check(plan[:3] != plan[3:] and sorted(plan[0] + plan[1] + plan[2]) == sorted(plan[3] + plan[4] + plan[5]),
          "epochs are shuffled differently")
    total = 200
    check(tr.lr_factor(0, total) == 1 / 6 and abs(tr.lr_factor(5, total) - 1.0) < 1e-12,
          "3% warmup = 6 steps, linear")
    check(abs(tr.lr_factor(6, total) - 1.0) < 1e-12 and tr.lr_factor(total, total) < 1e-9, "cosine from 1 to 0")
    mid = tr.lr_factor(6 + (total - 6) // 2, total)
    check(abs(mid - 0.5) < 0.01, mid)


def test_read_examples_validates():
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "d.jsonl"
        good = {"id": "a", "messages": msgs("x"), "gold_removed": False, "typo": False}
        f.write_text(json.dumps(good) + "\n\n" + json.dumps({**good, "id": "b"}) + "\n")
        check([r["id"] for r in tr.read_examples(f)] == ["a", "b"], "blank lines skipped")
        for bad in ({**good, "id": "a"}, {"messages": msgs("x")}, {"id": "z", "messages": msgs("x")[:-1]}):
            f.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n")
            try:
                tr.read_examples(f)
                check(False, f"should reject {bad}")
            except ValueError:
                pass


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
