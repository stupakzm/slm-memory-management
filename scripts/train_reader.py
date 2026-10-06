#!/usr/bin/env python3
"""Fine-tune the reader (Qwen3-4B-Instruct-2507) with LoRA on RAFT chat data.

Input is the chat-format JSONL that scripts/build_raft_data.py writes, one example per
line: {id, messages: [system, user, assistant], gold_removed, typo}. The loss is token
cross-entropy over the ASSISTANT ANSWER ONLY (the prompt is masked to -100; the answer's
end-of-turn marker is trained on). A hash-chosen share of the examples is held out as
dev (split_of); dev loss, and refusal precision/recall by greedy generation, are logged
before and after. The result is a hand-written LoRA (no peft; same idiom as
scripts/train_embedder.py) on every attention and MLP projection, the adapter weights,
and a merged HF model directory (merged/, safetensors + config + tokenizer) that
llama.cpp's convert_hf_to_gguf.py can read.

Training-only: run under the training venv (requirements-train.txt: torch 2.6.0,
transformers 4.57.6), never the retrieval .venv. torch and transformers are imported only
inside the functions that train, so `import train_reader` works without either.

  .venv-train/bin/python scripts/train_reader.py --data data/raft/train.jsonl --out data/reader/r1
  ... --resume --stop-after-s 6600     # under a job cap: finish the step, save, exit 0, re-run
  ... --eval-only --out /tmp/base-eval # dev loss + refusal P/R of the untouched base model
  ... --merge-only --out data/reader/r1  # re-merge adapter.pt into merged/ (no training)
  python llama.cpp/convert_hf_to_gguf.py data/reader/r1/merged --outtype q8_0

One optimizer step = --grad-accum micro-batches of ONE example each (no padding); the
step loss is the mean over all answer tokens in those examples (token-weighted, not a
mean of per-example means). Examples whose tokenised length exceeds --max-len are
dropped and counted, never truncated. Resume state (resume_state.json beside adapter.pt)
holds step, epoch, rng position and the sha256 of the data file; the shuffle is a pure
function of (seed, epoch), so the rng position is (epoch, step_in_epoch). A resume against
different data, or a different plan, is refused.

WHAT HAS BEEN RUN, AND WHAT HAS NOT. Written where there is no torch, no weights and no
GPU. Run (tests/test_train_reader.py, stdlib only, a fake tokenizer): the pure parts -
the dev split, the prompt/answer boundary and loss mask, the length drop, the refusal
rule and metrics, LoRA target selection, the step plan and schedule, resume state, and
the time-cap control flow with a fake trainer. NOT RUN, read twice but never executed: load_model,
the LoRA wrapper and its merge, the loss on the real model, gradient checkpointing, the
optimizer, checkpoint save/load, greedy generation, save_merged, and the real tokenizer's
chat template (Qwen's renders an assistant turn as `<|im_start|>assistant\\n...<|im_end|>\\n`;
END_OF_TURN assumes that). The first GPU run should be a few steps on a tiny --data file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm import grammar  # noqa: E402  (stdlib only; the refusal text is never copied)

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
END_OF_TURN = "<|im_end|>"  # what Qwen's template closes an assistant turn with
IGNORE = -100
WARMUP_FRAC = 0.03
DEFAULT_MODEL_DIR = "models/hf/Qwen3-4B-Instruct-2507"
# Files copied from the base model dir into merged/, so convert_hf_to_gguf.py reads it
# exactly as it reads the original (it fingerprints the tokenizer).
BASE_SIDE_FILES = ("config.json", "generation_config.json", "tokenizer.json",
                   "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json",
                   "special_tokens_map.json", "chat_template.jinja")
STATE_NAME = "resume_state.json"


class ResumeError(Exception):
    """--resume refused: the saved state does not belong to this data or plan."""


# ---------------------------------------------------------------------------
# pure helpers (stdlib only; the tokenizer is passed in)
# ---------------------------------------------------------------------------

def split_of(example_id: str, dev_frac: float) -> str:
    """"dev" if sha256(id) % 10000 < dev_frac * 10000, else "train". Per example,
    independent of the rest of the file and of its order."""
    h = int(hashlib.sha256(str(example_id).encode()).hexdigest(), 16) % 10000
    return "dev" if h < dev_frac * 10000 else "train"


def build_example(tok, messages, max_len: int, end_marker: str = END_OF_TURN):
    """(input_ids, labels), or None if the example is longer than max_len tokens.

    prompt = the chat template of messages[:-1] with add_generation_prompt=True; the
    answer = messages[-1]["content"] + end_marker. They are tokenised SEPARATELY and
    concatenated, so the boundary is exact by construction and the prompt tokens are the
    ones the model sees at inference (no merge across the boundary). labels are -100 over
    the prompt and equal input_ids over the answer, end marker included."""
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("the last message must be the assistant answer")
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    prompt_ids = list(tok(prompt, add_special_tokens=False)["input_ids"])
    answer_ids = list(tok(messages[-1]["content"] + end_marker,
                          add_special_tokens=False)["input_ids"])
    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_len:
        return None
    return input_ids, [IGNORE] * len(prompt_ids) + answer_ids


def n_target_tokens(labels) -> int:
    """How many tokens the loss is taken over: labels[1:] != -100 (the logits at position
    t predict token t+1, so the first label can never be a target)."""
    return sum(1 for x in labels[1:] if x != IGNORE)


def read_examples(path) -> list[dict]:
    """The JSONL rows, validated: an id (unique), and messages ending in an assistant turn."""
    rows, seen = [], set()
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        msgs = row.get("messages")
        if "id" not in row or not isinstance(msgs, list) or len(msgs) < 2 \
                or msgs[-1].get("role") != "assistant" or not isinstance(msgs[-1].get("content"), str):
            raise ValueError(f"{path}:{n}: want {{id, messages: [..., assistant]}}")
        if row["id"] in seen:
            raise ValueError(f"{path}:{n}: duplicate id {row['id']!r}")
        seen.add(row["id"])
        rows.append(row)
    return rows


def prepare(tok, rows, max_len: int, dev_frac: float) -> dict:
    """Tokenise and split. {"train": [...], "dev": [...], "dropped": [ids]}; an example is
    {id, input_ids, labels, n_prompt, gold_refusal}. Over-long examples are dropped, not cut."""
    out = {"train": [], "dev": [], "dropped": []}
    for row in rows:
        built = build_example(tok, row["messages"], max_len)
        if built is None:
            out["dropped"].append(row["id"])
            continue
        input_ids, labels = built
        out[split_of(row["id"], dev_frac)].append({
            "id": row["id"], "input_ids": input_ids, "labels": labels,
            "n_prompt": sum(1 for x in labels if x == IGNORE),
            "gold_refusal": is_refusal(row["messages"][-1]["content"]),
        })
    return out


def is_refusal(text) -> bool:
    """True only for exactly the grammar's refusal (after strip)."""
    return isinstance(text, str) and text.strip() == grammar.REFUSAL


def refusal_metrics(pairs) -> dict:
    """pairs of (gold_is_refusal, predicted_is_refusal) -> {precision, recall, n}.
    Refusal is the positive class; a ratio with an empty denominator is None."""
    pairs = list(pairs)
    tp = sum(1 for g, p in pairs if g and p)
    fp = sum(1 for g, p in pairs if p and not g)
    fn = sum(1 for g, p in pairs if g and not p)
    return {"precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "n": len(pairs)}


def lora_target_names(module_names) -> list[str]:
    """Of the dotted names of nn.Linear modules, every attention (q/k/v/o_proj) and MLP
    (gate/up/down_proj) projection, in the order given. lm_head and the rest are left alone."""
    return [n for n in module_names if n.rsplit(".", 1)[-1] in LORA_TARGETS]


def step_plan(n_train: int, grad_accum: int, epochs: int, seed: int) -> list[list[int]]:
    """Per optimizer step, the indices of its examples: each epoch is a shuffle by
    Random(seed * 1000 + epoch) cut into chunks of grad_accum (the last may be short)."""
    plan: list[list[int]] = []
    for e in range(epochs):
        order = list(range(n_train))
        random.Random(seed * 1000 + e).shuffle(order)
        plan.extend(order[i:i + grad_accum] for i in range(0, n_train, grad_accum))
    return plan


def lr_factor(step: int, total: int, warmup_frac: float = WARMUP_FRAC) -> float:
    """Linear warmup over ceil(warmup_frac * total) steps, then cosine decay to 0."""
    warmup = max(1, math.ceil(warmup_frac * total))
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def make_resume_state(step: int, plan: dict, data_sha256: str) -> dict:
    """What a resume needs: step, epoch, rng position and the data hash. `plan` is
    {n_train, grad_accum, epochs, seed, total_steps}. The shuffle is a pure function of
    (seed, epoch), so the rng position is (seed, epoch, step_in_epoch)."""
    per_epoch = max(1, plan["total_steps"] // max(1, plan["epochs"]))
    return {"step": step, "epoch": step // per_epoch,
            "rng_position": {"seed": plan["seed"], "epoch": step // per_epoch,
                             "step_in_epoch": step % per_epoch},
            "data_sha256": data_sha256, "plan": dict(plan)}


def save_resume_state(path, state: dict) -> None:
    write_json(Path(path), state)


def load_resume_state(path, data_sha256: str, plan: dict) -> dict:
    """The saved state; ResumeError if its data hash or plan differs from this run's."""
    state = json.loads(Path(path).read_text())
    if state.get("data_sha256") != data_sha256:
        raise ResumeError(f"{path} was written for data sha256 {state.get('data_sha256')}, "
                          f"this --data is {data_sha256}")
    if state.get("plan") != plan:
        raise ResumeError(f"{path} was written for plan {state.get('plan')}, this run has {plan}")
    return state


def train_loop(trainer, start: int, total: int, *, stop_after_s: float = 0,
               clock=time.monotonic, log=print) -> dict:
    """Run steps start..total-1 on `trainer` (.step(i) -> loss, .save(step), .checkpoint(step)
    -> bool). After the step that crosses stop_after_s the trainer is saved and the loop
    returns status "paused" with exit_code 0 (the caller re-runs with --resume). Otherwise
    "complete", also exit_code 0; the final save is the caller's."""
    t0 = clock()
    step = start
    losses: list[tuple[int, float]] = []
    while step < total:
        loss = trainer.step(step)
        step += 1
        losses.append((step, loss))
        saved = False
        if step < total and trainer.checkpoint(step):
            trainer.save(step)
            saved = True
        if stop_after_s and step < total and clock() - t0 >= stop_after_s:
            if not saved:
                trainer.save(step)
            log(f"paused at step {step}/{total}: re-run with --resume")
            return {"status": "paused", "step": step, "exit_code": 0, "losses": losses}
    return {"status": "complete", "step": step, "exit_code": 0, "losses": losses}


# ---------------------------------------------------------------------------
# model (torch / transformers imported lazily; never run where this was written)
# ---------------------------------------------------------------------------

def load_tokenizer(model_dir: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir)


def load_model(model_dir: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to(device)
    model.config.use_cache = False
    return model


def _make_lora_cls():
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        """y = base(x) + dropout(x) @ A^T @ B^T * (alpha / rank); A, B kept in float32
        (copied from scripts/train_embedder.py, which cannot be imported without smm.embed)."""

        def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
            super().__init__()
            self.base = base
            self.scale = alpha / rank
            dev = base.weight.device
            self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=dev))
            self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            y = self.base(x)
            h = self.dropout(x).to(self.lora_A.dtype) @ self.lora_A.t()
            return y + ((h @ self.lora_B.t()) * self.scale).to(y.dtype)

    return LoRALinear


def apply_lora(model, rank: int, alpha: float, dropout: float) -> int:
    """Freeze the base and wrap every target nn.Linear (lora_target_names); returns how many."""
    from torch import nn
    cls = _make_lora_cls()
    for p in model.parameters():
        p.requires_grad_(False)
    modules = dict(model.named_modules())
    names = lora_target_names(n for n, m in modules.items() if isinstance(m, nn.Linear))
    if not names:
        raise RuntimeError("no target projections found: is this a Qwen3 model?")
    for name in names:
        parent_name, _, child = name.rpartition(".")
        setattr(modules[parent_name], child, cls(modules[name], rank, alpha, dropout))
    return len(names)


def lora_params(model) -> dict:
    return {n: p for n, p in model.named_parameters() if "lora_" in n}


def merge_lora(model) -> int:
    """W += (B @ A) * alpha/rank in float32, cast back to the weight dtype; each LoRA
    wrapper is replaced by its (now merged) base Linear. Returns the count."""
    import torch
    n = 0
    for _, parent in list(model.named_modules()):
        for name, child in list(parent.named_children()):
            if hasattr(child, "lora_A") and hasattr(child, "lora_B") and hasattr(child, "base"):
                base = child.base
                with torch.no_grad():
                    delta = (child.lora_B.float() @ child.lora_A.float()) * child.scale
                    base.weight.copy_((base.weight.float() + delta).to(base.weight.dtype))
                setattr(parent, name, base)
                n += 1
    return n


def load_adapter(model, path: Path) -> None:
    import torch
    saved = torch.load(path, map_location="cpu", weights_only=True)
    params = lora_params(model)
    if set(params) != set(saved):
        raise SystemExit(f"{path}: LoRA weights do not match this model/--lora-rank")
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(saved[n].to(p.device))


def answer_loss_sum(model, example: dict, device: str):
    """(sum of token cross-entropy over the answer, its token count) for one example.
    Only the answer positions go through lm_head: the vocabulary is 152k wide and the
    prompt is thousands of tokens, so full logits would cost gigabytes for nothing."""
    import torch
    ids = torch.tensor([example["input_ids"]], device=device)
    labels = torch.tensor([example["labels"]], device=device)
    hidden = model.model(input_ids=ids).last_hidden_state  # (1, T, H); batch of one, no padding
    target = labels[:, 1:]
    keep = target != IGNORE
    logits = model.lm_head(hidden[:, :-1][keep]).float()
    loss = torch.nn.functional.cross_entropy(logits, target[keep], reduction="sum")
    return loss, int(keep.sum())


def dev_loss(model, dev: list[dict], device: str) -> dict:
    """Mean token cross-entropy over the answer tokens of `dev`."""
    import torch
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for ex in dev:
            s, k = answer_loss_sum(model, ex, device)
            total += s.item()
            n += k
    if was_training:
        model.train()
    return {"loss": total / n if n else None, "n_examples": len(dev), "n_tokens": n}


def generate_answer(model, tok, prompt_ids: list[int], max_new: int, device: str) -> str:
    import torch
    ids = torch.tensor([prompt_ids], device=device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=max_new, do_sample=False, use_cache=True,
                             pad_token_id=pad)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def dev_refusals(model, tok, dev: list[dict], n: int, max_new: int, device: str) -> dict:
    """Refusal precision/recall of greedy answers on the first n dev examples (by id)."""
    was_training = model.training
    model.eval()
    subset = sorted(dev, key=lambda e: e["id"])[:n]
    pairs = [(ex["gold_refusal"],
              is_refusal(generate_answer(model, tok, ex["input_ids"][:ex["n_prompt"]],
                                         max_new, device)))
             for ex in subset]
    if was_training:
        model.train()
    res = refusal_metrics(pairs)
    res.update(gold_refusals=sum(1 for g, _ in pairs if g),
               predicted_refusals=sum(1 for _, p in pairs if p))
    return res


def evaluate(model, tok, dev, args) -> dict:
    res = {"dev_loss": dev_loss(model, dev[:args.dev_loss_max], args.device)}
    if args.gen_dev:
        res["refusal"] = dev_refusals(model, tok, dev, args.gen_dev, args.gen_max_new, args.device)
    return res


class TorchTrainer:
    """Adapts the real model to train_loop: .step(i), .save(step), .checkpoint(step)."""

    def __init__(self, model, tok, train, dev, plan_idx, plan, data_sha, out: Path, args):
        import torch
        self.model, self.tok, self.train, self.dev = model, tok, train, dev
        self.plan_idx, self.plan, self.data_sha, self.out, self.args = plan_idx, plan, data_sha, out, args
        self.params = list(lora_params(model).values())
        total = len(plan_idx)
        self.opt = torch.optim.AdamW(self.params, lr=args.lr, weight_decay=0.0)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: lr_factor(s, total))
        self.dev_curve: list[list] = []

    def checkpoint(self, step: int) -> bool:
        return step % self.args.checkpoint_every == 0

    def step(self, i: int) -> float:
        import torch
        idxs = self.plan_idx[i]
        n_tok = max(1, sum(n_target_tokens(self.train[j]["labels"]) for j in idxs))
        total = 0.0
        for j in idxs:
            s, _ = answer_loss_sum(self.model, self.train[j], self.args.device)
            (s / n_tok).backward()  # each token weighs 1 / (tokens in this step)
            total += s.item()
        torch.nn.utils.clip_grad_norm_(self.params, self.args.max_grad_norm)
        self.opt.step()
        self.sched.step()
        self.opt.zero_grad(set_to_none=True)
        done = i + 1
        if self.args.eval_every and done % self.args.eval_every == 0 and self.dev:
            r = dev_loss(self.model, self.dev[:self.args.dev_loss_max], self.args.device)
            self.dev_curve.append([done, r["loss"]])
            print(f"  step {done}: dev loss {r['loss']:.4f}", flush=True)
        return total / n_tok

    def save(self, step: int) -> None:
        import torch
        self.out.mkdir(parents=True, exist_ok=True)
        lora = {n: p.detach().cpu() for n, p in lora_params(self.model).items()}
        state = {"lora": lora, "optimizer": self.opt.state_dict(),
                 "scheduler": self.sched.state_dict(), "step": step,
                 "rng": {"python": random.getstate(), "torch": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
        for name, obj in (("ckpt.pt", state), ("adapter.pt", lora)):
            tmp = self.out / (name + ".tmp")
            torch.save(obj, tmp)
            os.replace(tmp, self.out / name)
        # the JSON goes last: it is the commit marker a resume trusts
        save_resume_state(self.out / STATE_NAME, make_resume_state(step, self.plan, self.data_sha))

    def restore(self, step: int) -> None:
        import torch
        state = torch.load(self.out / "ckpt.pt", map_location="cpu", weights_only=False)
        if state["step"] != step:
            raise SystemExit(f"ckpt.pt is at step {state['step']} but {STATE_NAME} says {step}")
        params = lora_params(self.model)
        if set(params) != set(state["lora"]):
            raise SystemExit("checkpoint LoRA weights do not match this model/--lora-rank")
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(state["lora"][n].to(p.device))
        self.opt.load_state_dict(state["optimizer"])
        self.sched.load_state_dict(state["scheduler"])
        random.setstate(state["rng"]["python"])
        torch.set_rng_state(state["rng"]["torch"].cpu())
        if state["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["rng"]["cuda"]])


def save_merged(model, tok, model_dir: str, out_dir: Path) -> None:
    """Merged weights (safetensors) + tokenizer + the base model's config and tokenizer
    files, so convert_hf_to_gguf.py reads it like the original."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    for name in BASE_SIDE_FILES:
        if (Path(model_dir) / name).exists():
            shutil.copy2(Path(model_dir) / name, out_dir / name)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="chat JSONL from build_raft_data.py")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dev-frac", type=float, default=0.02,
                    help="share of examples (hash of the id) held out as dev")
    ap.add_argument("--max-len", type=int, default=3072,
                    help="examples tokenising longer than this are dropped, not truncated")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16,
                    help="examples per optimizer step (micro-batch is always 1)")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--eval-every", type=int, default=100, help="dev loss every N steps (0: never)")
    ap.add_argument("--dev-loss-max", type=int, default=200, help="dev examples used for dev loss")
    ap.add_argument("--gen-dev", type=int, default=64,
                    help="dev examples answered greedily for refusal precision/recall (0: skip)")
    ap.add_argument("--gen-max-new", type=int, default=200)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after-s", type=float, default=0,
                    help="after the step that crosses this many seconds: save, exit 0")
    ap.add_argument("--eval-only", action="store_true",
                    help="dev loss and refusal P/R of the untouched base model; no training")
    ap.add_argument("--merge-only", action="store_true",
                    help="merge --out/adapter.pt into --out/merged; no training")
    return ap.parse_args(argv)


def _load_prior_log(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.eval_only and args.merge_only:
        print("--eval-only and --merge-only are exclusive", file=sys.stderr)
        return 2
    data_path = ROOT / args.data
    out = ROOT / args.out
    model_dir = str(ROOT / args.model_dir)
    state_path = out / STATE_NAME
    training = not (args.eval_only or args.merge_only)
    if training and state_path.exists() and not args.resume:
        print(f"{state_path} exists: pass --resume to continue it, or use a fresh --out",
              file=sys.stderr)
        return 2
    if args.resume and not state_path.exists():
        print(f"--resume but no {state_path}", file=sys.stderr)
        return 2

    data_sha = file_sha256(data_path)
    tok = load_tokenizer(model_dir)
    data = prepare(tok, read_examples(data_path), args.max_len, args.dev_frac)
    train, dev = data["train"], data["dev"]
    if not train:
        print(f"no training examples ({len(dev)} dev, {len(data['dropped'])} dropped)", file=sys.stderr)
        return 2
    print(f"train {len(train)} / dev {len(dev)}; dropped over {args.max_len} tokens: "
          f"{len(data['dropped'])}", flush=True)

    plan_idx = step_plan(len(train), args.grad_accum, args.epochs, args.seed)
    plan = {"n_train": len(train), "grad_accum": args.grad_accum, "epochs": args.epochs,
            "seed": args.seed, "total_steps": len(plan_idx)}
    log_path = out / "train_log.json"
    log = _load_prior_log(log_path) if args.resume else {}
    log.update(hyperparameters=vars(args), data_sha256=data_sha,
               counts={"train": len(train), "dev": len(dev), "dropped_over_max_len": len(data["dropped"]),
                       "dropped_ids": data["dropped"][:50], "steps": plan["total_steps"]})

    import torch
    model = load_model(model_dir, args.device)
    t_start = time.monotonic()

    if args.eval_only:
        model.eval()
        log.update(mode="eval-only", dev_base=evaluate(model, tok, dev, args))
        print("dev (base):", log["dev_base"], flush=True)
        write_json(log_path, log)
        return 0

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    n_wrapped = apply_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
    log["counts"]["lora_modules"] = n_wrapped

    if args.merge_only:
        load_adapter(model, out / "adapter.pt")
        merged = merge_lora(model)
        save_merged(model, tok, model_dir, out / "merged")
        log.update(mode="merge-only", merged_modules=merged)
        write_json(log_path, log)
        print(f"merged {merged} modules -> {out / 'merged'}", flush=True)
        return 0

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    trainer = TorchTrainer(model, tok, train, dev, plan_idx, plan, data_sha, out, args)
    print(f"{plan['total_steps']} steps x up to {args.grad_accum} examples; {n_wrapped} LoRA modules, "
          f"{sum(p.numel() for p in trainer.params):,} trainable parameters", flush=True)

    start = 0
    before_path = out / "dev_before.json"
    if args.resume:
        try:
            state = load_resume_state(state_path, data_sha, plan)
        except ResumeError as e:
            print(f"refusing to resume: {e}", file=sys.stderr)
            return 2
        start = state["step"]
        trainer.restore(start)
        log["dev_before"] = json.loads(before_path.read_text()) if before_path.exists() else None
        print(f"resumed at step {start}", flush=True)
    else:
        log["dev_before"] = evaluate(model, tok, dev, args)
        write_json(before_path, log["dev_before"])
        print("dev (before):", log["dev_before"], flush=True)
        model.train()

    result = train_loop(trainer, start, plan["total_steps"], stop_after_s=args.stop_after_s)
    curve = log.get("loss_curve", []) + [[s, l] for s, l in result["losses"]]
    log.update(step=result["step"], loss_curve=curve,
               dev_loss_curve=log.get("dev_loss_curve", []) + trainer.dev_curve)
    log.setdefault("runs", []).append({"from_step": start, "to_step": result["step"],
                                       "seconds": time.monotonic() - t_start})
    if result["status"] == "paused":
        log["status"] = "paused"
        write_json(log_path, log)
        return result["exit_code"]

    trainer.save(result["step"])
    log["dev_after"] = evaluate(model, tok, dev, args)
    print("dev (after):", log["dev_after"], flush=True)
    merged = merge_lora(model)
    save_merged(model, tok, model_dir, out / "merged")
    log.update(status="complete", merged_modules=merged)
    write_json(log_path, log)
    print(f"merged {merged} modules -> {out / 'merged'}", flush=True)
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
