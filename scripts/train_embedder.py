#!/usr/bin/env python3
"""Phase 15 R13: fine-tune Qwen3-Embedding-0.6B on (generated question, chunk) pairs.

Hand-written LoRA (no peft) on every attention and MLP projection, in-batch-negative
InfoNCE (query -> chunk), a hash-chosen dev split of CHUNKS (so no chunk is both
trained on and held out), dev recall before and after, and a merged HF model dir that
llama.cpp's convert_hf_to_gguf.py can read.

Training-only: run under the training venv (requirements-train.txt), never the
retrieval .venv. torch and transformers are imported inside the functions that need
them, so `import train_embedder` works without either (tests/test_train_embedder.py).

  .venv-train/bin/python scripts/train_embedder.py --eval-only --out /tmp/base-eval \\
      --questions data/index/qvec-emacs.json data/index/qvec-man.json
  .venv-train/bin/python scripts/train_embedder.py --out data/index/r13 \\
      --questions data/index/qvec-emacs.json data/index/qvec-man.json
  ... --resume --stop-after-s 6600     # stages under a 2 h job cap: save, exit 0, resume

--eval-only writes its result to --out/train_log.json too: give it its own --out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smm.embed import query_text as _runtime_query_text  # noqa: E402  (never copy the instruction)

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
EOS_TOKEN = "<|endoftext|>"  # appended by the HF tokenizer; its hidden state is the embedding
# Files a sentence-transformers/llama.cpp conversion reads next to the weights.
BASE_SIDE_FILES = ("config.json", "generation_config.json", "modules.json",
                   "config_sentence_transformers.json")


# ---------------------------------------------------------------------------
# pure helpers (stdlib only)
# ---------------------------------------------------------------------------

def query_text(q: str) -> str:
    """The query exactly as the runtime embeds it (smm.embed.query_text)."""
    return _runtime_query_text(q)


def doc_text(prefix: str, text: str) -> str:
    """The chunk exactly as the runtime embeds it (Chunk.embed_text)."""
    return f"{prefix}{text}"


def _db_chunk_ids(db_path) -> set[str]:
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return {r[0] for r in db.execute("SELECT chunk_id FROM chunks")}
    finally:
        db.close()


def load_pairs(db_path, question_files) -> tuple[list[tuple[str, str]], int]:
    """(pairs, skipped): every (chunk_id, question) from JSON caches {chunk_id: [q, ...]}
    whose chunk is in the db, in file then cache order, exact duplicates dropped.
    `skipped` counts the cache entries (chunks) that are absent from the db."""
    known = _db_chunk_ids(db_path)
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    skipped = 0
    for f in question_files:
        cache = json.loads(Path(f).read_text())
        for cid, qs in cache.items():
            if cid not in known:
                skipped += 1
                continue
            for q in qs:
                if isinstance(q, str) and q.strip() and (cid, q) not in seen:
                    seen.add((cid, q))
                    pairs.append((cid, q))
    return pairs, skipped


def is_dev(chunk_id: str, seed: int, dev_frac: float) -> bool:
    """Stable per-chunk coin: sha256(f"{seed}:{chunk_id}") below dev_frac."""
    h = hashlib.sha256(f"{seed}:{chunk_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64 < dev_frac


def holdout_split(pairs, seed: int, dev_frac: float):
    """(train_pairs, dev_pairs): the split is by chunk, so the two never share one."""
    train, dev = [], []
    for p in pairs:
        (dev if is_dev(p[0], seed, dev_frac) else train).append(p)
    return train, dev


def batches(pairs, size: int, rng: random.Random) -> list[list[tuple[str, str]]]:
    """Shuffled batches of `size` in which no chunk appears twice (two questions of one
    chunk in a batch would be each other's false negative). A pair whose chunk is
    already in the open batch is deferred to the next one. The last batches may be short."""
    order = list(pairs)
    rng.shuffle(order)
    out: list[list[tuple[str, str]]] = []
    deferred: list[tuple[str, str]] = []
    batch: list[tuple[str, str]] = []
    in_batch: set[str] = set()

    def flush():
        nonlocal batch, in_batch, deferred
        out.append(batch)
        batch, in_batch = [], set()
        pending, deferred = deferred, []
        for p in pending:
            offer(p)

    def offer(p):
        if p[0] in in_batch:
            deferred.append(p)
            return
        batch.append(p)
        in_batch.add(p[0])
        if len(batch) == size:
            flush()

    for p in order:
        offer(p)
    while batch or deferred:
        if not batch:  # a deferred pair always fits an empty batch
            pending, deferred = deferred, []
            for p in pending:
                offer(p)
            continue
        flush()
    return out


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(s.lower()))


def leakage(train_questions, eval_questions) -> float:
    """Max token Jaccard between any eval question and any training question
    (lower-cased [a-z0-9]+ tokens). 1.0 = an eval question appears verbatim."""
    train = {_tokens(q) for q in train_questions}
    train.discard(frozenset())
    postings: dict[str, list[frozenset]] = {}
    for t in train:
        for w in t:
            postings.setdefault(w, []).append(t)
    best = 0.0
    for q in eval_questions:
        e = _tokens(q)
        if not e:
            continue
        seen: set[frozenset] = set()
        for w in e:
            for t in postings.get(w, ()):
                if t in seen:
                    continue
                seen.add(t)
                j = len(e & t) / len(e | t)
                if j > best:
                    best = j
    return best


def read_questions_file(path) -> list[str]:
    """Eval questions for the leakage check: a JSON list of strings or of {"question":..}
    objects, or a JSONL file of such objects."""
    text = Path(path).read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(l) for l in text.splitlines() if l.strip()]
    return [d if isinstance(d, str) else d["question"] for d in data]


def recall_at(ranks, ks=(1, 10)) -> dict:
    """ranks: 0-based rank of the gold chunk per query."""
    n = len(ranks)
    return {f"recall@{k}": (sum(1 for r in ranks if r < k) / n if n else 0.0) for k in ks}


def lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warmup over `warmup` steps, then linear decay to 0 at `total`."""
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_chunk_texts(db_path, chunk_ids) -> dict[str, str]:
    """chunk_id -> doc_text(prefix, text) for the given ids."""
    ids = set(chunk_ids)
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return {cid: doc_text(pre, txt)
                for cid, pre, txt in db.execute("SELECT chunk_id, prefix, text FROM chunks")
                if cid in ids}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# model (torch / transformers imported lazily)
# ---------------------------------------------------------------------------

def load_model(base_model: str, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    model = AutoModel.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device)
    model.config.use_cache = False
    return model, tok


def pool_last(hidden, attention_mask=None):
    """L2-normalised hidden state of the LAST position (the tokenizer pads on the left,
    so the last position is always the appended <|endoftext|>), in float32."""
    import torch
    v = hidden[:, -1].float()
    return torch.nn.functional.normalize(v, dim=-1)


def encode(model, tok, texts, max_len: int, device: str):
    """Normalised (len(texts), dim) float32 embeddings, with grad if the model is in
    train mode and the LoRA weights require it."""
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    eos = tok.convert_tokens_to_ids(EOS_TOKEN)
    if not bool((enc["input_ids"][:, -1] == eos).all()):
        raise RuntimeError("tokenizer did not end every sequence with <|endoftext|>: "
                           "last-position pooling would be wrong")
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    return pool_last(out.last_hidden_state, enc["attention_mask"])


def _make_lora_cls():
    import math
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        """y = base(x) + dropout(x) @ A^T @ B^T * (alpha / rank); A, B kept in float32."""

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


def apply_lora(model, rank: int, alpha: float, dropout: float, targets=LORA_TARGETS) -> int:
    """Freeze the base and wrap every target nn.Linear; returns how many were wrapped."""
    from torch import nn
    cls = _make_lora_cls()
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for _, parent in list(model.named_modules()):
        for name, child in list(parent.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(parent, name, cls(child, rank, alpha, dropout))
                n += 1
    if n == 0:
        raise RuntimeError("no target projections found: is this a Qwen3 model?")
    return n


def lora_params(model) -> dict:
    return {n: p for n, p in model.named_parameters() if "lora_" in n}


def merge_lora(model) -> int:
    """W += (B @ A) * alpha/rank in float32, cast back to the weight dtype; the LoRA
    wrappers are replaced by their (now merged) base Linear. Returns the count."""
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


def dev_recall(model, tok, dev_pairs, texts, extra_ids, args, device) -> dict:
    """For each held-out question, the rank of its chunk among the held-out chunks plus
    `extra_ids` training chunks; recall@1 and recall@10."""
    import torch
    was_training = model.training
    model.eval()
    dev_ids = sorted({c for c, _ in dev_pairs})
    dev_set = set(dev_ids)
    doc_ids = dev_ids + [c for c in extra_ids if c not in dev_set]
    index = {c: i for i, c in enumerate(doc_ids)}
    bs = 64

    def embed_all(strings, max_len):
        order = sorted(range(len(strings)), key=lambda i: len(strings[i]), reverse=True)
        out = [None] * len(strings)
        with torch.no_grad():
            for s in range(0, len(order), bs):
                idx = order[s:s + bs]
                vecs = encode(model, tok, [strings[i] for i in idx], max_len, device)
                for i, v in zip(idx, vecs):
                    out[i] = v
        return torch.stack(out)

    D = embed_all([texts[c] for c in doc_ids], args.max_doc_tokens)
    Q = embed_all([query_text(q) for _, q in dev_pairs], args.max_query_tokens)
    gold = torch.tensor([index[c] for c, _ in dev_pairs], device=D.device)
    ranks: list[int] = []
    with torch.no_grad():
        for s in range(0, len(dev_pairs), 1024):
            sc = Q[s:s + 1024] @ D.t()
            g = sc.gather(1, gold[s:s + 1024, None])
            ranks.extend((sc > g).sum(1).tolist())
    if was_training:
        model.train()
    res = recall_at(ranks)
    res.update(n_queries=len(dev_pairs), n_dev_chunks=len(dev_ids), n_docs=len(doc_ids))
    return res


def save_checkpoint(path: Path, model, opt, sched, step: int, meta: dict) -> None:
    import torch
    state = {
        "lora": {n: p.detach().cpu() for n, p in lora_params(model).items()},
        "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "step": step,
        "meta": meta,
        "rng": {"python": random.getstate(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, model, opt, sched, device: str) -> tuple[int, dict]:
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)
    params = lora_params(model)
    if set(params) != set(state["lora"]):
        raise SystemExit("checkpoint LoRA weights do not match this model/--rank")
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(state["lora"][n].to(p.device))
    opt.load_state_dict(state["optimizer"])
    sched.load_state_dict(state["scheduler"])
    random.setstate(state["rng"]["python"])
    torch.set_rng_state(state["rng"]["torch"].cpu())
    if state["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["rng"]["cuda"]])
    return state["step"], state["meta"]


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def save_merged(model, tok, base_model: str, out_dir: Path) -> None:
    """Merged weights (safetensors) + tokenizer + the base model's config and
    sentence-transformers files, so convert_hf_to_gguf.py reads it like the original."""
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    base = Path(base_model)
    for name in BASE_SIDE_FILES:
        if (base / name).exists():
            shutil.copy2(base / name, out_dir / name)
    if (base / "1_Pooling").is_dir():
        shutil.copytree(base / "1_Pooling", out_dir / "1_Pooling", dirs_exist_ok=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default="data/index/phase11.db", help="index with the chunks")
    ap.add_argument("--questions", nargs="+", required=True,
                    help="question caches {chunk_id: [question, ...]} from build_qvec.py")
    ap.add_argument("--base-model", default="models/hf/Qwen3-Embedding-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-query-tokens", type=int, default=64)
    ap.add_argument("--max-doc-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--dev-frac", type=float, default=0.05,
                    help="share of chunks (hash-chosen) held out for the dev metric")
    ap.add_argument("--dev-extra", type=int, default=5000,
                    help="seed-chosen training chunks added to the dev candidate pool")
    ap.add_argument("--leakage-against", default=None,
                    help="eval questions file (JSON/JSONL): log max token Jaccard vs training questions")
    ap.add_argument("--leakage-max-eval", type=int, default=300)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after-s", type=float, default=0,
                    help="after this many seconds: save the checkpoint and exit 0")
    ap.add_argument("--eval-only", action="store_true",
                    help="dev recall of the untouched base model; no training")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    db_path = ROOT / args.db
    out = ROOT / args.out
    base_model = str(ROOT / args.base_model)
    qfiles = [ROOT / q for q in args.questions]
    ckpt = out / "ckpt" / "ckpt.pt"
    if ckpt.exists() and not args.resume and not args.eval_only:
        print(f"{ckpt} exists: pass --resume to continue it, or use a fresh --out",
              file=sys.stderr)
        return 2
    if args.resume and not ckpt.exists():
        print(f"--resume but no checkpoint at {ckpt}", file=sys.stderr)
        return 2

    pairs, skipped = load_pairs(db_path, qfiles)
    train_pairs, dev_pairs = holdout_split(pairs, args.seed, args.dev_frac)
    if not dev_pairs or not train_pairs:
        print(f"empty split ({len(train_pairs)} train / {len(dev_pairs)} dev pairs)",
              file=sys.stderr)
        return 2
    train_chunks = sorted({c for c, _ in train_pairs})
    extra = random.Random(args.seed).sample(train_chunks, min(args.dev_extra, len(train_chunks)))
    texts = load_chunk_texts(db_path, {c for c, _ in pairs})
    print(f"pairs {len(pairs)} (cache chunks not in db: {skipped}); train {len(train_pairs)} / "
          f"dev {len(dev_pairs)}; dev chunks {len({c for c, _ in dev_pairs})}", flush=True)

    log = {
        "hyperparameters": {k: v for k, v in vars(args).items()},
        "counts": {"pairs": len(pairs), "cache_chunks_not_in_db": skipped,
                   "train_pairs": len(train_pairs), "dev_pairs": len(dev_pairs),
                   "train_chunks": len(train_chunks),
                   "dev_chunks": len({c for c, _ in dev_pairs}), "dev_pool_extra": len(extra)},
        "input_sha256": {"db": file_sha256(db_path),
                         **{str(q.name): file_sha256(q) for q in qfiles}},
    }
    if args.leakage_against:
        ev = read_questions_file(ROOT / args.leakage_against)[:args.leakage_max_eval]
        log["leakage"] = {"file": args.leakage_against, "n_eval": len(ev),
                          "max_token_jaccard": leakage([q for _, q in train_pairs], ev)}
        print(f"leakage vs {args.leakage_against}: {log['leakage']['max_token_jaccard']:.3f}",
              flush=True)

    import torch
    model, tok = load_model(base_model, args.device)
    t_start = time.time()

    if args.eval_only:
        model.eval()
        log["mode"] = "eval-only"
        log["dev_base"] = dev_recall(model, tok, dev_pairs, texts, extra, args, args.device)
        print("dev (base):", log["dev_base"], flush=True)
        write_json(out / "train_log.json", log)
        return 0

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    n_wrapped = apply_lora(model, args.rank, args.alpha, args.dropout)
    log["counts"]["lora_modules"] = n_wrapped
    params = list(lora_params(model).values())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    all_batches: list[list] = []
    dropped = 0
    for e in range(args.epochs):
        for b in batches(train_pairs, args.batch, random.Random(args.seed * 1000 + e)):
            if len(b) >= 2:
                all_batches.append(b)
            else:
                dropped += len(b)
    total = len(all_batches)
    log["counts"].update(steps=total, dropped_singleton_pairs=dropped)
    print(f"{total} steps of up to {args.batch}; {n_wrapped} LoRA modules, "
          f"{sum(p.numel() for p in params):,} trainable parameters", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, args.warmup, total))
    step = 0
    dev_before_path = out / "dev_before.json"
    if args.resume:
        step, meta = load_checkpoint(ckpt, model, opt, sched, args.device)
        if meta.get("steps") != total:
            raise SystemExit(f"checkpoint was for {meta.get('steps')} steps, this run has {total}")
        log["dev_before"] = (json.loads(dev_before_path.read_text())
                             if dev_before_path.exists() else None)
        print(f"resumed at step {step}", flush=True)
    else:
        log["dev_before"] = dev_recall(model, tok, dev_pairs, texts, extra, args, args.device)
        write_json(dev_before_path, log["dev_before"])
        print("dev (before):", log["dev_before"], flush=True)
        model.train()
    meta = {"steps": total, "rank": args.rank, "seed": args.seed}

    losses: list[float] = []
    loss_curve: list[list] = []
    t0 = time.time()
    while step < total:
        b = all_batches[step]
        q = encode(model, tok, [query_text(x) for _, x in b], args.max_query_tokens, args.device)
        d = encode(model, tok, [texts[c] for c, _ in b], args.max_doc_tokens, args.device)
        logits = q @ d.t() / args.temperature
        loss = torch.nn.functional.cross_entropy(
            logits, torch.arange(len(b), device=logits.device))
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        losses.append(loss.item())
        if step % 10 == 0 or step == total:
            mean = sum(losses) / len(losses)
            loss_curve.append([step, mean])
            print(f"  step {step}/{total}  loss {mean:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{(time.time() - t0) / len(losses):.2f} s/step", flush=True)
            losses = []
        if step % args.checkpoint_every == 0 and step < total:
            save_checkpoint(ckpt, model, opt, sched, step, meta)
        if args.stop_after_s and time.time() - t0 > args.stop_after_s and step < total:
            save_checkpoint(ckpt, model, opt, sched, step, meta)
            log.update(status="paused", step=step, loss_curve=loss_curve)
            write_json(out / "train_log.json", log)
            print(f"paused at step {step}/{total}: rerun with --resume", flush=True)
            return 0

    save_checkpoint(ckpt, model, opt, sched, step, meta)
    log["dev_after"] = dev_recall(model, tok, dev_pairs, texts, extra, args, args.device)
    print("dev (after):", log["dev_after"], flush=True)
    merged = merge_lora(model)
    save_merged(model, tok, base_model, out / "merged")
    log.update(status="complete", step=step, merged_modules=merged, loss_curve=loss_curve,
               seconds_this_run=time.time() - t_start)
    write_json(out / "train_log.json", log)
    print(f"merged {merged} modules -> {out / 'merged'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
