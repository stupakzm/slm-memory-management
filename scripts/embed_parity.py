#!/usr/bin/env python3
"""Phase 15 R13: check a llama-server embedder against its HF source model.

Embeds the same sampled chunk texts (document side) and queries with the HF
model dir (AutoModel, left padding, last-position pooling, L2-normalised; the
HF tokenizer appends <|endoftext|> itself) and with a llama-server's
/v1/embeddings, then reports min/mean cosine for queries and docs separately.
Exit 1 if either minimum is below --min-cos.

  .venv-train/bin/python scripts/embed_parity.py --hf models/hf-ft --url http://127.0.0.1:8081
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def cosine(a, b) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if not na or not nb:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def parity_report(hf: dict, srv: dict, min_cos: float) -> tuple[dict, bool]:
    """hf/srv: {"docs": [vec], "queries": [vec]}. Returns (stats, ok).

    stats[kind] = {"n", "min", "mean"}; ok is False if any kind's min < min_cos
    (or a kind has no vectors / mismatched lengths).
    """
    stats, ok = {}, True
    for kind in ("queries", "docs"):
        a, b = hf[kind], srv[kind]
        if not a or len(a) != len(b):
            stats[kind] = {"n": min(len(a), len(b)), "min": float("nan"), "mean": float("nan")}
            ok = False
            continue
        cs = [cosine(x, y) for x, y in zip(a, b)]
        stats[kind] = {"n": len(cs), "min": min(cs), "mean": sum(cs) / len(cs)}
        if stats[kind]["min"] < min_cos:
            ok = False
    return stats, ok


def format_report(stats: dict, min_cos: float, ok: bool) -> str:
    lines = [f"{k:8s} n={s['n']:4d}  min cos {s['min']:.5f}  mean cos {s['mean']:.5f}"
             for k, s in stats.items()]
    lines.append(f"threshold {min_cos}: {'PASS' if ok else 'FAIL'}")
    return "\n".join(lines)


def sample_texts(db_path: Path, n: int, seed: int, questions: Path | None):
    from smm.embed import query_text
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = db.execute("SELECT chunk_id, prefix, text FROM chunks ORDER BY rowid").fetchall()
    finally:
        db.close()
    rng = random.Random(seed)
    picked = rng.sample(rows, min(n, len(rows)))
    docs = [f"{p}{t}" if p else t for _, p, t in picked]
    if questions:
        cache = json.loads(questions.read_text())
        qs = [q for v in cache.values() for q in v]
        qs = rng.sample(qs, min(n, len(qs)))
    else:
        qs = [(t.strip().splitlines() or [""])[0][:200] for _, _, t in picked]
    return docs, [query_text(q) for q in qs]


def hf_embed(model_dir: Path, texts: list[str], batch: int = 8) -> list[list[float]]:
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=8192,
                      return_tensors="pt")
            h = model(**enc).last_hidden_state[:, -1]
            h = torch.nn.functional.normalize(h, dim=-1)
            out.extend(h.tolist())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hf", type=Path, required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--db", type=Path, default=Path("data/index/phase11.db"))
    ap.add_argument("--questions", type=Path, default=None)
    ap.add_argument("--min-cos", type=float, default=0.99)
    a = ap.parse_args()

    from smm.embed import Embedder
    docs, queries = sample_texts(a.db, a.n, a.seed, a.questions)
    emb = Embedder(a.url)
    srv = {"docs": [], "queries": []}
    for kind, texts in (("docs", docs), ("queries", queries)):
        for i in range(0, len(texts), 16):
            srv[kind].extend(emb.embed(texts[i:i + 16]))
    hf = {"docs": hf_embed(a.hf, docs), "queries": hf_embed(a.hf, queries)}
    stats, ok = parity_report(hf, srv, a.min_cos)
    print(format_report(stats, a.min_cos, ok))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
