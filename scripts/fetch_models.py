#!/usr/bin/env python3
"""Fetch the GGUF models this project runs on, into models/.

English-only corpus, so the embedder is Qwen3-Embedding-0.6B rather than the
multilingual bge-m3 the briefing assumed.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

MODELS = {
    "embedder": ("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf"),
    "generator": ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
    "reranker": ("ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF", "qwen3-reranker-0.6b-q8_0.gguf"),
}
DEST = Path(__file__).resolve().parent.parent / "models"


def fetch(repo: str, filename: str) -> Path:
    out = DEST / filename
    if out.exists():
        print(f"  have {filename} ({out.stat().st_size/1e6:.0f} MB)")
        return out
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    tmp = out.with_suffix(out.suffix + ".part")
    print(f"  get  {filename}")
    with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as fh:
        total = int(r.headers.get("content-length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r       {done/1e6:7.0f}/{total/1e6:.0f} MB  {done/total:5.1%}", end="", flush=True)
    print()
    tmp.rename(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roles", nargs="*", metavar="ROLE",
                    help=f"which models to fetch: {', '.join(MODELS)} (default: all)")
    args = ap.parse_args()
    unknown = [r for r in args.roles if r not in MODELS]
    if unknown:
        ap.error(f"unknown role(s): {', '.join(unknown)}; choose from {', '.join(MODELS)}")

    DEST.mkdir(parents=True, exist_ok=True)
    want = args.roles or list(MODELS)
    for role in want:
        repo, filename = MODELS[role]
        print(f"{role}:")
        fetch(repo, filename)
    print("\nmodels/")
    for f in sorted(DEST.glob("*.gguf")):
        print(f"  {f.stat().st_size/1e6:8.0f} MB  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
