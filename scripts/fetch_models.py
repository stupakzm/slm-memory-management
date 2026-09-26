#!/usr/bin/env python3
"""Fetch the GGUF models this project runs on, into models/.

English-only corpus, so the embedder is Qwen3-Embedding-0.6B rather than the
multilingual bge-m3 the briefing assumed.

reranker-4b (pyarn/Qwen3-Reranker-4B-Q4_K_M-GGUF) is opt-in, not part of
DEFAULT_ROLES: at query-sized settings it alone runs ~4.2 GB, so on a 6 GB
card it cannot share with the generator the way the 0.6B reranker does, and a
phase using it has to run it as a staged, single-model swap rather than as
part of the regular three-way `serve` mix. It's pinned to this specific
conversion (by sha256) because a header check across ten public 4B GGUF
conversions found it is the only one that ships the `cls.output.weight` head
and rank pooling; the other nine lack the head and would give meaningless
rerank scores despite loading and serving without error.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

# role -> (hf repo, filename, sha256 or None). A pinned hash is verified both
# after a fresh download and against a file that already exists on disk.
MODELS = {
    "embedder": ("Qwen/Qwen3-Embedding-0.6B-GGUF", "Qwen3-Embedding-0.6B-Q8_0.gguf", None),
    "generator": ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", None),
    "reranker": ("ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF", "qwen3-reranker-0.6b-q8_0.gguf", None),
    "reranker-4b": (
        "pyarn/Qwen3-Reranker-4B-Q4_K_M-GGUF",
        "qwen3-reranker-4b-q4_k_m.gguf",
        "5b798f2918b6bc2c79dc106d83428e052b2efdcbe54f517f05a958a5c4c4d65a",
    ),
}
# What `fetch_models.py` with no arguments fetches. reranker-4b is deliberately
# excluded - see the module docstring - and must be named explicitly to fetch.
DEFAULT_ROLES = ("embedder", "generator", "reranker")
DEST = Path(__file__).resolve().parent.parent / "models"


def verify_sha256(path: Path, expected: str) -> None:
    """Raise SystemExit on mismatch, reporting both hashes. Never deletes
    `path` - on a bad download that means the .part file is left in place
    (the caller must not rename it over the real name first)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        raise SystemExit(
            f"sha256 mismatch for {path}: expected {expected}, got {actual}. "
            f"Leaving {path} in place for inspection; nothing was deleted."
        )


def fetch(repo: str, filename: str, sha256: str | None) -> Path:
    out = DEST / filename
    if out.exists():
        if sha256:
            verify_sha256(out, sha256)
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
    if sha256:
        # Verify before the .part -> real-name rename, so a bad download
        # never lands under the filename the rest of the project trusts.
        verify_sha256(tmp, sha256)
    tmp.rename(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roles", nargs="*", metavar="ROLE",
                    help=f"which models to fetch: {', '.join(MODELS)} "
                         f"(default: {', '.join(DEFAULT_ROLES)})")
    args = ap.parse_args()
    unknown = [r for r in args.roles if r not in MODELS]
    if unknown:
        ap.error(f"unknown role(s): {', '.join(unknown)}; choose from {', '.join(MODELS)}")

    DEST.mkdir(parents=True, exist_ok=True)
    want = args.roles or list(DEFAULT_ROLES)
    for role in want:
        repo, filename, sha256 = MODELS[role]
        print(f"{role}:")
        fetch(repo, filename, sha256)
    print("\nmodels/")
    for f in sorted(DEST.glob("*.gguf")):
        print(f"  {f.stat().st_size/1e6:8.0f} MB  {f.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
