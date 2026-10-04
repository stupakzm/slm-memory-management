#!/usr/bin/env python3
"""Derive a reranker GGUF whose built-in rerank template carries a different instruction.

Qwen3-Reranker's llama.cpp GGUF stores its prompt in the metadata key
`tokenizer.chat_template.rerank`, and that template embeds the task instruction
after "<Instruct>: ". Phase 15 R14 measures whether the instruction matters, so this
copies a GGUF with exactly that one string changed - every other metadata field and
every tensor is written back as it was read (the same GGUFReader -> GGUFWriter copy
as gguf-py's scripts/gguf_new_metadata.py).

  .venv-train/bin/python scripts/rerank_instruct.py --in models/reranker.gguf \
      --out models/reranker-r14.gguf --instruction "Given a question from a user ..."

Exit 2 (nothing written) if --out exists, --in is missing, the key is missing or not a
string, or the old instruction does not occur exactly once. `gguf` (llama.cpp's
gguf-py) is imported lazily so swap_instruction can be tested without it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

KEY = "tokenizer.chat_template.rerank"
OLD_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def swap_instruction(template: str, new: str) -> str:
    """`template` with its one occurrence of OLD_INSTRUCTION replaced by `new`.
    Raises ValueError unless OLD_INSTRUCTION occurs exactly once."""
    n = template.count(OLD_INSTRUCTION)
    if n != 1:
        raise ValueError(f"expected the old instruction exactly once in the template, found {n}")
    return template.replace(OLD_INSTRUCTION, new)


def derive(src: Path, dst: Path, instruction: str) -> None:
    import gguf  # noqa: PLC0415  (lazy: only the GGUF I/O needs gguf-py)

    reader = gguf.GGUFReader(src, "r")
    field = reader.get_field(KEY)
    if field is None:
        raise KeyError(f"{src} has no {KEY}")
    if field.types[0] != gguf.GGUFValueType.STRING:
        raise TypeError(f"{KEY} is not a string field")
    new_template = swap_instruction(field.contents(), instruction)

    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()

    for f in reader.fields.values():
        # Virtual fields and the ones GGUFWriter writes itself.
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        val_type = f.types[0]
        sub_type = f.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = new_template if f.name == KEY else f.contents()
        if value is not None:
            writer.add_key_value(f.name, value, val_type, sub_type=sub_type)

    for t in reader.tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="src", required=True, type=Path, help="source reranker GGUF")
    ap.add_argument("--out", dest="dst", required=True, type=Path, help="derived GGUF to write")
    ap.add_argument("--instruction", required=True,
                    help="text that replaces the template's instruction")
    args = ap.parse_args()
    if args.dst.exists():
        print(f"{args.dst} exists: refusing to overwrite", file=sys.stderr)
        return 2
    if not args.src.is_file():
        print(f"{args.src} is not a file", file=sys.stderr)
        return 2
    try:
        derive(args.src, args.dst, args.instruction)
    except (KeyError, TypeError, ValueError) as e:
        print(e.args[0] if e.args else e, file=sys.stderr)
        if args.dst.exists():  # only reachable once the writer opened it
            args.dst.unlink()
        return 2
    print(f"wrote {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
