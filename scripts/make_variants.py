#!/usr/bin/env python3
"""Generate typo variants of the eval questions (R2a, tsk_20260927_typos;
docs/robustness-plan.md).

The user's requirement, in their own words: a request must still work when
"some function can be called in other words or with some mistakes like
'mostake' mistyped o instead of i". This script builds the mechanical half:
deterministic, seeded typo injection. The written variants (synonym, casual,
terse, no-name) are a separate, later task.

For every CLEAN base row (no `variant_of`, no `paraphrase_of`) of
data/eval/questions.jsonl then data/eval/emacs_questions.jsonl, in file
order, this emits two rows:

  <qid>.y1   variant_kind "typo1"   exactly 1 edit
  <qid>.y3   variant_kind "typo3"   exactly 3 edits, in 3 distinct words

Each edit is one of four ops, weighted roughly like real typing errors
(substitution and transposition most common, at 0.35 each; deletion and
doubling at 0.15 each):

  substitution   swap one letter for a QWERTY-adjacent one ("mistake" ->
                 "mostake": 'i' -> 'o', which are keyboard-adjacent)
  deletion       drop one letter
  doubling       repeat one letter
  transposition  swap two adjacent letters

A word's first letter is never touched (position 0 is off limits to every
op), only letters in words of >= 4 letters are eligible, and a word is never
touched if it overlaps the row's own answer_contains tokens or gold_hint
strings - "overlaps" means the word (case-insensitively) is one of the
alphabetic sub-words of any answer_contains token or gold_hint string
(`_protected_words`), so e.g. "install" is protected by the token
"python -m pip install" even though the token is not the question's whole
answer. Case of the edited letter is preserved.

Word-pool fallback (data-driven, not spec-literal): typo3 needs 3 distinct
eligible words at >= 4 letters, but two real base rows do not have three
(u07 "How do I set a breakpoint in gdb?" has exactly one 4+ letter word at
all; u22 "How do I install Python packages with pip?" has two, once
"install"/"Python"/"pip" are protected). For those rows ONLY, and only for
typo3, the >= 4 letter floor is relaxed to >= 3 and then >= 2 letters, just
far enough to reach 3 candidates - the protected-word and first-letter rules
are never relaxed. This is flagged in the task report; it is a deliberate,
minimal, fully-deterministic departure from the letter of "only letters in
words of >= 4 letters" forced by two short real questions, not a shortcut
taken for convenience.

Determinism: seed = int(sha256(f"{qid}|{kind}").hexdigest(), 16) where `kind`
is "typo1"/"typo3" and `qid` is the BASE row's qid (never the row's own
order or position), so output is independent of row order and of anything
but the base question text + its answer_contains/gold_hint. Word selection
and op choice are both drawn from one `random.Random(seed)` in a fixed
sequence (sample the words, then per word in that order pick an op and a
position), so re-running produces byte-identical output.

Every other field is copied from the base row unchanged; `tags` gets the
variant tag appended, and `variant_of`, `variant_kind`, `edits` are new.
`edits` is a list of {"word_index", "op", "before", "after"} sufficient to
replay the base question into the variant exactly (`apply_edits`).

Usage:
  make_variants.py --typos --out data/eval/variations_typo.jsonl
  make_variants.py --check data/eval/variations_typo.jsonl

Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BASES = [
    ROOT / "data" / "eval" / "questions.jsonl",
    ROOT / "data" / "eval" / "emacs_questions.jsonl",
]

WORD_RE = re.compile(r"[A-Za-z]+")

# Standard QWERTY row/column adjacency (self excluded from every entry).
QWERTY_ADJ = {
    "q": "wa", "w": "qeas", "e": "wrds", "r": "etdf", "t": "ryfg",
    "y": "tugh", "u": "yihj", "i": "uojk", "o": "ipkl", "p": "ol",
    "a": "qwsz", "s": "awedxz", "d": "serfcx", "f": "drtgvc",
    "g": "ftyhbv", "h": "gyujnb", "j": "huikmn", "k": "jiolm", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
    "n": "bhjm", "m": "njk",
}

# substitution and transposition most common, matching real typing errors.
_OP_WEIGHTS = {"sub": 0.35, "transpose": 0.35, "delete": 0.15, "double": 0.15}

COPIED_FIELDS = [
    "kind", "doc", "domain", "answer_contains", "gold_sec_ids", "gold_primary",
    "gold_hint", "unanswerable_reason", "unanswerable_detail",
]

_WORD_FALLBACK_TIERS = (4, 3, 2)


def _protected_words(row: dict) -> set:
    """Every alphabetic sub-word (lowercased) of the row's own
    answer_contains tokens and gold_hint strings. A question word is never
    edited if it is one of these - see the module docstring."""
    texts = list(row.get("answer_contains") or [])
    texts += list(row.get("gold_hint") or [])
    words = set()
    for t in texts:
        for w in WORD_RE.findall(t):
            words.add(w.lower())
    return words


def _find_words(question: str) -> list:
    """Every maximal alphabetic run in `question`, in text order - this is
    what `word_index` indexes into."""
    return list(WORD_RE.finditer(question))


def _eligible_indices(words: list, protected: set, min_len: int) -> list:
    return [i for i, m in enumerate(words)
            if len(m.group()) >= min_len and m.group().lower() not in protected]


def _word_pool(words: list, protected: set, n_needed: int) -> list:
    """Eligible word indices at >= 4 letters; if that has fewer than
    `n_needed` candidates, widens to >= 3 then >= 2 letters (see the module
    docstring's "Word-pool fallback"). Never relaxes the protected-word
    exclusion. Raises if even >= 2 is not enough."""
    for min_len in _WORD_FALLBACK_TIERS:
        pool = _eligible_indices(words, protected, min_len)
        if len(pool) >= n_needed:
            return pool
    raise ValueError(f"not enough eligible words for {n_needed} distinct edits")


def _transpose_possible(word: str) -> bool:
    """True iff some adjacent pair at position >= 1 (first letter excluded)
    actually differs - a word like "YYYY" (every letter identical) has no
    transposition that changes anything, so it must never be offered."""
    return any(word[i] != word[i + 1] for i in range(1, len(word) - 1))


def _choose_op(rng: random.Random, word: str) -> str:
    ops = ["sub", "delete", "double"]
    if len(word) >= 3 and _transpose_possible(word):
        ops.append("transpose")
    weights = [_OP_WEIGHTS[o] for o in ops]
    return rng.choices(ops, weights=weights, k=1)[0]


def _apply_sub(rng: random.Random, word: str) -> str:
    i = rng.randrange(1, len(word))
    c = word[i]
    neighbors = QWERTY_ADJ.get(c.lower(), "")
    if not neighbors:
        return word  # non-letter or unmapped char: caller retries with a new draw
    nc = rng.choice(neighbors)
    if c.isupper():
        nc = nc.upper()
    return word[:i] + nc + word[i + 1:]


def _apply_delete(rng: random.Random, word: str) -> str:
    i = rng.randrange(1, len(word))
    return word[:i] + word[i + 1:]


def _apply_double(rng: random.Random, word: str) -> str:
    i = rng.randrange(1, len(word))
    return word[:i + 1] + word[i] + word[i + 1:]


def _apply_transpose(rng: random.Random, word: str) -> str:
    i = rng.randrange(1, len(word) - 1)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


_APPLY = {
    "sub": _apply_sub, "delete": _apply_delete,
    "double": _apply_double, "transpose": _apply_transpose,
}


def _edit_word(rng: random.Random, word: str) -> tuple:
    """(new_word, op): draws an op then a position from `rng`, retrying
    (same op, next draw) until the result actually differs from `word` - a
    transposition of two equal adjacent letters, or a substitution with no
    mapped neighbour, would otherwise record a no-op "edit"."""
    op = _choose_op(rng, word)
    for _ in range(50):
        new_word = _APPLY[op](rng, word)
        if new_word != word:
            return new_word, op
    raise RuntimeError(f"could not produce a distinct {op} edit for {word!r}")


def apply_edits(question: str, edits: list) -> str:
    """Replays `edits` (each {"word_index", "op", "before", "after"}) onto
    `question`, returning the resulting text. Edits are applied highest
    word_index first so earlier offsets (computed once, on the original
    `question`) stay valid regardless of length changes from later-in-text
    edits. Raises ValueError if a recorded `before` does not match the
    word actually at that index."""
    words = _find_words(question)
    out = question
    for e in sorted(edits, key=lambda x: -x["word_index"]):
        idx = e["word_index"]
        if idx < 0 or idx >= len(words):
            raise ValueError(f"word_index {idx} out of range (0..{len(words) - 1})")
        m = words[idx]
        actual = question[m.start():m.end()]
        if actual != e["before"]:
            raise ValueError(
                f"word_index {idx}: recorded before {e['before']!r} != "
                f"actual {actual!r}")
        out = out[:m.start()] + e["after"] + out[m.end():]
    return out


def make_edits(base_row: dict, n_edits: int, seed_kind: str) -> list:
    """n_edits edits (in n_edits distinct words) for base_row, seeded on
    sha256(f"{qid}|{seed_kind}")."""
    seed_str = f"{base_row['qid']}|{seed_kind}"
    seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16)
    rng = random.Random(seed)

    words = _find_words(base_row["question"])
    protected = _protected_words(base_row)
    pool = _word_pool(words, protected, n_edits)
    chosen = [rng.choice(pool)] if n_edits == 1 else rng.sample(pool, n_edits)

    edits = []
    for idx in chosen:
        word = words[idx].group()
        new_word, op = _edit_word(rng, word)
        edits.append({"word_index": idx, "op": op, "before": word, "after": new_word})
    edits.sort(key=lambda e: e["word_index"])
    return edits


def build_variant(base_row: dict, suffix: str, variant_kind: str, tag: str,
                   n_edits: int) -> dict:
    edits = make_edits(base_row, n_edits, variant_kind)
    question = apply_edits(base_row["question"], edits)

    out_row = dict(base_row)
    out_row["qid"] = f"{base_row['qid']}.{suffix}"
    out_row["question"] = question
    out_row["tags"] = list(base_row.get("tags", [])) + [tag]
    out_row["variant_of"] = base_row["qid"]
    out_row["variant_kind"] = variant_kind
    out_row["edits"] = edits
    return out_row


def load_rows(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_base_rows(paths: list) -> list:
    """Every CLEAN row (no variant_of, no paraphrase_of) of each path, in
    file order, file by file."""
    out = []
    for path in paths:
        for row in load_rows(path):
            if row.get("variant_of") or row.get("paraphrase_of"):
                continue
            out.append(row)
    return out


def generate(paths: list) -> list:
    out = []
    for base in clean_base_rows(paths):
        out.append(build_variant(base, "y1", "typo1", "variant-typo1", 1))
        out.append(build_variant(base, "y3", "typo3", "variant-typo3", 3))
    return out


def dump_jsonl(rows: list) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n"
                    for r in rows)


def cmd_typos(out_path: Path) -> int:
    rows = generate(DEFAULT_BASES)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dump_jsonl(rows), encoding="utf-8")
    n1 = sum(1 for r in rows if r["variant_kind"] == "typo1")
    n3 = sum(1 for r in rows if r["variant_kind"] == "typo3")
    print(f"wrote {out_path}: {len(rows)} rows ({n1} typo1, {n3} typo3)")
    return 0


def check(path: Path, bases: list | None = None) -> int:
    """`bases` defaults to DEFAULT_BASES; overridable so tests can point
    this at a synthetic pair of files instead of the real eval set."""
    violations = []
    bases = bases if bases is not None else DEFAULT_BASES

    base_rows_by_qid: dict = {}
    base_qids: set = set()
    for base_path in bases:
        if not base_path.exists():
            violations.append(f"missing base file {base_path}")
            continue
        for row in load_rows(base_path):
            base_rows_by_qid[row["qid"]] = row
            base_qids.add(row["qid"])

    if not path.exists():
        print(f"VIOLATION: no such file {path}")
        return 1
    rows = load_rows(path)

    seen_qids: set = set()
    for row in rows:
        qid = row.get("qid", "<missing>")
        loc = qid

        if qid in seen_qids:
            violations.append(f"{loc}: duplicate qid within {path.name}")
        seen_qids.add(qid)
        if qid in base_qids:
            violations.append(f"{loc}: qid collides with a base-file qid")

        base_qid = row.get("variant_of")
        base = base_rows_by_qid.get(base_qid)
        if base is None:
            violations.append(f"{loc}: variant_of {base_qid!r} is not a base qid")
            continue

        variant_kind = row.get("variant_kind")
        edits = row.get("edits")
        if not isinstance(edits, list):
            violations.append(f"{loc}: edits is not a list")
            continue

        if variant_kind == "typo1":
            if len(edits) != 1:
                violations.append(f"{loc}: typo1 must have exactly 1 edit, got {len(edits)}")
        elif variant_kind == "typo3":
            if len(edits) != 3:
                violations.append(f"{loc}: typo3 must have exactly 3 edits, got {len(edits)}")
            n_distinct = len({e.get("word_index") for e in edits})
            if n_distinct != len(edits):
                violations.append(f"{loc}: typo3 edits are not in distinct words")
        else:
            violations.append(f"{loc}: unrecognised variant_kind {variant_kind!r}")

        # replay: base question + edits -> this row's question, exactly.
        try:
            replayed = apply_edits(base["question"], edits)
            if replayed != row.get("question"):
                violations.append(
                    f"{loc}: replay {replayed!r} != row question {row.get('question')!r}")
        except ValueError as e:
            violations.append(f"{loc}: replay failed: {e}")

        # each edited word: >= 4 letters (unless the documented fallback
        # applies - see module docstring), first letter untouched, no
        # overlap with an answer token or gold_hint string.
        protected = _protected_words(base)
        for e in edits:
            before = e.get("before", "")
            after = e.get("after", "")
            if len(before) < 2:
                violations.append(f"{loc}: edited word {before!r} shorter than the 2-letter floor")
            if not before or not after or before[0] != after[0]:
                violations.append(f"{loc}: first letter changed ({before!r} -> {after!r})")
            if before.lower() in protected:
                violations.append(f"{loc}: edited word {before!r} overlaps an answer token/gold_hint")

        # every copied field equals the base's.
        for field in COPIED_FIELDS:
            if field in base or field in row:
                if row.get(field) != base.get(field):
                    violations.append(
                        f"{loc}: field {field!r} differs from base "
                        f"({row.get(field)!r} != {base.get(field)!r})")

    for v in violations:
        print("VIOLATION: " + v)
    if violations:
        print(f"\n{len(violations)} violation(s)")
        return 1
    print(f"check OK: {len(rows)} rows, 0 violations")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--typos", action="store_true", help="generate the typo variant pool")
    ap.add_argument("--out", default=None, help="output path for --typos")
    ap.add_argument("--check", default=None, metavar="FILE",
                     help="validate a generated variant file; exit 1 on any violation")
    args = ap.parse_args()

    if args.check:
        return check(Path(args.check))
    if args.typos:
        if not args.out:
            print("--typos requires --out FILE", file=sys.stderr)
            return 2
        return cmd_typos(Path(args.out))

    print("nothing to do: pass --typos --out FILE or --check FILE", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
