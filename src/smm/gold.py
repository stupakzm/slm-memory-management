"""Gold-token alias matching (tsk_20260926_a51d0707).

The gold label in data/eval/questions.jsonl uses whichever form the man page
happens to list first - usually the long-form option (`--no-clobber`) - so a
correct answer using the documented short form (`-n`) is scored wrong under
plain substring matching. scripts/derive_gold_aliases.py derives aliases
once, mechanically, from the corpus and questions alone (never from an
answer or a result), and this module applies them at scoring time.

Stdlib only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def load_aliases(path) -> dict:
    """{qid: {token: [alias-dict, ...]}}. A missing file gives {}."""
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def token_ok(text: str, token: str, aliases_for_token) -> bool:
    """True if `token` occurs in `text` verbatim (the original substring
    semantics, unchanged), or any of its aliases matches on option-token
    boundaries: `(?<![\\w-])` + alias + `(?![\\w-])`, so `-r` matches
    `scp -r dir` but not inside `--recursive` or `-rf`."""
    if token in text:
        return True
    for entry in aliases_for_token or []:
        alias = entry["alias"] if isinstance(entry, dict) else entry
        pattern = r"(?<![\w-])" + re.escape(alias) + r"(?![\w-])"
        if re.search(pattern, text):
            return True
    return False


def is_correct(text: str, tokens, qid_aliases) -> bool:
    """All tokens must be token_ok. With empty aliases (qid_aliases falsy,
    or every token's own alias list empty/missing) this is exactly
    `all(t in text for t in tokens)`."""
    qid_aliases = qid_aliases or {}
    return all(token_ok(text, t, qid_aliases.get(t)) for t in tokens)


def aliases_for(aliases: dict, qid: str, base_qid: str | None = None) -> dict:
    """Per-qid alias entry (tsk_20260927_typos), with fallback for variants.

    `aliases` is keyed by qid (load_aliases' shape), so a typo/paraphrase
    variant like `a01.y1` has no entry of its own even though its base
    `a01` does - scoring it straight would silently lose every alias its
    base earned. Returns `aliases[qid]` if that key is present (including
    an explicit empty dict - only a *missing* key falls back, never an
    empty-but-present one); otherwise `aliases.get(base_qid, {})` if
    `base_qid` is given; otherwise `{}`.

    On today's data every qid that is itself missing also has a missing (or
    no) base entry, so this is an exact no-op there - see
    scripts/rescore_answers.py and scripts/eval_answers.py, its two call
    sites."""
    if qid in aliases:
        return aliases[qid]
    if base_qid:
        return aliases.get(base_qid, {})
    return {}
