"""Spelling normalisation of a question against the index's own vocabulary
(phase 11 R4a, tsk_20260927_spellnorm; docs/phase11-results.md's "R4a
pre-registration"). R3 showed a single typo mostly breaks the 4B READER even
when retrieval is fine, so this snaps each out-of-corpus word to the corpus
word closest in Damerau-Levenshtein (optimal string alignment) distance,
*before* the question is used for anything - retrieval and generation both
read the corrected text.

Stdlib only, no sqlite_vec: `build_vocab` opens the index read-only with
plain `sqlite3` and reads only the `chunks` table (its vector tables need an
extension this package deliberately does not import).

Mechanism, exactly as pre-registered:
  - vocabulary: lower-cased alphabetic words (regex [a-z]+ after lower()) of
    length >= 3 seen at least twice across every chunk's prefix + text.
  - a token is a candidate only if its punctuation-stripped core is purely
    alphabetic, length >= 4, and not in the vocabulary (case-insensitively).
    A digit, '-', '/', '=', '.', '_' or '~' ANYWHERE in the core (leading,
    trailing or embedded - the boundary strip deliberately excludes these
    six characters, see `_STRIP_CHARS`) makes the core non-alphabetic and so
    disqualifies the token: flags, paths, `KEY=VALUE` and versions are exact.
  - distance <= 1 for a 4-7 letter core, <= 2 for 8+; ties go to a
    keyboard-adjacent single substitution, then the more frequent vocab
    word, then the word itself (deterministic).
  - the first letter is NOT protected - the typo generator
    (scripts/make_variants.py) never touches it, so protecting it here would
    tune the mechanism to the generator rather than to people.
  - case pattern (all-lower / Capitalised / ALL-CAPS) and the token's
    original surrounding punctuation are preserved; anything else is left
    exactly as it was written.

Phase 15 R12 (`normalize_local`, CLI `--mode local`; docs/phase15-results.md
"R12 pre-registration"): the same eligibility, distances, tie-break and case
rules, but a word is corrected only toward words found in the extracts the
reader is about to read (`hit_extract`: doc id, prefix and text of each hit),
and the repaired question is for the reader only.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import string
import time
from itertools import combinations
from pathlib import Path

# spec: "regex [a-z]+ after lower()" for vocabulary extraction.
_VOCAB_WORD_RE = re.compile(r"[a-z]+")

# QWERTY row/column adjacency (self excluded), copied from
# scripts/make_variants.py's QWERTY_ADJ (not imported - normalize.py must
# stay import-free of scripts/, and this project's word-model for keyboard
# adjacency lives in one place per file, not shared as a runtime dependency).
QWERTY_ADJ = {
    "q": "wa", "w": "qeas", "e": "wrds", "r": "etdf", "t": "ryfg",
    "y": "tugh", "u": "yihj", "i": "uojk", "o": "ipkl", "p": "ol",
    "a": "qwsz", "s": "awedxz", "d": "serfcx", "f": "drtgvc",
    "g": "ftyhbv", "h": "gyujnb", "j": "huikmn", "k": "jiolm", "l": "kop",
    "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn",
    "n": "bhjm", "m": "njk",
}

# Characters stripped from a token's leading/trailing edge to find its
# "core". Deliberately EXCLUDES '-', '/', '=', '.', '_', '~': those are
# never boundary punctuation here, they are exact-match structure (flags,
# paths, KEY=VALUE, versions), so if one sits at a token's edge it stays
# part of the core and fails the "purely alphabetic" candidate test below -
# see the module docstring.
_PROTECTED_CHARS = set("-/=._~")
_STRIP_CHARS = "".join(c for c in string.punctuation if c not in _PROTECTED_CHARS)


# --------------------------------------------------------------------------
# vocabulary
# --------------------------------------------------------------------------

def _scan_vocab(db_path: Path) -> dict:
    """Every lower-cased alphabetic word (length >= 3) in `chunks.prefix` +
    `chunks.text`, kept if its total count across the whole table is >= 2.
    Opens the db read-only via a `file:` URI so this can never write to it,
    and touches only the `chunks` table - never the vec tables, which need
    an extension this package does not import (see module docstring)."""
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        counts: dict = {}
        for prefix, text in conn.execute("SELECT prefix, text FROM chunks"):
            combined = f"{prefix or ''} {text or ''}".lower()
            for w in _VOCAB_WORD_RE.findall(combined):
                if len(w) >= 3:
                    counts[w] = counts.get(w, 0) + 1
    finally:
        conn.close()
    return {w: c for w, c in counts.items() if c >= 2}


def build_vocab(db_path, cache_path=None) -> dict:
    """dict[str, int]: word -> count, from `db_path`'s `chunks` table (see
    `_scan_vocab`). Cached next to the db as `<db>.vocab.json` unless
    `cache_path` overrides it (scripts/normalize's --vocab-cache; useful
    during development so the cache never lands next to a real, gitignored
    index). The cache is read instead of rescanned when it is newer than
    the db file itself."""
    db_path = Path(db_path)
    cache_path = Path(cache_path) if cache_path else Path(f"{db_path}.vocab.json")
    if (cache_path.exists() and db_path.exists()
            and cache_path.stat().st_mtime > db_path.stat().st_mtime):
        return load_vocab(cache_path)

    vocab = _scan_vocab(db_path)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(vocab, sort_keys=True), encoding="utf-8")
    except OSError:
        pass  # the cache is a speed optimisation; a read-only cache
              # directory must never stop normalization from working.
    return vocab


def load_vocab(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# distance + candidate search
# --------------------------------------------------------------------------

def osa_distance(a: str, b: str, max_dist: int | None = None) -> int:
    """Optimal string alignment (restricted Damerau-Levenshtein) distance:
    insertion, deletion, substitution, or one adjacent transposition per
    substring (no overlapping edits). `max_dist`, if given, only short-
    circuits an obviously-too-far pair via the length difference - the
    returned value is never itself capped."""
    la, lb = len(a), len(b)
    if max_dist is not None and abs(la - lb) > max_dist:
        return max_dist + 1
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            best = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                best = min(best, d[i - 2][j - 2] + 1)
            d[i][j] = best
    return d[la][lb]


def _max_dist_for(length: int) -> int:
    """4-7 letters -> 1, 8+ -> 2 (pre-registered distance bands)."""
    return 1 if length <= 7 else 2


def _is_kbd_substitution(core: str, candidate: str) -> bool:
    """True iff `core` and `candidate` are the same length, differ at
    exactly one position, and that position's two letters are QWERTY-
    adjacent - i.e. the best alignment between them IS a single keyboard-
    adjacent substitution (a transposition of two DIFFERENT adjacent
    letters also keeps the length equal but changes two positions, so this
    naturally excludes it)."""
    if len(core) != len(candidate):
        return False
    diffs = [i for i in range(len(core)) if core[i] != candidate[i]]
    if len(diffs) != 1:
        return False
    i = diffs[0]
    return candidate[i] in QWERTY_ADJ.get(core[i], "")


_INDEX_CACHE: dict = {}

# Deletion-neighbourhood ("SymSpell") lookup: brute-forcing every vocab word
# within a length band is O(bucket size) DP calls per out-of-vocab token -
# measured at tens of milliseconds against a 50k-word vocab, an order of
# magnitude over budget. Precomputing, once per vocab, the set of strings
# reachable by deleting up to 2 characters from each vocab word - and
# indexing vocab words by that string - turns a lookup into: delete up to
# `max_dist` characters from the query core (at most a few dozen strings for
# a typical word), probe the index, and verify the handful of hits that
# come back with the real `osa_distance`. Any pair of strings at true
# distance <= 2 shares at least one such deletion in common, so this never
# misses a real candidate; it only ever needs to verify false positives.
_DELETE_MAX = 2


def _deletions(word: str, n: int) -> set:
    """Every distinct string from deleting exactly `n` characters out of
    `word` (n <= len(word)); {word} itself for n == 0."""
    if n == 0:
        return {word}
    if n > len(word):
        return set()
    out = set()
    for combo in combinations(range(len(word)), n):
        skip = set(combo)
        out.add("".join(c for i, c in enumerate(word) if i not in skip))
    return out


def _deletions_upto(word: str, max_n: int) -> set:
    out = set()
    for n in range(0, max_n + 1):
        out |= _deletions(word, n)
    return out


def _delete_index(vocab: dict, depth: int = _DELETE_MAX) -> dict:
    """deletion-string -> set of vocab words that produce it, deleting up
    to `depth` (default `_DELETE_MAX`) characters; a lookup that allows
    `d` edits needs `depth >= d`. Cached by `id(vocab)` (re-validated
    against `len(vocab)`), so this is built once per vocab dict and reused
    across every `normalize()` call in the process - the normal CLI and
    eval_answers.py/ask.py usage. The entry keeps a reference to `vocab`
    itself, so its id cannot be recycled by a different dict of the same
    length while the entry exists (checked with `is`)."""
    key = (id(vocab), depth)
    cached = _INDEX_CACHE.get(key)
    if cached is not None and cached[0] == len(vocab) and cached[2] is vocab:
        return cached[1]
    idx: dict = {}
    for w in vocab:
        for deleted in _deletions_upto(w, depth):
            idx.setdefault(deleted, set()).add(w)
    _INDEX_CACHE[key] = (len(vocab), idx, vocab)
    return idx


def _best_candidate(core_lower: str, vocab: dict, idx: dict):
    """(word, distance) of the best correction for `core_lower`, or None.
    Picked by (distance, NOT a keyboard-adjacent substitution, -frequency,
    word) - see the module docstring. `idx` is a `_delete_index(vocab)`."""
    max_dist = _max_dist_for(len(core_lower))
    seen = set()
    for deleted in _deletions_upto(core_lower, max_dist):
        seen.update(idx.get(deleted, ()))
    best_key = None
    best = None
    for w in seen:
        d = osa_distance(core_lower, w, max_dist)
        if d > max_dist:
            continue
        key = (d, not _is_kbd_substitution(core_lower, w), -vocab[w], w)
        if best_key is None or key < best_key:
            best_key, best = key, (w, d)
    return best


def _split_token(token: str) -> tuple:
    """(prefix, core, suffix): strip leading/trailing punctuation (never
    the six protected structure characters, see `_STRIP_CHARS`)."""
    i, j = 0, len(token)
    while i < j and token[i] in _STRIP_CHARS:
        i += 1
    while j > i and token[j - 1] in _STRIP_CHARS:
        j -= 1
    return token[:i], token[i:j], token[j:]


def _match_case(word: str, original_core: str) -> str:
    """Preserve ALL-CAPS / Capitalised / anything else-as-lower, from
    `original_core`'s own case pattern - `word` is the vocab candidate,
    already lower-cased by `build_vocab`."""
    if original_core.isupper():
        return word.upper()
    if original_core[:1].isupper() and original_core[1:].islower():
        return word[:1].upper() + word[1:]
    return word


def _eligible(token: str, known: dict, targets: dict):
    """(prefix, core, suffix) if `token` may be corrected, else None: its core
    is purely alphabetic, 4+ letters, and its lower-cased form is in neither
    `known` (the corpus vocabulary) nor `targets` (a word that is already a
    correction target is not a typo)."""
    prefix, core, suffix = _split_token(token)
    if not core or not core.isalpha() or len(core) < 4:
        return None
    core_lower = core.lower()
    if core_lower in known or core_lower in targets:
        return None
    return prefix, core, suffix


def _repair(question: str, known: dict, targets: dict, get_idx) -> tuple:
    """The shared token loop behind `normalize` and `normalize_local`.
    A token is eligible iff its core is purely alphabetic, 4+ letters, and
    its lower-cased core is not in `known` (the corpus vocabulary); it is
    then snapped to the best word in `targets` (a word already in `targets`
    is left alone). `get_idx()` returns `_delete_index(targets)` and is only
    called once a token actually needs a candidate search. For `normalize`,
    `known is targets`, so this is exactly the original loop."""
    if not question:
        return question, []
    idx = None
    out_tokens = []
    edits = []
    for token in question.split(" "):
        if not token:
            out_tokens.append(token)
            continue
        parts = _eligible(token, known, targets)
        if parts is None:
            out_tokens.append(token)
            continue
        prefix, core, suffix = parts
        core_lower = core.lower()
        if idx is None:
            idx = get_idx()
        found = _best_candidate(core_lower, targets, idx)
        if found is None:
            out_tokens.append(token)
            continue
        word, dist = found
        new_core = _match_case(word, core)
        out_tokens.append(f"{prefix}{new_core}{suffix}")
        edits.append({"from": core, "to": new_core, "distance": dist})
    return " ".join(out_tokens), edits


def normalize(question: str, vocab: dict) -> tuple:
    """(new_question, edits). Tokenises on the space character (keeping
    punctuation attached and whitespace itself byte-exact via split(" ")/
    " ".join round-tripping), corrects each eligible token's core in place,
    and leaves everything else untouched. `edits` is
    [{"from": core, "to": corrected_core, "distance": d}, ...], in question
    order. The first letter is NOT protected (see module docstring)."""
    return _repair(question, vocab, vocab, lambda: _delete_index(vocab))


def hit_extract(hit: dict) -> str:
    """What the reader is shown for one hit, minus its "[i] " label -
    `smm.generate.build_prompt`'s f"{doc_id}\n{prefix}{text}"."""
    return f"{hit['doc_id']}\n{hit['prefix']}{hit['text']}"


def local_targets(extracts) -> dict:
    """Counter-style dict: lower-cased [a-z]+ words of length >= 3 counted
    over the extract strings (phase 15 R12)."""
    counts: dict = {}
    for ex in extracts:
        for w in _VOCAB_WORD_RE.findall(ex.lower()):
            if len(w) >= 3:
                counts[w] = counts.get(w, 0) + 1
    return counts


def normalize_local(question: str, extracts, vocab: dict) -> tuple:
    """(new_question, edits) - phase 15 R12. Eligibility is `normalize()`'s
    (alphabetic core, 4+ letters, not in the corpus `vocab`), but the words
    a token may be corrected TO are only those found in `extracts` (the
    strings the reader is about to read, see `hit_extract`), counted as in
    `local_targets`. Distance bands, tie-break order, case and punctuation
    handling, and the edits shape are `normalize()`'s own. A token that is
    itself a word of the extracts is evidence, not a typo, and is left
    alone."""
    targets = local_targets(extracts)
    # Only extract words within reach (by length) of an eligible token can be
    # a correction, so index just those: same candidates, a much smaller
    # deletion index, which is what keeps this per-question call cheap.
    lengths = set()
    depth = 0
    for token in question.split(" "):
        parts = _eligible(token, vocab, targets) if token else None
        if parts is not None:
            n = len(parts[1])
            lengths.update(range(n - _max_dist_for(n), n + _max_dist_for(n) + 1))
            depth = max(depth, _max_dist_for(n))
    targets = {w: c for w, c in targets.items() if len(w) in lengths}

    def get_idx():
        # `_delete_index` memoises by id(vocab); `targets` is a throwaway
        # per-call dict, so evict its entry rather than keep it alive. Only as
        # deep as the longest eligible token needs (1 edit -> 1 deletion).
        try:
            return _delete_index(targets, depth)
        finally:
            _INDEX_CACHE.pop((id(targets), depth), None)

    return _repair(question, vocab, targets, get_idx)


def normalize_query(question: str, mode: str, vocab: dict | None = None,
                    extracts=None) -> tuple:
    """The single call site scripts/eval_answers.py and scripts/ask.py use.
    `mode="off"` returns `(question, [])` unchanged - byte-identical to
    before this feature existed. `mode="spell"` requires `vocab` and
    delegates to `normalize()`. `mode="local"` (phase 15 R12) requires
    `vocab` and `extracts` and delegates to `normalize_local()`."""
    if mode == "off":
        return question, []
    if mode not in ("spell", "local"):
        raise ValueError(f"unknown normalize mode {mode!r}")
    if vocab is None:
        raise ValueError(f"normalize_query(mode={mode!r}) requires vocab")
    if mode == "local":
        if extracts is None:
            raise ValueError("normalize_query(mode='local') requires extracts")
        return normalize_local(question, extracts, vocab)
    return normalize(question, vocab)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

_RUN_RE = re.compile(r"[A-Za-z]+")  # same regex make_variants.py's word_index counts over


def _load_pool_rows(paths) -> list:
    rows = []
    seen = set()
    for p in paths:
        for line in Path(p).open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["qid"] in seen:
                raise SystemExit(f"duplicate qid across pool files: {row['qid']!r}")
            seen.add(row["qid"])
            rows.append(row)
    return rows


def _classify_kind(row: dict) -> str:
    """Same assignment as scripts/robustness_report.py's classify_kind - a
    local copy, not an import, so this stdlib-only package never depends on
    scripts/."""
    if not row.get("variant_of") and not row.get("paraphrase_of"):
        return "clean"
    if row.get("paraphrase_of"):
        return "paraphrase"
    return row["variant_kind"]


def _percentile(values: list, p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _score_typo_row(row: dict, normalized_question: str) -> tuple:
    """(reverted, partial, new_wrong) for one typo1/typo3 row: classify the
    generator's own recorded `edits` (word_index/before/after, replayable -
    blk_make_variants_typo_procedure) against what normalize() did at the
    same word position - word runs found with the SAME regex
    scripts/make_variants.py indexes `word_index` over, so positions line
    up - plus any change normalize() made to a word that was NOT one of the
    recorded edits at all ("new wrong edits")."""
    orig_words = _RUN_RE.findall(row["question"])
    norm_words = _RUN_RE.findall(normalized_question)
    if len(orig_words) != len(norm_words):
        # Structurally this should never happen: normalize() only ever
        # swaps one purely-alphabetic run for another, never adds or
        # removes one. Treat as unscored rather than crash the report.
        return 0, 0, 0
    edited_idx = {e["word_index"] for e in row["edits"]}
    reverted = partial = new_wrong = 0
    for e in row["edits"]:
        i = e["word_index"]
        if i >= len(norm_words):
            continue
        if norm_words[i] == e["before"]:
            reverted += 1
        elif norm_words[i] != orig_words[i]:
            partial += 1
    for i in range(len(orig_words)):
        if i in edited_idx:
            continue
        if norm_words[i] != orig_words[i]:
            new_wrong += 1
    return reverted, partial, new_wrong


def _unpack_entry(entry):
    """(hits, gate_hits) of one eval_answers retrieval-cache entry, either
    shape: a plain list is both, a dict carries both. A local copy of
    scripts/eval_answers.py's `unpack_entry` (this package imports nothing
    from scripts/)."""
    if isinstance(entry, dict):
        return entry["hits"], entry["gate_hits"]
    return entry, entry


def _gate_score(hits: list) -> float:
    """Local copy of smm.retrieve.gate_score (that module needs sqlite_vec):
    the top hit's rerank_score, else its score; -inf for no hits."""
    if not hits:
        return float("-inf")
    return hits[0].get("rerank_score", hits[0].get("score", 0.0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--pool", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab-cache", default=None,
                     help="override the vocab cache path (default: "
                          "<db>.vocab.json next to --db)")
    ap.add_argument("--mode", choices=("spell", "local"), default="spell",
                     help="'spell' (default) is the phase 11 corpus-vocabulary "
                          "corrector; 'local' (phase 15 R12) repairs only toward words "
                          "in the extracts the reader would read, from --retrieved")
    ap.add_argument("--retrieved", default=None,
                     help="--mode local: an eval_answers NAME-retrieved.json cache")
    ap.add_argument("--gate", type=float, default=0.65,
                     help="--mode local: a row whose gate_hits top score is below "
                          "this is reported unchanged and counted as gated")
    ap.add_argument("--qids-out", default=None,
                     help="write the JSON list of changed qids, in pool order "
                          "(the format eval_answers.py --qids reads)")
    args = ap.parse_args()
    local = args.mode == "local"
    if local and not args.retrieved:
        ap.error("--mode local requires --retrieved")
    if args.retrieved and not local:
        ap.error("--retrieved is only valid with --mode local")

    vocab = build_vocab(args.db, cache_path=args.vocab_cache)
    rows = _load_pool_rows(args.pool)
    retrieved = json.loads(Path(args.retrieved).read_text(encoding="utf-8")) if local else None
    if local:
        missing = [r["qid"] for r in rows if r["qid"] not in retrieved]
        if missing:
            raise SystemExit(f"{len(missing)} pool qids not in {args.retrieved}: "
                             f"{missing[:5]}")

    kind_totals: dict = {}
    typo_stats = {"typo1": [0, 0, 0, 0], "typo3": [0, 0, 0, 0]}  # reverted, partial, new_wrong, n_rows
    clean_changed = []
    times_ms = []
    results = []

    changed_qids = []

    for row in rows:
        gated = False
        t0 = time.perf_counter()
        if local:
            hits, gate_hits = _unpack_entry(retrieved[row["qid"]])
            gated = _gate_score(gate_hits) < args.gate
            if gated:
                new_q, edits = row["question"], []
            else:
                new_q, edits = normalize_local(
                    row["question"], [hit_extract(h) for h in hits], vocab)
        else:
            new_q, edits = normalize(row["question"], vocab)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

        kind = _classify_kind(row)
        changed = new_q != row["question"]
        bucket = kind_totals.setdefault(kind, {"n": 0, "changed": 0})
        bucket["n"] += 1
        bucket["changed"] += int(changed)
        if local:
            bucket["gated"] = bucket.get("gated", 0) + int(gated)
        if changed:
            changed_qids.append(row["qid"])

        if kind in typo_stats and row.get("edits"):
            reverted, partial, new_wrong = _score_typo_row(row, new_q)
            s = typo_stats[kind]
            s[0] += reverted
            s[1] += partial
            s[2] += new_wrong
            s[3] += 1

        if kind == "clean" and changed:
            clean_changed.append({"qid": row["qid"], "from": row["question"], "to": new_q})

        rec = {
            "qid": row["qid"], "kind": kind, "changed": changed,
            "question": row["question"], "question_normalized": new_q,
            "edits": edits,
        }
        if local:
            rec["gated"] = gated
        results.append(rec)

    clean_changed.sort(key=lambda r: r["qid"])
    results.sort(key=lambda r: r["qid"])

    mean_ms = sum(times_ms) / len(times_ms) if times_ms else 0.0
    p95_ms = _percentile(times_ms, 0.95)

    report = {
        "db": args.db,
        "pool": sorted(str(p) for p in args.pool),
        "n_questions": len(rows),
        "kinds": kind_totals,
        "typo_edit_recovery": {
            k: {"reverted": v[0], "partial": v[1], "new_wrong": v[2], "n_rows": v[3]}
            for k, v in typo_stats.items()
        },
        "timing_ms": {"mean": mean_ms, "p95": p95_ms},
        "clean_changed": clean_changed,
        "results": results,
    }
    if local:
        report.update({"mode": "local", "retrieved": args.retrieved, "gate": args.gate})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"normalised {len(rows)} pool questions from {len(args.pool)} file(s)\n")
    print(f"{'kind':12} {'n':>5} {'changed':>8}" + (f" {'gated':>6}" if local else ""))
    for kind in sorted(kind_totals):
        b = kind_totals[kind]
        print(f"{kind:12} {b['n']:5d} {b['changed']:8d}"
              + (f" {b['gated']:6d}" if local else ""))
    print()
    for k in ("typo1", "typo3"):
        s = typo_stats[k]
        if s[3]:
            print(f"{k}: {s[3]} rows with recorded edits - "
                  f"reverted {s[0]}, partial {s[1]}, new-wrong {s[2]}")
    print(f"\ntiming: mean {mean_ms:.3f} ms/question, p95 {p95_ms:.3f} ms/question")
    print(f"\nclean questions changed: {len(clean_changed)}")
    for c in clean_changed:
        print(f"  {c['qid']}: {c['from']!r} -> {c['to']!r}")
    if args.qids_out:
        qo = Path(args.qids_out)
        qo.parent.mkdir(parents=True, exist_ok=True)
        qo.write_text(json.dumps(changed_qids), encoding="utf-8")
        print(f"wrote {qo} ({len(changed_qids)} changed qids)")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
