"""Extract structured records from the man pages installed on this machine.

The hierarchy in a man page is authored, not inferred: groff renders section
headings at column 0, subsections at column 3, body at column 7. SEE ALSO is a
curated cross-document edge list. Both are parsed here, never guessed.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, asdict, field
from pathlib import Path

MAN_ROOTS = (Path("/usr/share/man"), Path("/usr/local/share/man"))
COMPRESSION_SUFFIXES = {".gz", ".bz2", ".xz", ".lzma", ".Z"}
RENDER_WIDTH = "100"

# Indentation groff uses for each structural level.
SUBSECTION_INDENT = 3
BODY_INDENT = 7

# groff always indents body text, so a non-blank line at column 0 is a heading.
# Pages vary between ALL CAPS, Title Case, "[UNIT] SECTION OPTIONS" and "2.6 KERNELS",
# so shape is not a usable test; length and a small artifact blocklist are.
_HEADING_MAX_LEN = 80
_HEADING_ARTIFACTS = frozenset({"delim $$", "delim off"})


def _is_heading(line: str) -> bool:
    return (
        0 < len(line) <= _HEADING_MAX_LEN
        and line not in _HEADING_ARTIFACTS
        and not line.endswith((".", ",", ";", ":"))
    )

_RUNNING_HEAD_RE = re.compile(r"\(\d[^)]*\)\s*$")
_XREF_RE = re.compile(r"\b([A-Za-z0-9][A-Za-z0-9_.:+-]*)\((\d[A-Za-z]*)\)")
_SO_RE = re.compile(r"^\.so\s+(\S+)", re.M)
_NAME_SPLIT_RE = re.compile(r"\s+[-–—]+\s+")


@dataclass
class ManFile:
    path: Path
    name: str
    section: str

    @property
    def doc_id(self) -> str:
        return f"{self.name}.{self.section}"


@dataclass
class Section:
    sec_id: str
    heading: str
    level: int
    parent: str | None
    text: str


@dataclass
class ManDoc:
    doc_id: str
    name: str
    section: str
    path: str
    title: str
    summary: str
    aliases: list[str] = field(default_factory=list)
    see_also: list[dict] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sections"] = [asdict(s) if not isinstance(s, dict) else s for s in self.sections]
        return d


def _split_name_section(filename: str) -> tuple[str, str] | None:
    """`tar.1.gz` -> ('tar', '1'); `Mail.1p.gz` -> ('Mail', '1p')."""
    stem = filename
    while True:
        suffix = Path(stem).suffix
        if suffix in COMPRESSION_SUFFIXES:
            stem = stem[: -len(suffix)]
        else:
            break
    name, dot, sec = stem.rpartition(".")
    if not dot or not sec or not sec[0].isdigit():
        return None
    return name, sec


def discover(sections: tuple[str, ...] = ("1", "8"), roots=MAN_ROOTS) -> list[ManFile]:
    """English man pages in the requested sections.

    Only top-level manN directories are walked; `/usr/share/man/de/man1` and
    friends are translations and are skipped.
    """
    found: dict[str, ManFile] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for sec in sections:
            for d in sorted(root.glob(f"man{sec}*")):
                if not d.is_dir():
                    continue
                for f in sorted(d.iterdir()):
                    if not f.is_file():
                        continue
                    parsed = _split_name_section(f.name)
                    if parsed is None:
                        continue
                    name, full_sec = parsed
                    if full_sec[0] != sec:
                        continue
                    mf = ManFile(path=f, name=name, section=full_sec)
                    found.setdefault(mf.doc_id, mf)
    return list(found.values())


def read_source(path: Path) -> str:
    """Decompressed roff source, for cheap checks that don't need rendering."""
    suffix = path.suffix
    if suffix == ".gz":
        import gzip

        opener = gzip.open
    elif suffix == ".bz2":
        import bz2

        opener = bz2.open
    elif suffix in (".xz", ".lzma"):
        import lzma

        opener = lzma.open
    else:
        opener = open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def redirect_target(path: Path) -> str | None:
    """A `.so man1/gzip.1` stub is an alias, not a document. Return its target."""
    try:
        src = read_source(path)
    except OSError:
        return None
    body = [ln for ln in src.splitlines() if ln.strip() and not ln.startswith('.\\"')]
    if len(body) <= 3:
        m = _SO_RE.search(src)
        if m:
            return m.group(1)
    return None


def render(path: Path) -> str:
    """groff-rendered plain text, overstrike stripped."""
    env = dict(os.environ)
    env.update(
        MANWIDTH=RENDER_WIDTH,
        MAN_KEEP_FORMATTING="",
        GROFF_NO_SGR="1",
        LC_ALL="C.UTF-8",
        LANG="C.UTF-8",
    )
    man = subprocess.run(
        ["man", "--no-hyphenation", "--no-justification", "-P", "cat", str(path)],
        capture_output=True,
        env=env,
        timeout=60,
    )
    if man.returncode != 0 and not man.stdout:
        raise RuntimeError(man.stderr.decode("utf-8", "replace")[:200])
    col = subprocess.run(["col", "-bx"], input=man.stdout, capture_output=True, timeout=60)
    return col.stdout.decode("utf-8", "replace")


def _strip_running_heads(lines: list[str]) -> list[str]:
    """Drop the repeated `TAR(1) ... TAR(1)` header and footer lines."""
    out = []
    for ln in lines:
        stripped = ln.strip()
        if stripped and _RUNNING_HEAD_RE.search(stripped) and _XREF_RE.match(stripped):
            # A running head starts and ends with the same NAME(sec) token.
            first = _XREF_RE.match(stripped)
            if stripped.endswith(first.group(0)):
                continue
        out.append(ln)
    return out


def _dedent(block: list[str], amount: int) -> str:
    out = []
    for ln in block:
        out.append(ln[amount:] if ln[:amount].strip() == "" else ln.lstrip())
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse(rendered: str, mf: ManFile) -> ManDoc:
    lines = _strip_running_heads(rendered.splitlines())

    # Title from the running head's centre field, before it was stripped.
    title = ""
    for ln in rendered.splitlines():
        s = ln.strip()
        if s and _RUNNING_HEAD_RE.search(s):
            middle = re.sub(r"^\S+\(\d[A-Za-z]*\)\s*|\s*\S+\(\d[A-Za-z]*\)$", "", s).strip()
            title = middle
            break

    sections: list[Section] = []
    cur_heading: str | None = None
    cur_parent: str | None = None
    cur_level = 1
    buf: list[str] = []

    def flush():
        if cur_heading is None:
            return
        text = _dedent(buf, BODY_INDENT)
        if not text and cur_level != 1:
            return
        slug = cur_heading if cur_level == 1 else f"{cur_parent}/{cur_heading}"
        sections.append(
            Section(
                sec_id=f"{mf.doc_id}#{slug}",
                heading=cur_heading,
                level=cur_level,
                parent=cur_parent,
                text=text,
            )
        )

    for ln in lines:
        if not ln.strip():
            buf.append(ln)
            continue
        indent = len(ln) - len(ln.lstrip())
        stripped = ln.strip()
        if indent == 0 and _is_heading(stripped):
            flush()
            buf = []
            cur_heading, cur_parent, cur_level = stripped, None, 1
        elif indent == SUBSECTION_INDENT and cur_heading is not None and not stripped.startswith("-"):
            flush()
            buf = []
            cur_parent = cur_heading if cur_level == 1 else cur_parent
            cur_heading, cur_level = stripped, 2
        else:
            buf.append(ln)
    flush()

    by_heading = {s.heading.upper(): s for s in sections if s.level == 1}

    aliases: list[str] = []
    summary = ""
    name_sec = by_heading.get("NAME")
    if name_sec:
        flat = " ".join(name_sec.text.split())
        parts = _NAME_SPLIT_RE.split(flat, maxsplit=1)
        if len(parts) == 2:
            names_part, summary = parts[0], parts[1]
        else:
            names_part, summary = "", flat
        aliases = [n.strip() for n in names_part.split(",") if n.strip() and n.strip() != mf.name]

    see_also: list[dict] = []
    sa_text = "\n".join(
        s.text
        for s in sections
        if s.heading.upper() == "SEE ALSO" or (s.parent or "").upper() == "SEE ALSO"
    )
    if sa_text:
        seen = set()
        for m in _XREF_RE.finditer(sa_text):
            key = (m.group(1), m.group(2))
            if key in seen or m.group(1) == mf.name:
                continue
            seen.add(key)
            see_also.append({"name": m.group(1), "section": m.group(2)})

    return ManDoc(
        doc_id=mf.doc_id,
        name=mf.name,
        section=mf.section,
        path=str(mf.path),
        title=title,
        summary=summary,
        aliases=aliases,
        see_also=see_also,
        sections=sections,
    )


def extract(mf: ManFile) -> ManDoc | None:
    try:
        return parse(render(mf.path), mf)
    except Exception:
        return None
