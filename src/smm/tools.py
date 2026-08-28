"""Tool calling for a 4B model: a grammar per schema, and a validator before execution.

Finding 05 is unusually specific about what makes this work at this scale. Free-form
function calling is mediocre - Qwen3-4B scores 62% on BFCL - but schema-enforced
decoding with validator-first execution takes a 4B-class model to =>97%. So nothing
here asks the model to produce well-formed JSON, or to pick a tool that exists, or to
stay inside the range of extracts it was given. The grammar makes all three
impossible to get wrong, and what the model is left deciding is the only thing it is
actually good at: which of four things the user asked for.

The tool set is four. That is not minimalism for its own sake - the same finding puts
the cost of adding semantically-related distractor tools at 1-8% absolute, so a tool
that is not needed for Linux documentation is a tool that makes the other four worse.
`scripts/eval_tools.py --distractors` measures that on this corpus rather than
trusting the number.

The safety model has one rule that no configuration can relax: **only a tool marked
`auto_ok` may run without a human saying yes, and `propose_command` is never
`auto_ok`.** A proposed command is a string for a person to read. The eval asserts
this as its exit gate, with a confirmer that always refuses, and counts anything that
ran anyway.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Param:
    name: str
    kind: str                      # "string" | "int" | "enum"
    values: tuple[str, ...] = ()   # for enum
    description: str = ""


@dataclass(frozen=True)
class Tool:
    name: str
    summary: str
    params: tuple[Param, ...]
    auto_ok: bool = False          # may run without confirmation, if execution is on
    cites: bool = True             # call must name the extract it rests on


# The four. `answer` and `refuse` are tools rather than a fallback path, so that
# "say nothing" is a decision the grammar can express and the eval can score, instead
# of the absence of a decision.
TOOLS: tuple[Tool, ...] = (
    Tool("answer", "Explain something. Use ONLY when the request is a question about "
                   "how something works and asks for no action.",
         (Param("text", "string", description="the answer, one or two sentences"),)),
    Tool("propose_command", "Give the command that does it. Use whenever the request "
                            "asks for something to be done - anything phrased as an "
                            "instruction. Do not explain instead of answering with a "
                            "command; the explanation field is where explaining goes.",
         (Param("command", "string", description="the exact command"),
          Param("explanation", "string", description="what it does, one sentence"))),
    Tool("show_manpage", "Open a manual page.",
         (Param("page", "string", description="page name, e.g. tar"),
          Param("section", "enum", ("1", "5", "7", "8"))),
         auto_ok=True, cites=False),
    Tool("refuse", "The extracts do not support an answer or a command. Use when the "
                   "tool in question is not documented on this machine.",
         (), cites=False),
)

# Semantically related, plausible, and absent from this machine's documentation. Only
# used to measure the distractor cost the briefing warns about.
DISTRACTORS: tuple[Tool, ...] = (
    Tool("edit_config_file", "Edit a configuration file in place.",
         (Param("path", "string"), Param("change", "string"))),
    Tool("restart_service", "Restart a systemd service.", (Param("unit", "string"),)),
    Tool("install_package", "Install a package with the system package manager.",
         (Param("package", "string"),)),
    Tool("search_web", "Search the web for documentation.", (Param("query", "string"),)),
)


# ---------------------------------------------------------------- grammar


def _gbnf_value(p: Param) -> str:
    if p.kind == "enum":
        # Parenthesised: `|` binds at rule level in GBNF, so a bare alternation here
        # would silently turn one tool rule into several broken ones.
        return "(" + " | ".join(f'"\\"{v}\\""' for v in p.values) + ")"
    if p.kind == "int":
        return "int"
    return "str"


def grammar(tools: tuple[Tool, ...], n_extracts: int) -> str:
    """One GBNF alternative per tool, built from the schema rather than hand-written.

    Every field is required and ordered, so there is no partial call to validate
    around, and `cite` is restricted to the extracts actually supplied - the phase 2
    argument, applied to arguments instead of prose.
    """
    n = max(1, min(n_extracts, 9))
    refs = " | ".join(f'"{i}"' for i in range(1, n + 1))
    alts, rules = [], []
    for t in tools:
        rule = f"call-{t.name.replace('_', '-')}"
        alts.append(rule)
        parts = [f'"{{\\"tool\\": \\"{t.name}\\""']
        for p in t.params:
            parts.append(f'", \\"{p.name}\\": " {_gbnf_value(p)}')
        if t.cites:
            parts.append('", \\"cite\\": " ref')
        parts.append('"}"')
        rules.append(f"{rule} ::= " + " ".join(parts))
    body = "\n".join(rules)
    extra = "int ::= [0-9]+\n" if any(p.kind == "int" for t in tools for p in t.params) else ""
    return f"""root ::= {" | ".join(alts)}
{body}
ref ::= {refs}
{extra}str ::= "\\"" chr* "\\""
chr ::= [^"\\\\\\x00-\\x1F] | "\\\\" ["\\\\/bfnrt]
"""


def tool_prompt(tools: tuple[Tool, ...]) -> str:
    lines = ["You have exactly these tools. Reply with one call and nothing else."]
    for t in tools:
        args = ", ".join(f"{p.name}" + (f" ({'|'.join(p.values)})" if p.values else "")
                         for p in t.params)
        cite = ", cite" if t.cites else ""
        lines.append(f'- {t.name}({args}{cite}) - {t.summary}')
    lines.append("Cite the number of the extract the call rests on. "
                 "If the extracts do not document the tool the user named, use refuse.")
    return "\n".join(lines)


SYSTEM = """You help with the Linux documentation installed on this machine.
Use only the manual page extracts provided; never use knowledge from outside them.

{tools}"""


def build_messages(request: str, chunks: list[dict], toolset=TOOLS) -> list[dict]:
    parts = [f"[{i}] {c['doc_id']}\n{c.get('prefix','')}{c['text']}"
             for i, c in enumerate(chunks, 1)]
    context = "\n\n".join(parts) if parts else "(no extracts found)"
    return [
        {"role": "system", "content": SYSTEM.format(tools=tool_prompt(toolset))},
        {"role": "user", "content": f"Manual page extracts:\n\n{context}\n\n"
                                    f"Request: {request}"},
    ]


# ---------------------------------------------------------------- validation

READ_ONLY = frozenset("""ls cat head tail wc stat file df du ps ss ip date uptime free
who id groups uname hostname printenv env find grep egrep fgrep sort uniq cut awk sed
journalctl systemctl man dpkg apt-cache lsblk mount""".split())

# Executing any of these unasked is the failure this phase exists to prevent.
DESTRUCTIVE = frozenset("""rm rmdir shred truncate dd mkfs mkfs.ext4 mkfs.ext3 mke2fs
mkswap fdisk parted sfdisk wipefs chmod chown chgrp chattr userdel usermod useradd
groupadd groupdel passwd kill killall pkill reboot shutdown halt poweroff init swapoff
umount tee mv install ln""".split())

# Subcommands that make an otherwise read-only binary change the system.
DESTRUCTIVE_SUBCOMMANDS = {
    "systemctl": frozenset("start stop restart reload enable disable mask unmask "
                           "isolate kill set-property daemon-reload".split()),
    "mount": frozenset(),          # mount with no -o ro still changes system state
    "apt": frozenset("install remove purge upgrade".split()),
}

_UNSAFE_SHELL = re.compile(r"[;&|`$><]|\$\(|&&|\|\|")


@dataclass
class Call:
    tool: str
    args: dict
    cite: int | None = None
    risk: str = "unknown"          # "read-only" | "destructive" | "unknown"
    problems: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.problems

    @property
    def needs_confirmation(self) -> bool:
        """True unless the tool is auto_ok *and* nothing about it looks risky."""
        tool = BY_NAME.get(self.tool)
        return not (tool and tool.auto_ok and self.risk == "read-only" and self.valid)


BY_NAME = {t.name: t for t in TOOLS + DISTRACTORS}


def classify(command: str) -> str:
    """Risk of a proposed command, erring towards `destructive` on anything unclear.

    Deliberately crude and deliberately pessimistic. This decides whether a human is
    asked before something runs, so the only expensive mistake is calling a
    destructive command safe - and `unknown` is treated as needing confirmation
    anyway, which makes the safe default free.
    """
    if _UNSAFE_SHELL.search(command):
        return "destructive"       # a pipeline can hide anything; never auto-run one
    try:
        parts = shlex.split(command)
    except ValueError:
        return "destructive"
    if not parts:
        return "unknown"
    binary = parts[0].rsplit("/", 1)[-1]
    if binary in ("sudo", "doas", "env"):
        return "destructive"
    sub = next((p for p in parts[1:] if not p.startswith("-")), None)
    if binary in DESTRUCTIVE_SUBCOMMANDS:
        if sub in DESTRUCTIVE_SUBCOMMANDS[binary]:
            return "destructive"
        return "read-only" if binary in READ_ONLY else "unknown"
    if binary in DESTRUCTIVE:
        return "destructive"
    if binary in READ_ONLY:
        return "read-only"
    return "unknown"


def parse_call(text: str, tools: tuple[Tool, ...], n_extracts: int,
               known_pages: set[str] | None = None) -> Call:
    """Validate a decoded call. Runs even though the grammar should guarantee shape.

    The grammar is a property of how the text was produced, not of the text. A caller
    that forgets to pass it, a future model served without it, or a replayed log all
    arrive here as ordinary strings, and this is the layer that decides whether
    anything runs.
    """
    names = {t.name for t in tools}
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return Call("", {}, problems=[f"not JSON: {e.msg}"])
    if not isinstance(obj, dict):
        return Call("", {}, problems=["call is not an object"])

    name = obj.get("tool")
    if name not in names:
        return Call(str(name), {}, problems=[f"unknown tool {name!r}"])
    spec = BY_NAME[name]
    call = Call(name, {})

    for p in spec.params:
        if p.name not in obj:
            call.problems.append(f"missing argument {p.name!r}")
            continue
        v = obj[p.name]
        if p.kind == "enum" and str(v) not in p.values:
            call.problems.append(f"{p.name}={v!r} not one of {p.values}")
        elif p.kind == "string" and (not isinstance(v, str) or not v.strip()):
            call.problems.append(f"{p.name} must be a non-empty string")
        else:
            call.args[p.name] = v
    for extra in set(obj) - {p.name for p in spec.params} - {"tool", "cite"}:
        call.problems.append(f"unexpected argument {extra!r}")

    if spec.cites:
        cite = obj.get("cite")
        try:
            cite = int(cite)
        except (TypeError, ValueError):
            call.problems.append("cite must be a number")
            cite = None
        if cite is not None and not 1 <= cite <= n_extracts:
            call.problems.append(f"cite {cite} outside 1..{n_extracts}")
            cite = None
        call.cite = cite

    if name == "propose_command":
        call.risk = classify(call.args.get("command", "")) if call.valid else "unknown"
    elif name == "show_manpage":
        page, sec = str(call.args.get("page", "")), str(call.args.get("section", ""))
        # The model often writes the page as it appears in an extract - `glob.7`,
        # `lslocks.8` - which already carries the section. That call is right and
        # only spelled differently, so the suffix is normalised away rather than
        # rejected. Existence is still checked against the corpus afterwards.
        if sec and page.endswith(f".{sec}"):
            page = page[: -(len(sec) + 1)]
            call.args["page"] = page
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", page):
            call.problems.append(f"page {page!r} is not a page name")
        elif known_pages is not None and f"{page}.{sec}" not in known_pages:
            call.problems.append(f"{page}({sec}) is not a page on this machine")
        call.risk = "read-only" if call.valid else "unknown"
    else:
        call.risk = "read-only"
    return call


# ---------------------------------------------------------------- execution


def execute(call: Call, confirm=None, allow_execution: bool = False) -> dict:
    """Run a call, or explain why it was not run. Nothing runs without passing here.

    `confirm` is asked for anything that needs it and its answer is obeyed; a missing
    confirmer means no. `allow_execution` gates even the auto_ok path, so the default
    posture of the whole system is that it proposes and does not act.
    """
    if not call.valid:
        return {"ran": False, "reason": "invalid call", "problems": call.problems}
    if not allow_execution:
        return {"ran": False, "reason": "execution disabled"}
    if call.needs_confirmation:
        if confirm is None or not confirm(call):
            return {"ran": False, "reason": "not confirmed", "risk": call.risk}
        # Even confirmed, a proposed command is handed back rather than run: this
        # system's job is to be right about the flag, not to be a shell.
        return {"ran": False, "reason": "confirmed, left for the user to run",
                "risk": call.risk}
    if call.tool != "show_manpage":
        return {"ran": False, "reason": "no executor for this tool"}
    page, sec = call.args["page"], call.args["section"]
    out = subprocess.run(["man", sec, page], capture_output=True, timeout=30,
                         env={"PATH": "/usr/bin:/bin", "MANWIDTH": "100",
                              "LC_ALL": "C.UTF-8"})
    return {"ran": True, "reason": "read-only, auto-approved",
            "output": out.stdout.decode("utf-8", "replace")[:2000]}
