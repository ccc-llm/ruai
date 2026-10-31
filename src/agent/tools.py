"""Text-protocol tools.

Agents call tools by writing XML-like tags in their reply, which works with every model
(no native function-calling needed):

    <read_file path="a.py"/>
    <write_file path="a.py">...</write_file>
    <edit_file path="a.py"><old>x</old><new>y</new></edit_file>
    <list_files path="." depth="2"/>
    <search pattern="def main" path="." glob="*.py"/>
    <shell timeout="60">pytest -q</shell>
    <delegate agent="coder">task</delegate>

Results are returned as <tool_result name=... status="ok|error">...</tool_result>.
(`delegate` is run by the runtime, see runtime.py.)
"""
import fnmatch
import html
import os
import re
import subprocess
import sys
from dataclasses import dataclass

from rust.regex import Regex

import ui

MAX_RESULT = 30000


def tool_doc(name: str) -> str:
    if name == "read_file":
        return '<read_file path="src/app.py"/> — read a file (optional start="1" end="200" for a line range)'
    if name == "write_file":
        return ('<write_file path="docs/a.md">full content</write_file> — create/overwrite a file; '
                'append="true" appends instead')
    if name == "edit_file":
        return ('<edit_file path="src/app.py"><old>exact existing text</old><new>replacement</new></edit_file> '
                '— each <old> must match exactly once; several <old>/<new> pairs allowed')
    if name == "list_files":
        return '<list_files path="." depth="2"/> — show a directory tree'
    if name == "search":
        return '<search pattern="regex" path="." glob="*.py"/> — grep file contents'
    if name == "shell":
        return ('<shell timeout="120">command</shell> — run a shell command in the workspace root '
                '(the user may refuse)')
    return ('<delegate agent="name">complete, self-contained brief</delegate> — hand a subtask to a team '
            'member and get its final report; several delegate tags in one reply run in parallel')


def tool_names() -> list[str]:
    return ["read_file", "write_file", "edit_file", "list_files", "search", "shell", "delegate"]


def skip_dir(name: str) -> bool:
    return name in [".git", ".agge", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache",
                    ".pytest_cache", "dist", "build", ".idea", ".DS_Store"]


class ToolError(Exception):
    pass


@dataclass
class Call:
    name: str
    attrs: dict[str, str]
    body: str = ""
    has_body: bool = False
    closed: bool = True

    def attr(self, key: str, default: str) -> str:
        return self.attrs[key] if key in self.attrs else default


def parse_calls(text: str) -> list[Call]:
    calls: list[Call] = []
    pos = 0
    while True:
        m = re.search(r'<(read_file|write_file|edit_file|list_files|search|shell|delegate)((?:\s+[\w-]+\s*=\s*"[^"]*")*)\s*(/?)>',
                      text[pos:])
        if m is None:
            return calls
        name = m.group(1)
        attrs: dict[str, str] = {}
        for k, v in re.findall(r'([\w-]+)\s*=\s*"([^"]*)"', m.group(2)):
            attrs[k] = html.unescape(v)
        end_tag = pos + m.end()
        if m.group(3) == "/":
            calls.append(Call(name, attrs))
            pos = end_tag
            continue
        close = f"</{name}>"
        end = text.find(close, end_tag)
        if end < 0:
            calls.append(Call(name, attrs, text[end_tag:], True, False))
            return calls
        calls.append(Call(name, attrs, text[end_tag:end], True))
        pos = end + len(close)


def strip_tool_tags(text: str) -> str:
    """Remove tool tags from a reply (for display / summaries)."""
    out: list[str] = []
    pos = 0
    while True:
        m = re.search(r'<(read_file|write_file|edit_file|list_files|search|shell|delegate)\b[^>]*?(/?)>', text[pos:])
        if m is None:
            break
        start = pos + m.start()
        after = pos + m.end()
        if m.group(2) == "/":
            out.append(text[pos:start])
            pos = after
            continue
        close = f"</{m.group(1)}>"
        end = text.find(close, after)
        if end < 0:
            out.append(text[pos:after])
            pos = after
            continue
        out.append(text[pos:start])
        pos = end + len(close)
    out.append(text[pos:])
    return "".join(out).strip()


def trim_nl(s: str) -> str:
    if s.startswith("\r\n"):
        return s[2:]
    if s.startswith("\n"):
        return s[1:]
    return s


def under(p: str, d: str) -> bool:
    return p == d or p.startswith(d.rstrip("/") + "/")


def add_dir(dirs: list[str], root: str, d: str) -> str:
    p = os.path.expanduser(d)
    p = os.path.realpath(p if os.path.isabs(p) else os.path.join(root, p))
    if p not in dirs:
        dirs.append(p)
    return p


class Workspace:
    def __init__(self, root: str, dirs: list[str]) -> None:
        real = os.path.realpath(os.path.expanduser(root))
        self.root = real
        self.dirs: list[str] = [real]
        for d in dirs:
            add_dir(self.dirs, real, d)

    def add(self, d: str) -> str:
        root = self.root
        return add_dir(self.dirs, root, d)

    def resolve(self, path: str) -> str:
        if path == "":
            raise ToolError("missing path attribute")
        p = os.path.expanduser(path)
        p = os.path.realpath(p if os.path.isabs(p) else os.path.join(self.root, p))
        if not any(under(p, d) for d in self.dirs):
            raise ToolError(f"path {path} is outside the allowed directories: "
                            + ", ".join(self.dirs) + " (the user can allow more with /add-dir)")
        return p

    def rel(self, p: str) -> str:
        if under(p, self.root):
            return os.path.relpath(p, self.root)
        return p


class Approver:
    """Asks the user before risky actions. mode: ask | auto | deny"""

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def confirm(self, question: str) -> bool:
        if self.mode == "auto":
            return True
        if self.mode == "deny" or not sys.stdin.isatty():
            return False
        try:
            ans = input(ui.yellow(f"\n? {question}\n  允許? [y]es / [N]o / [a]lways: ")).strip().lower()
        except EOFError:
            return False
        if ans in ["a", "always"]:
            self.mode = "auto"
            return True
        return ans in ["y", "yes"]


def read_text(p: str) -> str:
    with open(p, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return text


def write_text(p: str, text: str, append: bool) -> None:
    if append:
        with open(p, "a", encoding="utf-8") as f:
            f.write(text)
    else:
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)


def run_tool(call: Call, ws: Workspace, approver: Approver, agent: str) -> str:
    if call.name == "read_file":
        return read_file(call, ws)
    if call.name == "write_file":
        return write_file(call, ws)
    if call.name == "edit_file":
        return edit_file(call, ws)
    if call.name == "list_files":
        return list_files(call, ws)
    if call.name == "search":
        return search(call, ws)
    if call.name == "shell":
        return shell(call, ws, approver, agent)
    raise ToolError(f"unknown tool {call.name}")


def split_keep(text: str) -> list[str]:
    """The lines of text with their line ends (`splitlines(keepends=True)` for \n)."""
    parts = text.split("\n")
    lines = [x + "\n" for x in parts[:-1]]
    if parts[-1] != "":
        lines.append(parts[-1])
    return lines


def read_file(call: Call, ws: Workspace) -> str:
    path = call.attr("path", "")
    p = ws.resolve(path)
    if os.path.isdir(p):
        raise ToolError(f"{path} is a directory; use list_files")
    if not os.path.exists(p):
        raise ToolError(f"{path} does not exist")
    lines = split_keep(read_text(p))
    start = int(call.attr("start", "1"))
    end = int(call.attr("end", str(len(lines))))
    body = "".join(lines[start - 1:end])
    if start > 1 or end < len(lines):
        return f"(lines {start}-{min(end, len(lines))} of {len(lines)})\n{body}"
    return body


def write_file(call: Call, ws: Workspace) -> str:
    p = ws.resolve(call.attr("path", ""))
    content = trim_nl(call.body)
    if content != "" and not content.endswith("\n"):
        content += "\n"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    append = call.attr("append", "").lower() == "true"
    write_text(p, content, append)
    return f"{'appended' if append else 'wrote'} {len(content)} chars to {ws.rel(p)}"


def edit_file(call: Call, ws: Workspace) -> str:
    path = call.attr("path", "")
    p = ws.resolve(path)
    if not os.path.exists(p):
        raise ToolError(f"{path} does not exist; use write_file to create it")
    text = read_text(p)
    pairs = re.findall(r"<old>(.*?)</old>\s*<new>(.*?)</new>", call.body, re.S)
    if len(pairs) == 0:
        raise ToolError("edit_file needs <old>...</old><new>...</new> pairs")
    for old0, new0 in pairs:
        old = trim_nl(old0)
        new = trim_nl(new0)
        if old.endswith("\n") and new.endswith("\n"):
            old = old[:-1]
            new = new[:-1]
        n = text.count(old)
        if n == 0:
            raise ToolError(f"<old> text not found in {path} (no change made); "
                            f"read the file and copy the text exactly:\n{old[:300]}")
        if n > 1:
            raise ToolError(f"<old> text occurs {n} times in {path}; include more context")
        text = text.replace(old, new, 1)
    write_text(p, text, False)
    return f"applied {len(pairs)} edit(s) to {ws.rel(p)}"


def walk_files(d: str, out: list[str]) -> None:
    """The files under d, like os.walk (top-down, sorted, skipping SKIP_DIRS)."""
    names = sorted(os.listdir(d))
    subdirs: list[str] = []
    for n in names:
        p = os.path.join(d, n)
        if os.path.isdir(p):
            if not skip_dir(n):
                subdirs.append(p)
        elif not skip_dir(n):
            out.append(p)
    for s in subdirs:
        walk_files(s, out)


def tree(d: str, level: int, depth: int, lines: list[str]) -> None:
    names = sorted(os.listdir(d), key=lambda n: (os.path.isfile(os.path.join(d, n)), n))
    for n in names:
        if skip_dir(n):
            continue
        if len(lines) >= 500:
            return
        p = os.path.join(d, n)
        lines.append("  " * level + n + ("/" if os.path.isdir(p) else ""))
        if os.path.isdir(p) and level + 1 < depth:
            tree(p, level + 1, depth, lines)


def list_files(call: Call, ws: Workspace) -> str:
    root = ws.resolve(call.attr("path", "."))
    if not os.path.isdir(root):
        raise ToolError(f"{call.attr('path', '.')} is not a directory")
    lines: list[str] = []
    tree(root, 0, int(call.attr("depth", "2")), lines)
    return "\n".join(lines) if len(lines) > 0 else "(empty directory)"


def search(call: Call, ws: Workspace) -> str:
    try:
        rx = Regex.new(call.attr("pattern", ""))
    except ValueError as e:
        raise ToolError(f"bad regex: {e}")
    root = ws.resolve(call.attr("path", "."))
    glob = call.attr("glob", "")
    files: list[str] = []
    if os.path.isfile(root):
        files.append(root)
    else:
        walk_files(root, files)
    hits: list[str] = []
    for f in files:
        if glob != "" and not fnmatch.fnmatch(os.path.basename(f), glob):
            continue
        text = read_text(f)
        if "�" in text:
            continue
        i = 0
        for line in text.splitlines():
            i += 1
            if rx.is_match(line):
                hits.append(f"{ws.rel(f)}:{i}: {line.strip()[:200]}")
                if len(hits) >= 200:
                    return "\n".join(hits) + "\n... (more matches omitted)"
    return "\n".join(hits) if len(hits) > 0 else "no matches"


def shell(call: Call, ws: Workspace, approver: Approver, agent: str) -> str:
    cmd = call.body.strip()
    if cmd == "":
        raise ToolError("empty command")
    if not approver.confirm(f"{agent} 要執行 shell(在 {ws.root}):\n  {cmd}"):
        raise ToolError("the user denied this command")
    try:
        r = subprocess.run(cmd, shell=True, cwd=ws.root, capture_output=True, text=True,
                           timeout=float(call.attr("timeout", "120")))
    except OSError as e:
        raise ToolError(f"cannot run the command: {e}")
    except Exception:
        raise ToolError("command timed out")
    return f"exit code {r.returncode}\n{r.stdout}{r.stderr}".rstrip()
