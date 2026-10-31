"""Terminal output helpers (colour aware)."""
import os
import sys
from dataclasses import dataclass


def color_on() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def paint(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if color_on() else s


def dim(s: str) -> str:
    return paint("2", s)


def bold(s: str) -> str:
    return paint("1", s)


def red(s: str) -> str:
    return paint("31", s)


def green(s: str) -> str:
    return paint("32", s)


def yellow(s: str) -> str:
    return paint("33", s)


def blue(s: str) -> str:
    return paint("34", s)


def magenta(s: str) -> str:
    return paint("35", s)


def cyan(s: str) -> str:
    return paint("36", s)


def out(s: str = "", end: str = "\n") -> None:
    print(s, end=end)
    sys.stdout.flush()


def err(msg: str) -> None:
    sys.stdout.flush()
    sys.stderr.write(red(f"錯誤: {msg}") + "\n")


def short(text: str, n: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


@dataclass
class Event:
    """A runtime event: graph nodes, tools, delegation, loops, warnings."""
    kind: str
    agent: str = ""
    node: str = ""
    model: str = ""
    name: str = ""
    detail: str = ""
    target: str = ""
    task: str = ""
    msg: str = ""
    src: str = ""
    dst: str = ""
    output: str = ""
    secs: float = 0.0
    chars: int = 0
    rounds: int = 0
    limit: int = 0
    step: int = 0


def show_event(e: Event, verbose: bool) -> None:
    """Render a runtime event as a progress line."""
    a = e.agent
    if e.kind == "node_start":
        out(cyan(f"▶ {e.node}") + dim(f"  ({a} · {e.model})"))
    elif e.kind == "node_done":
        out(green(f"✔ {e.node}") + dim(f"  {e.secs:.1f}s · {e.chars} chars"))
        if verbose and e.output != "":
            out(dim(e.output))
    elif e.kind == "tool":
        out(dim(f"  ⚙ {a}: {e.name} {short(e.detail, 70)}"))
    elif e.kind == "delegate":
        out(magenta(f"  ↪ {a} → {e.target}: ") + dim(short(e.task, 70)))
    elif e.kind == "delegate_done":
        out(magenta(f"  ↩ {e.target} 完成") + dim(f"  ({e.chars} chars)"))
    elif e.kind == "loop":
        out(yellow(f"↺ {e.src} → {e.dst}  (第 {e.rounds}/{e.limit} 輪)"))
    elif e.kind == "warn":
        out(yellow(f"! {e.msg}"))
    elif e.kind == "llm" and verbose:
        out(dim(f"  … {a} 呼叫 {e.model} (step {e.step})"))
