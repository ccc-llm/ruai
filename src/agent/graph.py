"""Graph engineering: define agent workflows as graphs (the executor is in runtime.py).

Graph JSON:
{
  "description": "...",
  "nodes": {
    "design":    {"agent": "architect", "prompt": "Task: {{task}}"},
    "implement": {"agent": "coder", "prompt": "{{design}}\\n{{feedback}}", "model": "glm"},
    "chapters":  {"agent": "writer", "map": "outline", "prompt": "Write {{item.title}} to {{item.file}}"}
  },
  "edges": [
    ["design", "implement"],                                            # dependency
    {"from": "review", "to": "implement", "when": "REVISE", "max": 2}   # conditional loop-back
  ]
}

Template variables: {{task}}, {{<node name>}} (that node's output), {{deps}} (all upstream outputs),
{{feedback}} (output of the node that looped back here), {{item}}/{{item.field}}/{{index}} in map nodes.

Execution: nodes whose dependencies are done run (one after another in ruai). A "map" node runs its
agent once per item of the list found in its source node's output (```json block or bullet list).
A conditional edge whose regex `when` matches the source output re-runs the target and everything
downstream of it, up to `max` times.
"""
import re
from dataclasses import dataclass

from ruai.json import Json
from rust.regex import Regex

DEFAULT_PROMPT = "{{task}}\n\n{{deps}}\n\n{{feedback}}"


class GraphError(Exception):
    pass


@dataclass
class Edge:
    src: str
    dst: str
    when: str = ""
    limit: int = 3

    def __str__(self) -> str:
        if self.when == "":
            return f"{{'from': '{self.src}', 'to': '{self.dst}'}}"
        return f"{{'from': '{self.src}', 'to': '{self.dst}', 'when': '{self.when}', 'max': {self.limit}}}"


def edges(graph: Json) -> list[Edge]:
    out: list[Edge] = []
    for e in graph.get("edges").items():
        if e.kind() == "list":
            out.append(Edge(e.at(0).text(), e.at(1).text()))
        else:
            out.append(Edge(e.get("from").text(), e.get("to").text(), e.get_str("when", ""),
                            e.get("max").int_or(3)))
    return out


def node_agent(name: str, node: Json) -> str:
    return node.get_str("agent", name)


def dependencies(graph: Json) -> dict[str, list[str]]:
    deps: dict[str, list[str]] = {}
    nodes = graph.get("nodes")
    for n in nodes.keys():
        deps[n] = []
    for e in edges(graph):
        if e.when == "" and e.dst in deps and e.src not in deps[e.dst]:
            deps[e.dst].append(e.src)
    for n in nodes.keys():
        src = nodes.get(n).get_str("map", "")
        if src != "" and src not in deps[n]:
            deps[n].append(src)
    return deps


def visit(n: str, path: list[str], deps: dict[str, list[str]], state: dict[str, int], order: list[str]) -> None:
    s = state[n] if n in state else 0
    if s == 1:
        raise GraphError("依賴關係有循環: " + " → ".join(path + [n]) + "(迴圈請用帶 when 的條件邊)")
    if s == 2:
        return
    state[n] = 1
    if n in deps:
        for d in deps[n]:
            visit(d, path + [n], deps, state, order)
    state[n] = 2
    order.append(n)


def topo_order(graph: Json) -> list[str]:
    deps = dependencies(graph)
    order: list[str] = []
    state: dict[str, int] = {}
    for n in graph.get("nodes").keys():
        visit(n, [], deps, state, order)
    return order


def descendants(graph: Json, start: str) -> list[str]:
    children: dict[str, list[str]] = {}
    for n, ds in dependencies(graph).items():
        for d in ds:
            if d not in children:
                children[d] = []
            children[d].append(n)
    seen: list[str] = []
    stack = [start]
    while len(stack) > 0:
        x = stack.pop()
        if x in children:
            for c in children[x]:
                if c not in seen:
                    seen.append(c)
                    stack.append(c)
    return seen


def validate(graph: Json, agents: Json) -> list[str]:
    errors: list[str] = []
    nodes = graph.get("nodes")
    if nodes.size() == 0:
        errors.append("graph 沒有任何節點")
    for n in nodes.keys():
        node = nodes.get(n)
        agent = node_agent(n, node)
        if not agents.has(agent):
            errors.append(f"節點 {n}: 未知的 agent '{agent}'")
        src = node.get_str("map", "")
        if src != "" and not nodes.has(src):
            errors.append(f"節點 {n}: map 來源 '{src}' 不存在")
    for e in edges(graph):
        for k, v in [("from", e.src), ("to", e.dst)]:
            if not nodes.has(v):
                errors.append(f"邊 {e}: 節點 '{v}' 不存在")
        if e.when != "":
            try:
                Regex.new(e.when)
            except ValueError as ex:
                errors.append(f"邊 {e}: when 正規表示式錯誤 {ex}")
    if len(errors) == 0:
        try:
            topo_order(graph)
        except GraphError as ex:
            errors.append(str(ex))
    return errors


def lookup(key: str, variables: dict[str, Json]) -> Json:
    parts = key.split(".")
    if parts[0] not in variables:
        return Json.null()
    cur = variables[parts[0]]
    for part in parts[1:]:
        if cur.kind() == "dict" and cur.has(part):
            cur = cur.get(part)
        else:
            return Json.null()
    return cur


def render(template: str, variables: dict[str, Json]) -> str:
    out: list[str] = []
    pos = 0
    for m in re.finditer(r"\{\{\s*([\w.\-]+)\s*\}\}", template):
        out.append(template[pos:m.start()])
        v = lookup(m.group(1), variables)
        if v.is_null():
            out.append(m.group(0))
        elif v.kind() == "str":
            out.append(v.text())
        else:
            out.append(v.dumps(indent=2, ensure_ascii=False))
        pos = m.end()
    out.append(template[pos:])
    return "".join(out)


def extract_items(text: str) -> list[Json]:
    """Find a list in an agent's output: last ```json block, whole text as JSON, or a bullet list."""
    blocks = re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    blocks.reverse()
    blocks.append(text)
    for block in blocks:
        try:
            data = Json.loads(block)
        except ValueError:
            continue
        if data.kind() == "dict":
            found = Json.null()
            for v in data.items():
                if v.kind() == "list":
                    found = v
                    break
            data = found
        if data.kind() == "list" and data.size() > 0:
            return data.items()
    items = re.findall(r"^\s*(?:[-*]|\d+[.)])\s+(.+)$", text, re.M)
    return [Json.of_str(i.strip()) for i in items if i.strip() != ""]


def mermaid(graph: Json) -> str:
    lines = ["flowchart TD"]
    nodes = graph.get("nodes")
    for n in nodes.keys():
        node = nodes.get(n)
        label = f"{n}<br/>{node_agent(n, node)}" + (" ×map" if node.get_str("map", "") != "" else "")
        lines.append(f'  {n}["{label}"]')
    for e in edges(graph):
        if e.when != "":
            lines.append(f'  {e.src} -. "{e.when} (max {e.limit})" .-> {e.dst}')
        else:
            lines.append(f"  {e.src} --> {e.dst}")
    for n in nodes.keys():
        src = nodes.get(n).get_str("map", "")
        if src != "":
            lines.append(f"  {src} ==>|items| {n}")
    return "\n".join(lines)


def describe(graph: Json) -> str:
    """Human-readable text rendering of a graph."""
    deps = dependencies(graph)
    out: list[str] = []
    desc = graph.get_str("description", "")
    if desc != "":
        out.append(desc)
    nodes = graph.get("nodes")
    for n in topo_order(graph):
        node = nodes.get(n)
        extra: list[str] = []
        if node.get_str("map", "") != "":
            extra.append(f"map over {node.get_str('map', '')}")
        if node.get_str("model", "") != "":
            extra.append(f"model={node.get_str('model', '')}")
        after = f"  ← {', '.join(deps[n])}" if len(deps[n]) > 0 else "  (start)"
        ex = " " + " ".join(extra) if len(extra) > 0 else ""
        out.append(f"  [{n}] agent={node_agent(n, node)}{ex}{after}")
    for e in edges(graph):
        if e.when != "":
            out.append(f"  ↺ {e.src} → {e.dst} when /{e.when}/ (max {e.limit})")
    return "\n".join(out)
