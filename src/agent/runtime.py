"""Agents and the runtime that wires them to models, tools and each other; the graph executor.

ruai runs everything in one thread: several <delegate> tags in one reply, the nodes of a
graph that are ready together and the items of a map node run one after another."""
import html
import time

from ruai.json import Json
from rust.regex import Regex

import tools
import ui
from graph import DEFAULT_PROMPT, Edge, GraphError, dependencies, descendants, edges, extract_items, node_agent, render, topo_order
from project import Project, ProjectError, RunLog, append_event
from providers import LLM, Msg
from tools import Approver, Call, ToolError, Workspace

BASE_PROMPT = """You are "{name}", an agent in agge — a multi-agent environment for software engineering and long-form writing.

{system}

Workspace root: {root}
Allowed directories: {dirs}
Today: {date}
{lang}"""

TOOLS_PROMPT = """
## Tools
You act on the workspace by writing tool tags directly in your reply (never inside code fences).
After your reply, the environment executes every tag in order and sends back <tool_result> blocks.
Continue (inspect → change → verify) until the task is done, then give a final answer with NO tool tags.
{docs}
Paths are relative to the workspace root. Always write complete file contents."""

NUDGE = ("Your reply ended right before the action you announced, without any tool tag. "
         "Write the tool tag(s) now, or give your final answer if the task is complete.")
EMPTY_NUDGE = "Your previous reply was empty. Please continue with the task."


def node_vars(name: str, task: str, deps: list[str], outputs: dict[str, str],
              feedback: dict[str, str]) -> dict[str, Json]:
    v: dict[str, Json] = {"task": Json.of_str(task), "input": Json.of_str(task)}
    for k, o in outputs.items():
        v[k] = Json.of_str(o)
    parts: list[str] = []
    for d in deps:
        parts.append(f"## Output of `{d}`\n{outputs[d] if d in outputs else ''}")
    v["deps"] = Json.of_str("\n\n".join(parts))
    v["feedback"] = Json.of_str(feedback[name] if name in feedback else "")
    return v


class Agent:
    def __init__(self, name: str, spec: Json, model_name: str, llm: LLM, depth: int, max_depth: int) -> None:
        self.name = name
        self.spec = spec
        self.depth = depth
        self.model_name = model_name
        self.llm = llm
        allowed = [t for t in spec.get("tools").strs() if t in tools.tool_names()]
        if depth >= max_depth and "delegate" in allowed:
            allowed.remove("delegate")
        self.tools = allowed


class Runtime:
    """Everything an agent needs: project config, workspace, approval policy, model choices."""

    def __init__(self, project: Project, workspace: Workspace, approver: Approver, model: str,
                 model_map: dict[str, str], team: str, max_steps: int, language: str) -> None:
        cfg = project.config()
        self.project = project
        self.workspace = workspace
        self.approver = approver
        self.model = model
        self.model_map = model_map
        self.team = team
        self.max_steps = max_steps if max_steps > 0 else cfg.get("max_steps").int_or(40)
        self.max_depth = cfg.get("max_depth").int_or(2)
        self.language = language if language != "" else cfg.get_str("language", "")
        self.llms: dict[str, LLM] = {}
        self.printing = False
        self.verbose = False
        self.logging = False
        self.log_path = ""

    # ---------- events ----------
    def event(self, e: ui.Event) -> None:
        if self.logging:
            append_event(self.log_path, e)
        if self.printing:
            ui.show_event(e, self.verbose)

    def warn(self, agent: str, msg: str) -> None:
        self.event(ui.Event("warn", agent=agent, msg=msg))

    # ---------- agents and models ----------
    def agents(self) -> Json:
        return self.project.agents()

    def model_for(self, name: str) -> str:
        spec = self.agents().get(name)
        if name in self.model_map:
            return self.model_map[name]
        m = spec.get_str("model", "")
        if m != "":
            return m
        if self.model != "":
            return self.model
        return self.project.config().get_str("default_model", "echo")

    def llm(self, model_name: str) -> LLM:
        if model_name not in self.llms:
            self.llms[model_name] = LLM(self.project.resolve_model(model_name))
        return self.llms[model_name]

    def agent(self, name: str, model: str, depth: int) -> Agent:
        specs = self.agents()
        if not specs.has(name):
            raise ProjectError(f"沒有名為 '{name}' 的 agent(可用: {', '.join(specs.keys())})")
        model_name = model if model != "" else self.model_for(name)
        return Agent(name, specs.get(name), model_name, self.llm(model_name), depth, self.max_depth)

    def delegates_for(self, name: str) -> dict[str, str]:
        specs = self.agents()
        teams = self.project.teams()
        members: list[str] = []
        if teams.has(self.team) and teams.get(self.team).get_str("lead", "") == name:
            members = teams.get(self.team).get("members").strs()
        else:
            members = [n for n in specs.keys() if n not in [name, "lead", "assistant"]]
        out: dict[str, str] = {}
        for m in members:
            if specs.has(m):
                out[m] = specs.get(m).get_str("description", "")
        return out

    def system_prompt(self, a: Agent) -> str:
        ws = self.workspace
        lang = f"Write in {self.language}." if self.language != "" else "Reply in the same language the user writes in."
        text = (BASE_PROMPT.replace("{name}", a.name).replace("{system}", a.spec.get_str("system", ""))
                .replace("{root}", ws.root).replace("{dirs}", ", ".join(ws.dirs))
                .replace("{date}", time.strftime("%Y-%m-%d")).replace("{lang}", lang))
        if len(a.tools) > 0:
            text += TOOLS_PROMPT.replace("{docs}", "\n".join([f"- {tools.tool_doc(t)}" for t in a.tools]))
        if "delegate" in a.tools:
            team = self.delegates_for(a.name)
            text += "\n\n## Your team\n" + "\n".join([f"- {n}: {d}" for n, d in team.items()])
        return text

    # ---------- the agent loop ----------
    def run(self, a: Agent, messages: list[Msg], stream: bool, think: int) -> str:
        """Agent loop: call the model, execute tool tags, feed results back. Extends `messages`."""
        system = self.system_prompt(a)
        reply = ""
        nudges = 0
        for step in range(self.max_steps):
            self.event(ui.Event("llm", agent=a.name, model=a.llm.label, step=step))
            reply = a.llm.chat(system, messages, stream, think).strip()
            empty = reply == ""
            if empty:
                reply = "(empty reply)"
            messages.append(Msg("assistant", reply))
            calls = tools.parse_calls(reply) if len(a.tools) > 0 else []
            if len(calls) == 0:
                # Models sometimes return nothing, or announce an action ("Step 1: ask architect:")
                # and stop before writing the tag; nudge them to continue instead of finishing.
                r = reply.rstrip("*")
                stalled = len(a.tools) > 0 and (r.endswith(":") or r.endswith("："))
                if (empty or stalled) and nudges < 2:
                    nudges += 1
                    why = "回覆是空的" if empty else "宣告了動作但沒有寫出工具標籤"
                    self.warn(a.name, f"{a.name} {why},提醒它繼續")
                    messages.append(Msg("user", EMPTY_NUDGE if empty else NUDGE))
                    continue
                return reply
            if stream:
                ui.out()
            messages.append(Msg("user", self.run_calls(calls, a)))
        self.warn(a.name, f"{a.name} 達到最大步數 {self.max_steps}")
        return tools.strip_tool_tags(reply)

    def run_calls(self, calls: list[Call], a: Agent) -> str:
        """Execute calls in order; delegate calls run after the others."""
        results = ["" for c in calls]
        for i in range(len(calls)):
            if calls[i].name != "delegate":
                results[i] = self.execute(calls[i], a)
        for i in range(len(calls)):
            if calls[i].name == "delegate":
                results[i] = self.execute(calls[i], a)
        return "\n\n".join(results)

    def execute(self, call: Call, a: Agent) -> str:
        attrs = "".join([f' {k}="{html.escape(v)}"' for k, v in call.attrs.items()])
        status = "ok"
        out = ""
        try:
            if call.name not in a.tools:
                raise ToolError(f"agent '{a.name}' is not allowed to use {call.name}")
            if not call.closed:
                raise ToolError(f"<{call.name}> was not closed (reply truncated?). "
                                "Write long content in smaller parts with append=\"true\".")
            detail = call.attr("path", call.attr("agent", call.body.strip()))
            self.event(ui.Event("tool", agent=a.name, name=call.name, detail=detail))
            if call.name == "delegate":
                out = self.delegate_call(call, a)
            else:
                out = tools.run_tool(call, self.workspace, self.approver, a.name)
        except ToolError as e:
            out = str(e)
            status = "error"
        except ProjectError as e:
            out = f"ProjectError: {e}"
            status = "error"
        except OSError as e:
            out = f"OSError: {e}"
            status = "error"
        except ValueError as e:
            out = f"ValueError: {e}"
            status = "error"
        if len(out) > tools.MAX_RESULT:
            out = out[:tools.MAX_RESULT] + f"\n... [truncated, {len(out) - tools.MAX_RESULT} more chars]"
        return f'<tool_result name="{call.name}"{attrs} status="{status}">\n{out}\n</tool_result>'

    def delegate_call(self, call: Call, a: Agent) -> str:
        task = call.body.strip()
        if task == "":
            raise ToolError("empty delegate task")
        return self.delegate(a.name, call.attr("agent", ""), task, a.depth)

    def delegate(self, frm: str, target: str, task: str, depth: int) -> str:
        if not self.agents().has(target):
            raise ToolError(f"unknown agent '{target}'; team: {', '.join(self.delegates_for(frm).keys())}")
        self.event(ui.Event("delegate", agent=frm, target=target, task=task))
        sub = self.agent(target, "", depth + 1)
        out = self.run(sub, [Msg("user", task)], False, 0)
        self.event(ui.Event("delegate_done", agent=frm, target=target, chars=len(out)))
        return out

    # ---------- graphs ----------
    def run_graph(self, graph: Json, task: str, log: RunLog | None) -> dict[str, str]:
        """Run a graph's nodes in dependency order; conditional edges loop back. The outputs by node."""
        nodes = graph.get("nodes")
        deps = dependencies(graph)
        order = topo_order(graph)
        loops = [e for e in edges(graph) if e.when != ""]
        rounds = [0 for e in loops]
        outputs: dict[str, str] = {}
        feedback: dict[str, str] = {}
        done: list[str] = []
        while True:
            n = ""
            for m in order:
                if m not in done and all(d in done for d in deps.get(m, [])):
                    n = m
                    break
            if n == "":
                break
            variables = node_vars(n, task, deps.get(n, []), outputs, feedback)
            out = self.exec_node(n, nodes.get(n), variables, outputs, log)
            outputs[n] = out
            done.append(n)
            self.check_loops(n, out, graph, loops, rounds, done, feedback)
        missing = [n for n in nodes.keys() if n not in done]
        if len(missing) > 0:
            raise GraphError(f"以下節點無法執行: {', '.join(missing)}")
        return outputs

    def run_node_agent(self, name: str, node: Json, prompt: str, label: str, log: RunLog | None) -> str:
        agent_name = node_agent(name, node)
        model = node.get_str("model", "")
        if agent_name in self.model_map:
            model = self.model_map[agent_name]
        if name in self.model_map:
            model = self.model_map[name]
        a = self.agent(agent_name, model, 0)
        self.event(ui.Event("node_start", node=label, agent=agent_name, model=a.llm.label))
        t = time.time()
        out = self.run(a, [Msg("user", prompt)], False, 0)
        self.event(ui.Event("node_done", node=label, agent=agent_name, secs=time.time() - t, chars=len(out), output=out))
        if log is not None:
            log.save_output(label, f"<!-- agent: {agent_name} · model: {a.llm.label} -->\n"
                                   f"# Prompt\n\n{prompt}\n\n# Output\n\n{out}\n")
        return out

    def exec_node(self, name: str, node: Json, variables: dict[str, Json], outputs: dict[str, str],
                  log: RunLog | None) -> str:
        template = node.get_str("prompt", "")
        if template == "":
            template = DEFAULT_PROMPT
        src = node.get_str("map", "")
        if src == "":
            return self.run_node_agent(name, node, render(template, variables), name, log)
        items = extract_items(outputs[src] if src in outputs else "")
        if len(items) == 0:
            raise GraphError(f"節點 {name}: 無法從 '{src}' 的輸出取得清單(需要 ```json 陣列或條列清單)")
        items = items[:node.get("max_items").int_or(100)]
        outs: list[str] = []
        for i in range(len(items)):
            v = variables.copy()
            v["item"] = items[i]
            v["index"] = Json.of_str(str(i + 1))
            outs.append(self.run_node_agent(name, node, render(template, v), f"{name}[{i + 1}]", log))
        return "\n\n".join([f"### {name}[{i + 1}]\n{o}" for i, o in enumerate(outs)])

    def check_loops(self, n: str, out: str, graph: Json, loops: list[Edge], rounds: list[int],
                    done: list[str], feedback: dict[str, str]) -> None:
        for i in range(len(loops)):
            e = loops[i]
            if e.src != n or not Regex.new(e.when).is_match(out):
                continue
            if rounds[i] >= e.limit:
                self.warn("", f"{n} → {e.dst} 已達最大迴圈次數 {e.limit},繼續往下")
                continue
            rounds[i] += 1
            feedback[e.dst] = f"## Feedback from `{n}` (round {rounds[i]}) — address all of it\n{out}\n"
            self.event(ui.Event("loop", src=n, dst=e.dst, rounds=rounds[i], limit=e.limit))
            for m in [e.dst] + descendants(graph, e.dst):
                if m in done:
                    done.remove(m)
            return
