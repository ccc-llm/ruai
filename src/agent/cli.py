"""ruai agent — a git-style CLI for orchestrating LLM agent teams (agge, in py0).

    ruai agent                     開啟互動對話 (同 ruai agent chat)
    ruai agent <op> <args...>      執行子指令,見 ruai agent help
"""
import os
import sys

from ruai.json import Json

import ui
from graph import GraphError, describe, mermaid, validate
from project import Project, ProjectError, load_text, sorted_files
from providers import LLM, LLMError, Msg, providers
from repl import start_chat
from runtime import Runtime
from tools import Approver, ToolError, Workspace

VERSION = "3.4.0"

HELP = """用法: ruai agent <op> [選項]

ruai agent — 組織 LLM agent 團隊寫程式與寫書的瑞士刀 CLI

指令:
  init [dir] [-m MODEL]       建立 .agge/ 專案
  chat                        互動對話(預設)  -a AGENT -t TEAM -r [ID]
  ask [任務...]               單次提問 / 執行任務  -a AGENT -t TEAM --stream
  run <graph> [任務...]       執行 graph 工作流程
  code [任務...]              用開發團隊寫程式 (graph: code)  --stack STACK
  book [主題...]              用寫作團隊寫書 (graph: book)  -c N --audience A --style S
  graph <op> [name]           管理 graph: list|show|mermaid|validate|new|rm|add-node|add-edge|rm-node
  agent <op> [name]           管理 agent: list|show|add|rm
  team <op> [name]            管理團隊: list|add|rm
  model <op> [name]           管理模型: list|add|rm|default|test
  provider                    列出支援的 provider
  dir <op> [path]             管理 agent 可存取目錄: list|add|rm
  config [key [value]]        讀寫設定
  log [run|last] [node]       執行紀錄  -n N
  session <op> [id]           對話 session: list|show
  status                      顯示專案狀態
  help                        顯示說明

執行選項 (chat/ask/run/code/book):
  -m, --model M      預設模型(別名或 provider:model)
  -M, --assign A=M   指定某 agent/節點使用的模型,可重複
  -y, --yes          自動核准 shell 指令
  --add-dir DIR      允許存取的額外目錄
  --lang L           輸出語言,如 zh-TW、English
  --max-steps N      每個 agent 最多工具回合數
  -v, --verbose      顯示每個節點的輸出
  -o, --out DIR      工作目錄(輸出檔案寫到這裡)
graph 選項 (run/code/book):
  -g, --graph G      改用其他 graph
  -n, --dry-run      只顯示計畫與模型分配
  -w, --workers N    (接受但忽略:ruai 依序執行節點)

常用範例:
  ruai agent init                                 在目前目錄建立 .agge/
  ruai agent chat                                 互動對話 (/help 看指令)
  ruai agent chat --team code                     與 lead 對話,lead 會分派工作給團隊
  ruai agent ask "解釋 main.py 在做什麼"           單次提問
  ruai agent code "寫一個 todo CLI" -o todo        設計→實作→測試→審查 流程
  ruai agent book "EDA HLS 的歷史與現況" -o hlsBook --chapters 8
  ruai agent run brainstorm "如何設計快取層" -M a=glm -M merge=claude
  ruai agent graph show code / ruai agent graph mermaid code
  ruai agent model add claude anthropic claude-sonnet-5-5
  ruai agent model add glm nvidia z-ai/glm-5.3-flash --temperature 0.5
  ruai agent log                                  檢視執行紀錄"""


# ---------------------------------------------------------------- arguments
class Args:
    """Parsed options: values by name, repeated options, flags, positionals."""

    def __init__(self) -> None:
        self.pos: list[str] = []
        self.opts: dict[str, str] = {}
        self.multi: dict[str, list[str]] = {}
        self.flags: list[str] = []

    def opt(self, k: str) -> str:
        return self.opts[k] if k in self.opts else ""

    def has(self, k: str) -> bool:
        return k in self.opts

    def flag(self, k: str) -> bool:
        return k in self.flags

    def many(self, k: str) -> list[str]:
        return self.multi[k] if k in self.multi else []

    def num(self, k: str, default: int) -> int:
        if k not in self.opts:
            return default
        try:
            return int(self.opts[k])
        except ValueError:
            raise ProjectError(f"argument {k}: invalid int value: '{self.opts[k]}'")

    def at(self, i: int) -> str:
        return self.pos[i] if i < len(self.pos) else ""


def runtime_spec(out: bool) -> dict[str, str]:
    s = {"-m": "=model", "--model": "=model", "-M": "+assign", "--assign": "+assign", "-y": "!yes",
         "--yes": "!yes", "--add-dir": "+add_dir", "--lang": "=lang", "--max-steps": "=max_steps",
         "-v": "!verbose", "--verbose": "!verbose"}
    if out:
        s["-o"] = "=out"
        s["--out"] = "=out"
    return s


def graph_spec() -> dict[str, str]:
    s = runtime_spec(True)
    for k, v in [("-g", "=graph"), ("--graph", "=graph"), ("-w", "=workers"), ("--workers", "=workers"),
                 ("-n", "!dry_run"), ("--dry-run", "!dry_run")]:
        s[k] = v
    return s


def command_spec(cmd: str) -> dict[str, str]:
    """How each option of a command is read: =value, +repeated, !flag, ?optional value, *values."""
    if cmd == "init":
        return {"-m": "=model", "--model": "=model"}
    if cmd == "chat":
        s = runtime_spec(True)
        for k, v in [("-a", "=agent"), ("--agent", "=agent"), ("-t", "=team"), ("--team", "=team"),
                     ("-r", "?resume"), ("--resume", "?resume")]:
            s[k] = v
        return s
    if cmd == "ask":
        s = runtime_spec(True)
        for k, v in [("-a", "=agent"), ("--agent", "=agent"), ("-t", "=team"), ("--team", "=team"),
                     ("--stream", "!stream")]:
            s[k] = v
        return s
    if cmd == "run":
        return graph_spec()
    if cmd == "code":
        s = graph_spec()
        s["--stack"] = "=stack"
        return s
    if cmd == "book":
        s = graph_spec()
        for k, v in [("-c", "=chapters"), ("--chapters", "=chapters"), ("--audience", "=audience"),
                     ("--style", "=style")]:
            s[k] = v
        return s
    if cmd == "graph":
        return {"--from": "=source", "--force": "!force", "--json": "!json", "--agent": "=agent",
                "--prompt": "=prompt", "--map": "=map", "--node-model": "=node_model", "--after": "+after",
                "--when": "=when", "--max": "=max"}
    if cmd == "agent":
        return {"--base": "=base", "--description": "=description", "--system": "=system",
                "--system-file": "=system_file", "--model": "=model", "--tools": "=tools"}
    if cmd == "team":
        return {"--lead": "=lead", "--members": "*members", "--description": "=description"}
    if cmd == "model":
        return {"--temperature": "=temperature", "--top-p": "=top_p", "--max-tokens": "=max_tokens",
                "--base-url": "=base_url", "--api-key-env": "=api_key_env", "--prompt": "=prompt"}
    if cmd == "log":
        return {"-n": "=n"}
    return {}


def parse_args(argv: list[str], spec: dict[str, str]) -> Args:
    a = Args()
    i = 0
    only_pos = False
    while i < len(argv):
        x = argv[i]
        i += 1
        if only_pos or x == "-" or not x.startswith("-"):
            a.pos.append(x)
            continue
        if x == "--":
            only_pos = True
            continue
        name, eq, inline = x.partition("=")
        if name not in spec:
            raise ProjectError(f"unrecognized arguments: {x}")
        kind = spec[name][0]
        dest = spec[name][1:]
        if kind == "!":
            a.flags.append(dest)
            continue
        if kind == "?":
            # an optional value: the next word unless it is an option
            if eq != "":
                a.opts[dest] = inline
            elif i < len(argv) and not argv[i].startswith("-"):
                a.opts[dest] = argv[i]
                i += 1
            else:
                a.opts[dest] = ""
            continue
        if kind == "*":
            vals: list[str] = []
            while i < len(argv) and not argv[i].startswith("-"):
                vals.append(argv[i])
                i += 1
            if len(vals) == 0:
                raise ProjectError(f"argument {name}: expected at least one argument")
            a.multi[dest] = vals
            continue
        value = inline
        if eq == "":
            if i >= len(argv):
                raise ProjectError(f"argument {name}: expected one argument")
            value = argv[i]
            i += 1
        if kind == "+":
            if dest not in a.multi:
                a.multi[dest] = []
            a.multi[dest].append(value)
        else:
            a.opts[dest] = value
    return a


# ---------------------------------------------------------------- helpers
def need_project() -> Project:
    prj = Project.find("")
    if prj is None:
        raise ProjectError("目前目錄不是 agge 專案,請先執行 `ruai agent init`")
    return prj


def parse_kv(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for it in items:
        if "=" not in it:
            raise ProjectError(f"格式錯誤 '{it}',應為 name=model")
        k, _, v = it.partition("=")
        out[k.strip()] = v.strip()
    return out


def make_runtime(prj: Project, a: Args) -> Runtime:
    cfg = prj.config()
    root = prj.root
    if a.opt("out") != "":
        root = os.path.realpath(os.path.expanduser(a.opt("out")))
    os.makedirs(root, exist_ok=True)
    dirs = cfg.get("dirs").strs() + [prj.root] + a.many("add_dir")
    approval = "auto" if a.flag("yes") else cfg.get_str("approval", "ask")
    rt = Runtime(prj, Workspace(root, dirs), Approver(approval), a.opt("model"), parse_kv(a.many("assign")),
                 a.opt("team"), a.num("max_steps", 0), a.opt("lang"))
    rt.verbose = a.flag("verbose")
    return rt


def read_task(words: list[str]) -> str:
    task = " ".join(words).strip()
    if task == "-" or (task == "" and not sys.stdin.isatty()):
        task = sys.stdin.read().strip()
    if task == "":
        raise ProjectError("請提供任務描述(或用 - 從 stdin 讀取)")
    if task.startswith("@") and os.path.isfile(task[1:]):
        task = load_text(task[1:])
    return task


def py_repr(v: Json) -> str:
    """Python's repr of a JSON value."""
    k = v.kind()
    if k == "null":
        return "None"
    if k == "bool":
        return "True" if v.bool_or(False) else "False"
    if k == "int":
        return str(v.int_or(0))
    if k == "float":
        return repr(v.float_or(0.0))
    if k == "str":
        return repr(v.str_or(""))
    if k == "list":
        return "[" + ", ".join([py_repr(x) for x in v.items()]) + "]"
    return "{" + ", ".join([f"{repr(key)}: {py_repr(v.get(key))}" for key in v.keys()]) + "}"


# ---------------------------------------------------------------- commands
def cmd_init(a: Args) -> None:
    prj = Project.init(a.at(0) if a.at(0) != "" else os.getcwd(), a.opt("model"))
    ui.out(f"已初始化 agge 專案: {prj.dir}")
    ui.out(f"預設模型: {prj.config().get_str('default_model', '')}  (用 `ruai agent model default <name>` 修改)")


def cmd_chat(a: Args) -> None:
    found = Project.find("")
    prj = found if found is not None else Project.init(os.getcwd(), "")
    rt = make_runtime(prj, a)
    resuming = a.has("resume")
    resume = a.opt("resume")
    if a.opt("agent") != "" or a.opt("team") != "":
        session = prj.load_session(resume) if resuming else prj.new_session()
        if a.opt("team") != "":
            teams = prj.teams()
            if not teams.has(a.opt("team")):
                raise ProjectError(f"沒有團隊 '{a.opt('team')}'(可用: {', '.join(teams.keys())})")
            session.set("team", Json.of_str(a.opt("team")))
            session.set("speaker", Json.of_str(teams.get(a.opt("team")).get_str("lead", "lead")))
        if a.opt("agent") != "":
            rt.agent(a.opt("agent"), "", 0)
            session.set("speaker", Json.of_str(a.opt("agent")))
        prj.save_session(session)
        resume = session.get_str("id", "")
        resuming = True
    start_chat(prj, rt, resume, resuming)


def cmd_ask(a: Args) -> None:
    prj = need_project()
    rt = make_runtime(prj, a)
    rt.printing = True
    task = read_task(a.pos)
    name = a.opt("agent")
    if name == "":
        name = "lead" if a.opt("team") != "" else "assistant"
    agent = rt.agent(name, "", 0)
    stream = sys.stdout.isatty() or a.flag("stream")
    reply = rt.run(agent, [Msg("user", task)], stream, 0)
    ui.out("" if stream else reply)


def run_graph(prj: Project, a: Args, graph_name: str, graph: Json, task: str) -> None:
    errs = validate(graph, prj.agents())
    if len(errs) > 0:
        raise ProjectError("graph 有錯誤:\n  " + "\n  ".join(errs))
    log = prj.new_run(graph_name, task)
    rt = make_runtime(prj, a)
    rt.printing = True
    rt.logging = True
    rt.log_path = log.path
    ui.out(ui.bold(f"▶ 執行 graph '{graph_name}'") + ui.dim(f"  工作目錄 {rt.workspace.root} · 紀錄 {log.id}"))
    if a.flag("dry_run"):
        ui.out(describe(graph))
        nodes = graph.get("nodes")
        for n in nodes.keys():
            agent = nodes.get(n).get_str("agent", n)
            m = nodes.get(n).get_str("model", "")
            if agent in rt.model_map:
                m = rt.model_map[agent]
            if n in rt.model_map:
                m = rt.model_map[n]
            if m == "":
                m = rt.model_for(agent)
            ui.out(f"  {n:<12} → {m}")
        log.finish("dry-run", "")
        return
    outputs: dict[str, str] = {}
    try:
        outputs = rt.run_graph(graph, task, log)
    except Exception as e:
        log.finish("failed", str(e))
        raise
    log.finish("done", "")
    last = list(outputs.values())[-1]
    ui.out(ui.bold("\n═══ 最終輸出 ═══\n") + last)
    ui.out(ui.dim(f"\n完整紀錄: ruai agent log {log.id}"))


def cmd_run(a: Args) -> None:
    prj = need_project()
    if a.at(0) == "":
        raise ProjectError("用法: ruai agent run <graph> [任務...]")
    graph = prj.load_graph(a.at(0))
    name = os.path.splitext(os.path.basename(a.at(0)))[0]
    run_graph(prj, a, name, graph, read_task(a.pos[1:]))


def cmd_code(a: Args) -> None:
    prj = need_project()
    task = read_task(a.pos)
    if a.opt("stack") != "":
        task += f"\n\nProgramming language / stack: {a.opt('stack')}"
    run_graph(prj, a, "code", prj.load_graph(a.opt("graph") if a.opt("graph") != "" else "code"), task)


def cmd_book(a: Args) -> None:
    prj = need_project()
    task = read_task(a.pos)
    extra: list[str] = []
    if a.opt("chapters") != "":
        extra.append(f"The book should have about {a.num('chapters', 0)} chapters.")
    if a.opt("lang") != "":
        extra.append(f"Write the whole book in {a.opt('lang')}.")
    if a.opt("audience") != "":
        extra.append(f"Target audience: {a.opt('audience')}.")
    if a.opt("style") != "":
        extra.append(f"Style / format requirements: {a.opt('style')}")
    if len(extra) > 0:
        task += "\n\n" + "\n".join(extra)
    run_graph(prj, a, "book", prj.load_graph(a.opt("graph") if a.opt("graph") != "" else "book"), task)


def edge_has(e: Json, node: str) -> bool:
    if e.kind() == "list":
        return e.at(0).text() == node or e.at(1).text() == node
    return e.get_str("from", "") == node or e.get_str("to", "") == node


def cmd_graph(a: Args) -> None:
    prj = need_project()
    op = a.at(0)
    ops = ["list", "show", "mermaid", "validate", "new", "rm", "add-node", "add-edge", "rm-node"]
    if op not in ops:
        raise ProjectError(f"argument op: invalid choice: '{op}' (choose from {', '.join(ops)})")
    if op == "list":
        graphs = prj.graphs()
        for n in graphs.keys():
            g = graphs.get(n)
            builtin = n in ["solo", "code", "book", "brainstorm"] and not prj.is_project_graph(n)
            tag = ui.dim("(內建) ") if builtin else ""
            ui.out(f"  {ui.cyan(n):<20} {g.get('nodes').size()} nodes  {tag}{g.get_str('description', '')}")
        return
    name = a.at(1)
    rest = a.pos[2:]
    if name == "":
        raise ProjectError(f"用法: ruai agent graph {op} <name>")
    if op == "new":
        if prj.graphs().has(name) and not a.flag("force"):
            raise ProjectError(f"graph '{name}' 已存在(--force 覆寫)")
        g = Json.loads('{"description": "", "nodes": {}, "edges": []}')
        if a.opt("source") != "":
            g = prj.load_graph(a.opt("source"))
        prj.save_graph(name, g)
        ui.out(f"已建立 {prj.path('graphs', name + '.json')}")
        return
    if op == "rm":
        prj.delete_graph(name)
        ui.out(f"已刪除 {name}")
        return
    g = prj.load_graph(name)
    if op == "show":
        ui.out(g.dumps(indent=2, ensure_ascii=False) if a.flag("json") else describe(g))
    elif op == "mermaid":
        ui.out(mermaid(g))
    elif op == "validate":
        errs = validate(g, prj.agents())
        ui.out("\n".join([ui.red("✘ " + e) for e in errs]) if len(errs) > 0 else ui.green("✔ graph 正確"))
        if len(errs) > 0:
            sys.exit(1)
    else:
        nodes = g.get("nodes")
        edge_list = g.get("edges")
        if op == "add-node":
            if len(rest) == 0:
                raise ProjectError("用法: ruai agent graph add-node <graph> <node> [--agent A] [--prompt P] [--map SRC] "
                                   "[--node-model M] [--after X ...]")
            node = Json.new_dict()
            node.set("agent", Json.of_str(a.opt("agent") if a.opt("agent") != "" else rest[0]))
            node.set("prompt", Json.of_str(a.opt("prompt") if a.opt("prompt") != "" else "{{task}}\n\n{{deps}}\n\n{{feedback}}"))
            if a.opt("map") != "":
                node.set("map", Json.of_str(a.opt("map")))
            if a.opt("node_model") != "":
                node.set("model", Json.of_str(a.opt("node_model")))
            nodes.set(rest[0], node)
            for x in a.many("after"):
                edge_list.append(Json.of_strs([x, rest[0]]))
        elif op == "rm-node":
            if len(rest) == 0:
                raise ProjectError("用法: ruai agent graph rm-node <graph> <node>")
            nodes.remove(rest[0])
            kept = Json.new_list()
            for e in edge_list.items():
                if not edge_has(e, rest[0]):
                    kept.append(e)
            edge_list = kept
        else:
            if len(rest) != 2:
                raise ProjectError("用法: ruai agent graph add-edge <graph> <from> <to> [--when REGEX --max N]")
            if a.opt("when") != "":
                e = Json.new_dict()
                e.set("from", Json.of_str(rest[0]))
                e.set("to", Json.of_str(rest[1]))
                e.set("when", Json.of_str(a.opt("when")))
                e.set("max", Json.of_int(a.num("max", 2)))
                edge_list.append(e)
            else:
                edge_list.append(Json.of_strs(rest))
        g.set("nodes", nodes)
        g.set("edges", edge_list)
        errs = validate(g, prj.agents())
        prj.save_graph(name, g)
        ui.out(describe(g) if g.get("nodes").truthy() else "(空 graph)")
        for e2 in errs:
            ui.out(ui.yellow("! " + e2))


def cmd_agent(a: Args) -> None:
    prj = need_project()
    op = a.at(0)
    name = a.at(1)
    agents = prj.agents()
    if op == "list":
        rt = make_runtime(prj, Args())
        for n in agents.keys():
            ui.out(f"  {ui.cyan(n):<22} {ui.dim(rt.model_for(n)):<30} {agents.get(n).get_str('description', '')}")
    elif op == "show":
        if not agents.has(name):
            raise ProjectError(f"沒有名為 '{name}' 的 agent")
        ui.out(agents.get(name).dumps(indent=2, ensure_ascii=False))
    elif op == "add":
        if name == "":
            raise ProjectError("用法: ruai agent agent add <name> [--base A] ...")
        base_name = a.opt("base") if a.opt("base") != "" else name
        base = agents.get(base_name) if agents.has(base_name) else agents.get("assistant")
        for k in ["description", "system", "model"]:
            if a.opt(k) != "":
                base.set(k, Json.of_str(a.opt(k)))
        if a.opt("system_file") != "":
            base.set("system", Json.of_str(load_text(a.opt("system_file"))))
        if a.has("tools"):
            base.set("tools", Json.of_strs([t for t in a.opt("tools").split(",") if t != ""]))
        prj.save_agent(name, base)
        ui.out(f"已儲存 agent '{name}'")
    elif op == "rm":
        prj.save_agent(name, Json.null())
        ui.out(f"已移除專案 agent '{name}'(內建同名 agent 仍存在)")
    else:
        raise ProjectError(f"argument op: invalid choice: '{op}' (choose from list, show, add, rm)")


def cmd_team(a: Args) -> None:
    prj = need_project()
    op = a.at(0)
    name = a.at(1)
    if op == "list":
        teams = prj.teams()
        for n in teams.keys():
            t = teams.get(n)
            ui.out(f"  {ui.cyan(n):<16} lead={t.get_str('lead', '')}  members={', '.join(t.get('members').strs())}  "
                   f"{ui.dim(t.get_str('description', ''))}")
    elif op == "add":
        cfg = prj.raw_config()
        teams = cfg.get("teams")
        t = Json.new_dict()
        t.set("lead", Json.of_str(a.opt("lead") if a.opt("lead") != "" else "lead"))
        t.set("members", Json.of_strs(a.many("members")))
        t.set("description", Json.of_str(a.opt("description")))
        teams.set(name, t)
        cfg.set("teams", teams)
        prj.save_config(cfg)
        ui.out(f"已建立團隊 '{name}'")
    elif op == "rm":
        cfg = prj.raw_config()
        teams = cfg.get("teams")
        if teams.kind() == "dict":
            teams.remove(name)
            cfg.set("teams", teams)
        prj.save_config(cfg)
        ui.out(f"已移除團隊 '{name}'")
    else:
        raise ProjectError(f"argument op: invalid choice: '{op}' (choose from list, add, rm)")


def cmd_model(a: Args) -> None:
    prj = need_project()
    op = a.at(0)
    name = a.at(1)
    cfg = prj.raw_config()
    models = cfg.get("models") if cfg.get("models").kind() == "dict" else Json.new_dict()
    if op == "list":
        m_cfg = prj.config()
        all_models = m_cfg.get("models")
        for n in all_models.keys():
            m = all_models.get(n)
            mark = "*" if n == m_cfg.get_str("default_model", "") else " "
            p = providers().get(m.get_str("provider", ""))
            env = m.get_str("api_key_env", p.get_str("env", ""))
            key = ""
            if env != "":
                key = ui.green("key ✔") if os.environ.get(env) is not None else ui.red(f"缺 {env}")
            ui.out(f"{mark} {ui.cyan(n):<18} {m.get_str('provider', '')}:{m.get_str('model', ''):<36} {key}")
    elif op == "add":
        provider = a.at(2)
        model_id = a.at(3)
        if provider == "" or model_id == "":
            raise ProjectError("用法: ruai agent model add <alias> <provider> <model-id> [--base-url ...]")
        m = Json.new_dict()
        m.set("provider", Json.of_str(provider))
        m.set("model", Json.of_str(model_id))
        for k in ["temperature", "top_p"]:
            if a.has(k):
                try:
                    m.set(k, Json.of_float(float(a.opt(k))))
                except ValueError:
                    raise ProjectError(f"argument --{k.replace('_', '-')}: invalid float value: '{a.opt(k)}'")
        if a.has("max_tokens"):
            m.set("max_tokens", Json.of_int(a.num("max_tokens", 0)))
        for k in ["base_url", "api_key_env"]:
            if a.has(k):
                m.set(k, Json.of_str(a.opt(k)))
        if not providers().has(provider) and not m.has("base_url"):
            raise ProjectError(f"未知 provider '{provider}',請加 --base-url(OpenAI 相容端點)")
        models.set(name, m)
        cfg.set("models", models)
        prj.save_config(cfg)
        ui.out(f"已新增模型 {name} = {provider}:{model_id}")
    elif op == "rm":
        models.remove(name)
        cfg.set("models", models)
        prj.save_config(cfg)
        ui.out(f"已移除 {name}")
    elif op == "default":
        prj.resolve_model(name)
        cfg.set("default_model", Json.of_str(name))
        prj.save_config(cfg)
        ui.out(f"預設模型: {name}")
    elif op == "test":
        n = name if name != "" else prj.config().get_str("default_model", "echo")
        llm = LLM(prj.resolve_model(n))
        prompt = a.opt("prompt") if a.has("prompt") else "用一句話介紹你自己。"
        ui.out(ui.dim(f"測試 {n} ({llm.label}) ..."))
        llm.chat("You are a helpful assistant.", [Msg("user", prompt)], True, 2)
        ui.out()
    else:
        raise ProjectError(f"argument op: invalid choice: '{op}' (choose from list, add, rm, default, test)")


def cmd_provider(a: Args) -> None:
    ps = providers()
    for n in ps.keys():
        p = ps.get(n)
        env = p.get_str("env", "")
        status = ""
        if env != "":
            status = ui.green("✔") if os.environ.get(env) is not None else ui.dim("✘")
        ui.out(f"  {ui.cyan(n):<20} {p.get_str('kind', ''):<10} {p.get_str('base_url', ''):<55} {env} {status}")


def cmd_dir(a: Args) -> None:
    prj = need_project()
    op = a.at(0)
    cfg = prj.raw_config()
    dirs = cfg.get("dirs").strs()
    if op == "add" or op == "rm":
        if a.at(1) == "":
            raise ProjectError(f"用法: ruai agent dir {op} <path>")
        p = os.path.realpath(os.path.expanduser(a.at(1)))
        if op == "add":
            if p not in dirs:
                dirs.append(p)
            ui.out(f"已允許 agent 存取 {p}")
        else:
            dirs = [d for d in dirs if d != p]
            ui.out(f"已移除 {p}")
        cfg.set("dirs", Json.of_strs(dirs))
        prj.save_config(cfg)
    else:
        ui.out(f"  {prj.root} (專案)")
        for d in prj.config().get("dirs").strs():
            ui.out(f"  {d}")


def cmd_config(a: Args) -> None:
    prj = need_project()
    cfg = prj.raw_config()
    key = a.at(0)
    if key == "":
        ui.out(prj.config().dumps(indent=2, ensure_ascii=False))
    elif len(a.pos) < 2:
        ui.out(prj.config().get(key).dumps(indent=2, ensure_ascii=False))
    else:
        value = a.at(1)
        v = Json.of_str(value)
        try:
            v = Json.loads(value)
        except ValueError:
            v = Json.of_str(value)
        cfg.set(key, v)
        prj.save_config(cfg)
        ui.out(f"{key} = {py_repr(v)}")


def cmd_log(a: Args) -> None:
    prj = need_project()
    runs = prj.runs()
    run = a.at(0)
    if run == "":
        n = a.num("n", 20)
        for rid in runs[-n:]:
            meta = Json.loads(load_text(prj.path("runs", rid, "meta.json")))
            status = meta.get_str("status", "?")
            color = ui.green(status) if status == "done" else (ui.red(status) if status == "failed" else ui.yellow(status))
            ui.out(f"  {ui.cyan(rid):<40} {color:<10} {ui.short(meta.get_str('task', ''), 60)}")
        return
    rid = ""
    if run in runs:
        rid = run
    else:
        for r in reversed(runs):
            if r.startswith(run):
                rid = r
                break
    if run == "last" and len(runs) > 0:
        rid = runs[-1]
    if rid == "":
        raise ProjectError(f"找不到執行紀錄 {run}")
    d = prj.path("runs", rid)
    node = a.at(1)
    if node != "":
        files = [f for f in sorted_files(d, ".md") if os.path.basename(f).startswith(node)]
        if len(files) == 0:
            raise ProjectError(f"{rid} 沒有節點 {node} 的輸出")
        for f in files:
            ui.out(ui.bold(f"── {os.path.basename(f)} ──"))
            ui.out(load_text(f))
        return
    ui.out(Json.loads(load_text(os.path.join(d, "meta.json"))).dumps(indent=2, ensure_ascii=False))
    ui.out(ui.bold("輸出:"))
    for f in sorted_files(d, ".md"):
        ui.out(f"  {os.path.splitext(os.path.basename(f))[0]:<24} {os.path.getsize(f)} bytes")
    ui.out(ui.dim(f"用 ruai agent log {rid} <node> 查看節點輸出"))


def cmd_session(a: Args) -> None:
    prj = need_project()
    if a.at(0) == "list":
        for sid in prj.sessions():
            ui.out("  " + sid)
    elif a.at(0) == "show":
        s = prj.load_session(a.at(1))
        histories = s.get("histories")
        for agent in histories.keys():
            ui.out(ui.bold(f"── {agent} ──"))
            for m in histories.get(agent).items():
                who = ui.cyan("you") if m.get_str("role", "") == "user" else ui.green(agent)
                ui.out(f"{who}: {m.get_str('content', '')}\n")
    else:
        raise ProjectError(f"argument op: invalid choice: '{a.at(0)}' (choose from list, show)")


def cmd_status(a: Args) -> None:
    ui.out(f"ruai agent {VERSION}")
    prj = Project.find("")
    if prj is None:
        ui.out("目前不在 agge 專案中(ruai agent init 建立)")
        return
    cfg = prj.config()
    ui.out(f"專案:     {prj.root}")
    ui.out(f"預設模型: {cfg.get_str('default_model', '')}")
    ui.out(f"模型:     {', '.join(cfg.get('models').keys())}")
    ui.out(f"Agents:   {', '.join(prj.agents().keys())}")
    ui.out(f"Teams:    {', '.join(prj.teams().keys())}")
    ui.out(f"Graphs:   {', '.join(prj.graphs().keys())}")
    extra = ", ".join(cfg.get("dirs").strs())
    ui.out(f"核准模式: {cfg.get_str('approval', 'ask')}   額外目錄: {extra if extra != '' else '(無)'}")
    ui.out(f"Sessions: {len(prj.sessions())}   Runs: {len(prj.runs())}")


def dispatch(cmd: str, a: Args) -> None:
    if cmd == "init":
        cmd_init(a)
    elif cmd == "chat":
        cmd_chat(a)
    elif cmd == "ask":
        cmd_ask(a)
    elif cmd == "run":
        cmd_run(a)
    elif cmd == "code":
        cmd_code(a)
    elif cmd == "book":
        cmd_book(a)
    elif cmd == "graph":
        cmd_graph(a)
    elif cmd == "agent":
        cmd_agent(a)
    elif cmd == "team":
        cmd_team(a)
    elif cmd == "model":
        cmd_model(a)
    elif cmd == "provider":
        cmd_provider(a)
    elif cmd == "dir":
        cmd_dir(a)
    elif cmd == "config":
        cmd_config(a)
    elif cmd == "log":
        cmd_log(a)
    elif cmd == "session":
        cmd_session(a)
    elif cmd == "status":
        cmd_status(a)


COMMANDS = "init chat ask run code book graph agent team model provider dir config log session status"


def main(argv: list[str]) -> None:
    if len(argv) == 0:
        argv = ["chat"]
    cmd = argv[0]
    if cmd in ["help", "-h", "--help"]:
        ui.out(HELP)
        return
    if cmd in ["-V", "--version"]:
        ui.out(f"ruai agent {VERSION}")
        return
    if cmd not in COMMANDS.split():
        ui.err(f"未知的指令 '{cmd}'(見 ruai agent help)")
        sys.exit(2)
    try:
        dispatch(cmd, parse_args(argv[1:], command_spec(cmd)))
    except ProjectError as e:
        ui.err(str(e))
        sys.exit(1)
    except LLMError as e:
        ui.err(str(e))
        sys.exit(1)
    except ToolError as e:
        ui.err(str(e))
        sys.exit(1)
    except GraphError as e:
        ui.err(str(e))
        sys.exit(1)
    except ValueError as e:
        ui.err(str(e))
        sys.exit(1)
