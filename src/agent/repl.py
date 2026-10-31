"""Interactive chat (like codex / opencode): talk to one agent, a whole team, or launch graphs."""
import os
import shlex
import sys

from ruai.json import Json

import ui
from graph import describe, validate
from project import Project, ProjectError, save_text
from providers import LLMError, Msg, msgs_from, msgs_json
from runtime import Runtime
from tools import ToolError

CHAT_HELP = """指令:
  /help                    顯示說明
  /agent [name]            查看/切換對話的 agent      /agents   列出所有 agent
  @name 訊息               單次對某個 agent 說話
  /team [name|off]         讓 lead 帶領團隊(可 delegate 給成員)   /teams   列出團隊
  /model [name]            查看/設定目前 agent 的模型  /models   列出模型
  /run <graph> <任務>      執行 graph 工作流程        /graphs   列出 graph   /graph <name> 顯示結構
  /add-dir <path>          允許 agent 存取其他目錄     /dirs     列出可存取目錄
  /yes                     切換 shell 指令自動核准
  /compact                 把目前對話壓縮成摘要       /clear    清除目前 agent 的對話
  /new  /sessions  /resume [id]   session 管理
  /save <file>             把最後一則回覆存成檔案
  /exit                    離開(Ctrl-D 亦可)
多行輸入: 以 \"\"\" 開始與結束,或行尾加 \\"""


class Chat:
    def __init__(self, project: Project, rt: Runtime, session: Json) -> None:
        self.project = project
        self.rt = rt
        self.s = session
        self.rt.team = session.get("team").str_or("")
        for d in session.get("dirs").strs():
            self.rt.workspace.add(d)
        models = session.get("models")
        for k in models.keys():
            self.rt.model_map[k] = models.get(k).str_or("")

    # ---------- helpers ----------
    def speaker(self) -> str:
        sp = self.s.get_str("speaker", "")
        return sp if sp != "" else "assistant"

    def save(self) -> None:
        models = Json.new_dict()
        for k, v in self.rt.model_map.items():
            models.set(k, Json.of_str(v))
        self.s.set("models", models)
        dirs = Json.of_strs(self.rt.workspace.dirs[1:])
        self.s.set("dirs", dirs)
        self.project.save_session(self.s)

    def prompt(self) -> str:
        model = self.rt.model_for(self.speaker())
        team = self.s.get("team").str_or("")
        tag = f"[{team}] " if team != "" else ""
        return f"{tag}{self.speaker()}@{model}> "

    def read_input(self) -> str:
        line = input(ui.bold(self.prompt()))
        if line.strip() == '"""':
            buf: list[str] = []
            while True:
                nxt = input("... ")
                if nxt.strip() == '"""':
                    return "\n".join(buf)
                buf.append(nxt)
        buf2 = [line]
        while buf2[-1].endswith("\\"):
            buf2[-1] = buf2[-1][:-1]
            buf2.append(input("... "))
        return "\n".join(buf2)

    # ---------- main loop ----------
    def loop(self) -> None:
        ui.out(ui.bold("ruai agent chat") + ui.dim(f"  session {self.s.get_str('id', '')} · {self.project.root}"))
        ui.out(ui.dim("輸入 /help 查看指令,@agent 指定對象,/team code 啟用團隊模式"))
        while True:
            try:
                text = self.read_input().strip()
            except EOFError:
                ui.out()
                break
            if self.handle(text) == "exit":
                break
        self.save()

    def handle(self, text: str) -> str:
        """Process one input: a /command, an @agent message, or a message to the current speaker."""
        text = text.strip()
        if text == "":
            return ""
        if text.startswith("/"):
            return self.command(text)
        target = self.speaker()
        if text.startswith("@"):
            name, _, rest = text[1:].partition(" ")
            if not self.rt.agents().has(name):
                ui.err(f"沒有 agent '{name}'")
                return ""
            target = name
            text = rest.strip()
        self.send(target, text)
        return ""

    def send(self, target: str, text: str) -> None:
        hist = msgs_from(self.s.get("histories").get(target))
        hist.append(Msg("user", text))
        ui.out(ui.cyan(f"{target}") + ui.dim(f" ({self.rt.model_for(target)})"))
        try:
            a = self.rt.agent(target, "", 0)
            self.rt.run(a, hist, True, 1)
        except LLMError as e:
            ui.err(str(e))
            hist.pop()
        except ProjectError as e:
            ui.err(str(e))
            hist.pop()
        histories = self.s.get("histories")
        histories.set(target, msgs_json(hist))
        self.s.set("histories", histories)
        ui.out("\n")
        self.save()

    # ---------- slash commands ----------
    def command(self, text: str) -> str:
        parts: list[str] = []
        try:
            parts = shlex.split(text)
        except ValueError:
            parts = text.split()
        cmd = parts[0]
        args = parts[1:]
        rest = text[len(cmd):].strip()
        try:
            return self.run_command(cmd, args, rest)
        except ProjectError as e:
            ui.err(str(e))
        except LLMError as e:
            ui.err(str(e))
        except ToolError as e:
            ui.err(str(e))
        except ValueError as e:
            ui.err(str(e))
        return ""

    def run_command(self, cmd: str, args: list[str], rest: str) -> str:
        if cmd in ["/exit", "/quit"]:
            return "exit"
        if cmd == "/help":
            ui.out(CHAT_HELP)
        elif cmd == "/agents":
            agents = self.rt.agents()
            for n in agents.keys():
                mark = "*" if n == self.speaker() else " "
                ui.out(f"{mark} {ui.cyan(n):<22} {ui.dim(self.rt.model_for(n))}  {agents.get(n).get_str('description', '')}")
        elif cmd == "/agent":
            if len(args) > 0:
                self.rt.agent(args[0], "", 0)
                self.s.set("speaker", Json.of_str(args[0]))
            ui.out(f"目前 agent: {self.speaker()}")
        elif cmd == "/teams":
            teams = self.project.teams()
            for n in teams.keys():
                t = teams.get(n)
                ui.out(f"  {ui.cyan(n):<16} lead={t.get_str('lead', '')} members={', '.join(t.get('members').strs())}  "
                       f"{ui.dim(t.get_str('description', ''))}")
        elif cmd == "/team":
            if len(args) > 0 and args[0] == "off":
                self.s.set("team", Json.null())
                self.rt.team = ""
                self.s.set("speaker", Json.of_str("assistant"))
            elif len(args) > 0:
                teams = self.project.teams()
                if not teams.has(args[0]):
                    raise ValueError(f"沒有團隊 '{args[0]}'")
                self.s.set("team", Json.of_str(args[0]))
                self.rt.team = args[0]
                self.s.set("speaker", Json.of_str(teams.get(args[0]).get_str("lead", "lead")))
            team = self.s.get("team").str_or("")
            ui.out(f"團隊: {team if team != '' else '(無)'},對話對象: {self.speaker()}")
        elif cmd == "/models":
            cfg = self.project.config()
            models = cfg.get("models")
            for n in models.keys():
                m = models.get(n)
                mark = "*" if n == cfg.get_str("default_model", "") else " "
                ui.out(f"{mark} {ui.cyan(n):<18} {m.get_str('provider', '')}:{m.get_str('model', '')}")
        elif cmd == "/model":
            if len(args) > 0:
                self.rt.llm(args[0])
                self.rt.model_map[self.speaker()] = args[0]
            ui.out(f"{self.speaker()} 使用模型: {self.rt.model_for(self.speaker())}")
        elif cmd == "/graphs":
            graphs = self.project.graphs()
            for n in graphs.keys():
                ui.out(f"  {ui.cyan(n):<16} {graphs.get(n).get_str('description', '')}")
        elif cmd == "/graph":
            if len(args) == 0:
                raise ValueError("用法: /graph <name>")
            ui.out(describe(self.project.load_graph(args[0])))
        elif cmd == "/run":
            if len(args) < 2:
                raise ValueError("用法: /run <graph> <任務>")
            self.run_graph(args[0], rest[len(args[0]):].strip())
        elif cmd == "/add-dir":
            ui.out(f"已允許: {self.rt.workspace.add(rest)}")
        elif cmd == "/dirs":
            ui.out("\n".join([f"  {d}" for d in self.rt.workspace.dirs]))
        elif cmd == "/yes":
            self.rt.approver.mode = "ask" if self.rt.approver.mode == "auto" else "auto"
            ui.out(f"shell 核准模式: {self.rt.approver.mode}")
        elif cmd == "/clear":
            histories = self.s.get("histories")
            histories.set(self.speaker(), Json.new_list())
            self.s.set("histories", histories)
            ui.out(f"已清除 {self.speaker()} 的對話")
        elif cmd == "/compact":
            self.compact()
        elif cmd == "/new":
            self.save()
            old = self.s
            self.s = self.project.new_session()
            self.s.set("speaker", old.get("speaker"))
            self.s.set("team", old.get("team"))
            ui.out(f"新 session {self.s.get_str('id', '')}")
        elif cmd == "/sessions":
            sid = self.s.get_str("id", "")
            for x in self.project.sessions()[-20:]:
                ui.out(("* " if x == sid else "  ") + x)
        elif cmd == "/resume":
            self.save()
            self.s = self.project.load_session(args[0] if len(args) > 0 else "")
            self.rt.team = self.s.get("team").str_or("")
            ui.out(f"已載入 session {self.s.get_str('id', '')}")
        elif cmd == "/save":
            hist = msgs_from(self.s.get("histories").get(self.speaker()))
            last = ""
            for m in reversed(hist):
                if m.role == "assistant":
                    last = m.content
                    break
            if last == "" or rest == "":
                raise ValueError("用法: /save <file>(且需要有回覆)")
            p = self.rt.workspace.resolve(rest)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            save_text(p, last)
            ui.out(f"已寫入 {p}")
        else:
            ui.err(f"未知指令 {cmd},輸入 /help")
        return ""

    def run_graph(self, name: str, task: str) -> None:
        graph = self.project.load_graph(name)
        errs = validate(graph, self.rt.agents())
        if len(errs) > 0:
            raise ValueError("graph 有錯誤:\n  " + "\n  ".join(errs))
        log = self.project.new_run(name, task)
        self.rt.printing = True
        self.rt.logging = True
        self.rt.log_path = log.path
        outputs: dict[str, str] = {}
        try:
            outputs = self.rt.run_graph(graph, task, log)
            log.finish("done", "")
        except Exception as e:
            log.finish("failed", str(e))
            self.rt.printing = False
            self.rt.logging = False
            raise ValueError(str(e))
        self.rt.printing = False
        self.rt.logging = False
        last = list(outputs.values())[-1]
        ui.out(ui.bold("\n最終輸出:\n") + last + ui.dim(f"\n\n(紀錄: ruai agent log {log.id})"))
        hist = msgs_from(self.s.get("histories").get(self.speaker()))
        hist.append(Msg("user", f"[I ran the '{name}' workflow with task: {task}]"))
        hist.append(Msg("assistant", f"Workflow '{name}' finished. Final output:\n{last}"))
        histories = self.s.get("histories")
        histories.set(self.speaker(), msgs_json(hist))
        self.s.set("histories", histories)
        self.save()

    def compact(self) -> None:
        hist = msgs_from(self.s.get("histories").get(self.speaker()))
        if len(hist) < 4:
            ui.out("對話太短,不需要壓縮")
            return
        transcript = "\n\n".join([f"{m.role.upper()}: {m.content[:4000]}" for m in hist])
        model = self.rt.model_for(self.speaker())
        llm = self.rt.llm(model)
        summary = llm.chat("Summarize this conversation for your own future reference: goals, decisions, "
                           "files touched, current state and open TODOs. Be dense and factual.",
                           [Msg("user", transcript)], False, 0)
        histories = self.s.get("histories")
        histories.set(self.speaker(), msgs_json([
            Msg("user", "Summary of our conversation so far:\n" + summary),
            Msg("assistant", "Understood. I'll continue from this summary.")]))
        self.s.set("histories", histories)
        ui.out(ui.dim(summary))
        ui.out(f"已壓縮 {len(hist)} 則訊息")


def start_chat(project: Project, rt: Runtime, resume: str, resuming: bool) -> None:
    session = project.load_session(resume) if resuming else project.new_session()
    chat = Chat(project, rt, session)
    if not sys.stdin.isatty():
        # piped input: run each line as a message, no prompt decoration
        for line in sys.stdin:
            if chat.handle(line) == "exit":
                break
        chat.save()
        return
    chat.loop()
