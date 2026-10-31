"""Project state stored in .agge/ (like .git/).

.agge/
  config.json        default_model, models, teams, dirs, approval, language
  agents.json        project agents (override / extend built-ins)
  graphs/<name>.json project graphs   (graph.json = legacy graph named "default")
  sessions/          chat sessions (LAST points at the latest)
  runs/<id>/         graph run logs: meta.json, events.jsonl, <node>.md

User-wide settings live in ~/.agge/config.json (or $AGGE_HOME) and are merged under the project.
"""
import os
import time

from ruai.json import Json

import ui
from presets import builtins
from providers import parse_spec, providers

DIRNAME = ".agge"


class ProjectError(Exception):
    pass


def global_home() -> str:
    h = os.environ.get("AGGE_HOME")
    return h if h is not None else os.path.join(os.path.expanduser("~"), DIRNAME)


def default_config() -> Json:
    cfg = Json.loads("""{
  "default_model": "echo",
  "models": {
    "echo": {"provider": "echo", "model": "echo"},
    "glm": {"provider": "nvidia", "model": "z-ai/glm-5.3-flash",
            "temperature": 0.3, "top_p": 1.0, "max_tokens": 8192}
  },
  "approval": "ask",
  "dirs": []
}""")
    if os.environ.get("NVIDIA_API_KEY") is not None:
        cfg.set("default_model", Json.of_str("glm"))
    return cfg


def load_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return text


def save_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def read_json(path: str, default: Json) -> Json:
    if not os.path.isfile(path):
        return default
    try:
        return Json.loads(load_text(path))
    except ValueError:
        return default


def write_json(path: str, data: Json) -> None:
    d = os.path.dirname(path)
    if d != "":
        os.makedirs(d, exist_ok=True)
    save_text(path, data.dumps(indent=2, ensure_ascii=False) + "\n")


def merged(a: Json, b: Json) -> Json:
    """{**a, **b} of two dicts."""
    out = a.copy() if a.kind() == "dict" else Json.new_dict()
    for k in b.keys():
        out.set(k, b.get(k))
    return out


def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def sorted_files(d: str, suffix: str) -> list[str]:
    if not os.path.isdir(d):
        return []
    return sorted([os.path.join(d, n) for n in os.listdir(d) if n.endswith(suffix)])


class Project:
    def __init__(self, root: str) -> None:
        r = os.path.realpath(root)
        self.root = r
        self.dir = os.path.join(r, DIRNAME)

    # ---------- discovery ----------
    @staticmethod
    def find(start: str) -> "Project | None":
        p = os.path.realpath(start if start != "" else os.getcwd())
        home = os.path.realpath(global_home())
        while True:
            d = os.path.join(p, DIRNAME)
            if os.path.isdir(d) and os.path.realpath(d) != home:
                return Project(p)
            parent = os.path.dirname(p)
            if parent == p:
                return None
            p = parent

    @staticmethod
    def init(root: str, model: str) -> "Project":
        prj = Project(root)
        if not os.path.exists(os.path.join(prj.dir, "config.json")):
            cfg = default_config()
            if model != "":
                cfg.set("default_model", Json.of_str(model))
            write_json(os.path.join(prj.dir, "config.json"), cfg)
        if not os.path.exists(os.path.join(prj.dir, "agents.json")):
            write_json(os.path.join(prj.dir, "agents.json"), Json.new_dict())
        for sub in ["graphs", "sessions", "runs"]:
            os.makedirs(os.path.join(prj.dir, sub), exist_ok=True)
        return prj

    def path(self, *parts: str) -> str:
        p = self.dir
        for x in parts:
            p = os.path.join(p, x)
        return p

    # ---------- config ----------
    def raw_config(self) -> Json:
        return read_json(self.path("config.json"), Json.new_dict())

    def save_config(self, cfg: Json) -> None:
        write_json(self.path("config.json"), cfg)

    def config(self) -> Json:
        g = read_json(os.path.join(global_home(), "config.json"), Json.new_dict())
        p = self.raw_config()
        cfg = merged(g, p)
        cfg.set("models", merged(g.get("models"), p.get("models")))
        cfg.set("teams", merged(g.get("teams"), p.get("teams")))
        dirs: list[str] = []
        for d in g.get("dirs").strs() + p.get("dirs").strs():
            if d not in dirs:
                dirs.append(d)
        cfg.set("dirs", Json.of_strs(dirs))
        if not cfg.has("default_model"):
            cfg.set("default_model", Json.of_str("echo"))
        return cfg

    def resolve_model(self, name: str) -> Json:
        models = self.config().get("models")
        if models.has(name):
            return models.get(name)
        spec = parse_spec(name)
        if not spec.is_null():
            return spec
        raise ProjectError(f"未知的模型 '{name}'。用 `ruai agent model add` 新增,或用 provider:model 格式"
                           f"(provider: {', '.join(providers().keys())})")

    # ---------- agents / graphs / teams ----------
    def agents(self) -> Json:
        return merged(builtins().get("agents"), read_json(self.path("agents.json"), Json.new_dict()))

    def save_agent(self, name: str, spec: Json) -> None:
        data = read_json(self.path("agents.json"), Json.new_dict())
        if spec.is_null():
            data.remove(name)
        else:
            data.set(name, spec)
        write_json(self.path("agents.json"), data)

    def graphs(self) -> Json:
        out = builtins().get("graphs")
        legacy = read_json(self.path("graph.json"), Json.new_dict())
        if legacy.get("nodes").truthy():
            out.set("default", legacy)
        for f in sorted_files(self.path("graphs"), ".json"):
            out.set(stem(f), read_json(f, Json.new_dict()))
        return out

    def is_project_graph(self, name: str) -> bool:
        return os.path.exists(self.path("graphs", name + ".json"))

    def load_graph(self, name: str) -> Json:
        if name.endswith(".json") and os.path.exists(name):
            return read_json(name, Json.new_dict())
        graphs = self.graphs()
        if not graphs.has(name):
            raise ProjectError(f"找不到 graph '{name}'(可用: {', '.join(graphs.keys())})")
        return graphs.get(name)

    def save_graph(self, name: str, graph: Json) -> None:
        write_json(self.path("graphs", name + ".json"), graph)

    def delete_graph(self, name: str) -> None:
        f = self.path("graphs", name + ".json")
        if not os.path.exists(f):
            raise ProjectError(f"'{name}' 不是專案 graph(內建 graph 不能刪)")
        os.remove(f)

    def teams(self) -> Json:
        return merged(builtins().get("teams"), self.config().get("teams"))

    # ---------- sessions ----------
    def new_session(self) -> Json:
        s = Json.new_dict()
        s.set("id", Json.of_str(time.strftime("%Y%m%d-%H%M%S")))
        s.set("team", Json.null())
        s.set("speaker", Json.of_str("assistant"))
        s.set("histories", Json.new_dict())
        return s

    def save_session(self, s: Json) -> None:
        sid = s.get_str("id", "")
        write_json(self.path("sessions", sid + ".json"), s)
        save_text(self.path("sessions", "LAST"), sid)

    def load_session(self, sid: str) -> Json:
        if sid == "":
            last = self.path("sessions", "LAST")
            if not os.path.exists(last):
                raise ProjectError("沒有之前的 session")
            sid = load_text(last).strip()
        s = read_json(self.path("sessions", sid + ".json"), Json.null())
        if s.is_null():
            raise ProjectError(f"找不到 session {sid}")
        return s

    def sessions(self) -> list[str]:
        return [stem(f) for f in sorted_files(self.path("sessions"), ".json")]

    # ---------- runs ----------
    def new_run(self, graph_name: str, task: str) -> "RunLog":
        rid = time.strftime("%Y%m%d-%H%M%S") + f"-{graph_name}"
        return RunLog(self.path("runs", rid), graph_name, task)

    def runs(self) -> list[str]:
        d = self.path("runs")
        if not os.path.isdir(d):
            return []
        return sorted([n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n))])


def append_event(run_dir: str, e: ui.Event) -> None:
    """One line of a run's events.jsonl."""
    d = Json.new_dict()
    d.set("t", Json.of_float(time.time()))
    d.set("kind", Json.of_str(e.kind))
    for k, v in [("agent", e.agent), ("node", e.node), ("model", e.model), ("name", e.name),
                 ("detail", e.detail), ("target", e.target), ("task", e.task), ("msg", e.msg),
                 ("src", e.src), ("dst", e.dst)]:
        if v != "":
            d.set(k, Json.of_str(v))
    if e.kind == "node_done":
        d.set("secs", Json.of_float(e.secs))
    if e.chars > 0:
        d.set("chars", Json.of_int(e.chars))
    if e.kind == "loop":
        d.set("round", Json.of_int(e.rounds))
        d.set("max", Json.of_int(e.limit))
    if e.kind == "llm":
        d.set("step", Json.of_int(e.step))
    with open(os.path.join(run_dir, "events.jsonl"), "a", encoding="utf-8") as f:
        f.write(d.dumps(ensure_ascii=False) + "\n")


class RunLog:
    def __init__(self, path: str, graph_name: str, task: str) -> None:
        self.path = path
        self.id = os.path.basename(path)
        os.makedirs(path, exist_ok=True)
        meta = Json.new_dict()
        meta.set("id", Json.of_str(self.id))
        meta.set("graph", Json.of_str(graph_name))
        meta.set("task", Json.of_str(task))
        meta.set("started", Json.of_str(time.strftime("%Y-%m-%d %H:%M:%S")))
        meta.set("status", Json.of_str("running"))
        self.meta = meta
        write_json(os.path.join(path, "meta.json"), meta)

    def event(self, e: ui.Event) -> None:
        append_event(self.path, e)

    def save_output(self, label: str, text: str) -> None:
        safe = label.replace("/", "_")
        f = os.path.join(self.path, f"{safe}.md")
        n = 1
        while os.path.exists(f):
            n += 1
            f = os.path.join(self.path, f"{safe}.{n}.md")
        save_text(f, text)

    def finish(self, status: str, error: str) -> None:
        self.meta.set("status", Json.of_str(status))
        self.meta.set("finished", Json.of_str(time.strftime("%Y-%m-%d %H:%M:%S")))
        if error != "":
            self.meta.set("error", Json.of_str(error))
        write_json(os.path.join(self.path, "meta.json"), self.meta)
