"""LLM providers.

All providers expose the same interface: LLM(cfg).chat(system, messages, stream, think) -> str

- echo       : offline test provider, replies with the last user message
- openai     : any OpenAI-compatible /chat/completions endpoint
               (openai, nvidia, openrouter, deepseek, groq, gemini, ollama, lmstudio, custom base_url)
- anthropic  : Anthropic Messages API

`stream` prints the reply as it arrives; `think` shows reasoning: 0 not at all,
1 a "(思考中…)" marker, 2 the reasoning text (dimmed).
"""
import html
import os
import time
from dataclasses import dataclass

from ruai.http import Response, request
from ruai.json import Json

import ui

PROVIDERS_JSON = """{
  "echo": {"kind": "echo"},
  "openai": {"kind": "openai", "base_url": "https://api.openai.com/v1", "env": "OPENAI_API_KEY"},
  "nvidia": {"kind": "openai", "base_url": "https://integrate.api.nvidia.com/v1", "env": "NVIDIA_API_KEY"},
  "openrouter": {"kind": "openai", "base_url": "https://openrouter.ai/api/v1", "env": "OPENROUTER_API_KEY"},
  "deepseek": {"kind": "openai", "base_url": "https://api.deepseek.com/v1", "env": "DEEPSEEK_API_KEY"},
  "groq": {"kind": "openai", "base_url": "https://api.groq.com/openai/v1", "env": "GROQ_API_KEY"},
  "gemini": {"kind": "openai", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "env": "GEMINI_API_KEY"},
  "ollama": {"kind": "openai", "base_url": "http://localhost:11434/v1", "env": null},
  "lmstudio": {"kind": "openai", "base_url": "http://localhost:1234/v1", "env": null},
  "anthropic": {"kind": "anthropic", "base_url": "https://api.anthropic.com/v1", "env": "ANTHROPIC_API_KEY"}
}"""



class LLMError(Exception):
    pass


def providers() -> Json:
    return Json.loads(PROVIDERS_JSON)


@dataclass
class Msg:
    role: str
    content: str


def msgs_json(messages: list[Msg]) -> Json:
    out = Json.new_list()
    for m in messages:
        d = Json.new_dict()
        d.set("role", Json.of_str(m.role))
        d.set("content", Json.of_str(m.content))
        out.append(d)
    return out


def msgs_from(j: Json) -> list[Msg]:
    return [Msg(m.get_str("role", "user"), m.get_str("content", "")) for m in j.items()]


class Printer:
    """Streams a reply to the terminal; shows reasoning as `think` asks."""

    def __init__(self, stream: bool, think: int) -> None:
        self.stream = stream
        self.think = think
        self.thinking = False

    def token(self, t: str) -> None:
        if not self.stream:
            return
        if self.thinking:
            self.thinking = False
            ui.out()
        ui.out(t, end="")

    def reasoning(self, t: str) -> None:
        if self.think == 1 and not self.thinking:
            self.thinking = True
            ui.out(ui.dim("(思考中…)"), end="")
        elif self.think == 2:
            ui.out(ui.dim(t), end="")


class LLM:
    def __init__(self, cfg: Json) -> None:
        self.cfg = cfg
        self.provider = cfg.get_str("provider", "openai")
        preset = providers().get(self.provider)
        self.kind = cfg.get_str("kind", preset.get_str("kind", "openai"))
        self.model = cfg.get_str("model", "")
        self.base_url = cfg.get_str("base_url", preset.get_str("base_url", "")).rstrip("/")
        self.env = preset.get_str("env", "")
        if cfg.has("api_key_env"):
            self.env = cfg.get_str("api_key_env", "")
        self.api_key = cfg.get_str("api_key", "")
        if self.api_key == "" and self.env != "":
            key = os.environ.get(self.env)
            if key is not None:
                self.api_key = key
        self.timeout = cfg.get("timeout").float_or(600.0)
        if self.kind != "echo" and self.base_url == "":
            raise LLMError(f"provider '{self.provider}' 需要設定 base_url")

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"

    def chat(self, system: str, messages: list[Msg], stream: bool, think: int) -> str:
        p = Printer(stream, think)
        if self.kind == "echo":
            text = "[echo] " + (messages[-1].content if len(messages) > 0 else "")
            p.token(text)
            return text
        if self.env != "" and self.api_key == "":
            raise LLMError(f"{self.label}: 缺少 API key,請設定環境變數 {self.env}")
        for attempt in range(4):
            try:
                if self.kind == "openai":
                    return self.openai(system, messages, p)
                return self.anthropic(system, messages, p)
            except HTTPStatus as e:
                if e.code in [408, 429, 500, 502, 503, 504] and attempt < 3:
                    time.sleep(2.0 ** attempt * 2)
                    continue
                raise LLMError(f"{self.label} HTTP {e.code}: {e.body}")
            except OSError as e:
                if attempt < 3:
                    time.sleep(2.0 ** attempt * 2)
                    continue
                raise LLMError(f"{self.label} 連線失敗: {e}")
        raise LLMError(f"{self.label}: 重試失敗")

    def post(self, url: str, headers: dict[str, str], body: Json) -> "Reply":
        r = request("POST", url, headers, body.dumps(), self.timeout)
        if r.status >= 400:
            raise HTTPStatus(r.status, r.read()[:2000])
        return Reply(r.read() if not body.get("stream").bool_or(False) else "", r)

    def openai(self, system: str, messages: list[Msg], p: Printer) -> str:
        msgs = Json.new_list()
        if system != "":
            msgs = msgs_json([Msg("system", system)])
        for m in msgs_json(messages).items():
            msgs.append(m)
        body = Json.new_dict()
        body.set("model", Json.of_str(self.model))
        body.set("messages", msgs)
        body.set("stream", Json.of_bool(p.stream or p.think > 0))
        for k in ["temperature", "top_p", "max_tokens"]:
            if self.cfg.has(k):
                body.set(k, self.cfg.get(k))
        extra = self.cfg.get("extra")
        for k in extra.keys():
            body.set(k, extra.get(k))
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key != "":
            headers["Authorization"] = f"Bearer {self.api_key}"
        reply = self.post(f"{self.base_url}/chat/completions", headers, body)
        if not body.get("stream").bool_or(False):
            msg = Json.loads(reply.text).get("choices").at(0).get("message")
            return msg.get_str("content", "") + tool_calls_to_tags(msg.get("tool_calls").items())
        parts: list[str] = []
        names: dict[int, str] = {}
        args: dict[int, str] = {}
        while True:
            ev = reply.event()
            if ev.is_null():
                break
            choices = ev.get("choices")
            if choices.size() == 0:
                continue
            delta = choices.at(0).get("delta")
            for tc in delta.get("tool_calls").items():
                i = tc.get("index").int_or(len(names))
                if i not in names:
                    names[i] = ""
                    args[i] = ""
                fn = tc.get("function")
                names[i] += fn.get_str("name", "")
                args[i] += fn.get_str("arguments", "")
            think = delta.get_str("reasoning_content", "")
            if think == "":
                think = delta.get_str("reasoning", "")
            if think != "":
                p.reasoning(think)
            text = delta.get_str("content", "")
            if text != "":
                parts.append(text)
                p.token(text)
        calls: list[Json] = []
        for i in sorted(names.keys()):
            fn = Json.new_dict()
            fn.set("name", Json.of_str(names[i]))
            fn.set("arguments", Json.of_str(args[i]))
            tc = Json.new_dict()
            tc.set("function", fn)
            calls.append(tc)
        tags = tool_calls_to_tags(calls)
        if tags != "":
            p.token(tags)
        return "".join(parts) + tags

    def anthropic(self, system: str, messages: list[Msg], p: Printer) -> str:
        body = Json.new_dict()
        body.set("model", Json.of_str(self.model))
        body.set("messages", msgs_json(messages))
        body.set("max_tokens", Json.of_int(self.cfg.get("max_tokens").int_or(8192)))
        body.set("stream", Json.of_bool(p.stream or p.think > 0))
        if system != "":
            body.set("system", Json.of_str(system))
        if self.cfg.has("temperature"):
            body.set("temperature", self.cfg.get("temperature"))
        extra = self.cfg.get("extra")
        for k in extra.keys():
            body.set(k, extra.get(k))
        headers: dict[str, str] = {"Content-Type": "application/json", "x-api-key": self.api_key,
                                   "anthropic-version": "2023-06-01"}
        reply = self.post(f"{self.base_url}/messages", headers, body)
        if not body.get("stream").bool_or(False):
            data = Json.loads(reply.text)
            return "".join([b.get_str("text", "") for b in data.get("content").items() if b.get_str("type", "") == "text"])
        parts: list[str] = []
        while True:
            ev = reply.event()
            if ev.is_null():
                break
            if ev.get_str("type", "") == "error":
                raise LLMError(f"{self.label}: {ev.get('error').text()}")
            if ev.get_str("type", "") != "content_block_delta":
                continue
            delta = ev.get("delta")
            if delta.get_str("type", "") == "text_delta":
                t = delta.get_str("text", "")
                parts.append(t)
                p.token(t)
            elif delta.get_str("type", "") == "thinking_delta":
                p.reasoning(delta.get_str("thinking", ""))
        return "".join(parts)


class HTTPStatus(Exception):
    def __init__(self, code: int, body: str) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code
        self.body = body


class Reply:
    """A response: its whole text, or its server-sent events one by one."""

    def __init__(self, text: str, resp: Response) -> None:
        self.text = text
        self.resp = resp

    def event(self) -> Json:
        """The next `data:` event as JSON; null at the end (or `[DONE]`)."""
        while True:
            raw = self.resp.readline()
            if raw == "":
                return Json.null()
            line = raw.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return Json.null()
            try:
                return Json.loads(data)
            except ValueError:
                continue


def tool_calls_to_tags(tool_calls: list[Json]) -> str:
    """Some servers turn tool-like output into native tool_calls; map them back to agge tags."""
    out: list[str] = []
    for tc in tool_calls:
        fn = tc.get("function")
        name = fn.get_str("name", "")
        if name == "":
            continue
        raw = fn.get_str("arguments", "")
        args = Json.new_dict()
        try:
            args = Json.loads(raw if raw != "" else "{}")
        except ValueError:
            args.set("body", Json.of_str(raw))
        if args.kind() != "dict":
            wrapped = Json.new_dict()
            wrapped.set("body", args)
            args = wrapped
        body: str | None = None
        for k in ["task", "content", "command", "cmd", "body", "text"]:
            if args.has(k):
                body = args.get(k).text()
                args.remove(k)
                break
        attrs = "".join([f' {k}="{html.escape(args.get(k).text())}"' for k in args.keys()])
        if body is not None:
            out.append(f"<{name}{attrs}>{body}</{name}>")
        else:
            out.append(f"<{name}{attrs}/>")
    return ("\n" + "\n".join(out)) if len(out) > 0 else ""


def parse_spec(spec: str) -> Json:
    """'nvidia:z-ai/glm-5.3-flash' -> {"provider": "nvidia", "model": "z-ai/glm-5.3-flash"}; null if unknown."""
    provider, _, model = spec.partition(":")
    if not providers().has(provider):
        return Json.null()
    d = Json.new_dict()
    d.set("provider", Json.of_str(provider))
    d.set("model", Json.of_str(model if model != "" else provider))
    return d
