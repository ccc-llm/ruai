"""A fake LLM server for tests/agent.rs: OpenAI-compatible /v1/chat/completions
and Anthropic /v1/messages, streaming (server-sent events) or not. It replies
`[fake] <last user message>`, with some reasoning first; a message containing
`CALLTOOL` gets a native tool call (write_file) instead; `FAIL500` fails once
with HTTP 500 (the client retries). Prints its port, then serves until killed."""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

failed = set()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def sse(self, events):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for e in events:
            self.wfile.write(b"data: " + (e if isinstance(e, bytes) else json.dumps(e).encode()) + b"\n\n")
            self.wfile.flush()

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        last = req["messages"][-1]["content"]
        if "FAIL500" in last and last not in failed:
            failed.add(last)
            self.send_json(500, {"error": "try again"})
            return
        if self.headers.get("Authorization", self.headers.get("x-api-key", "")) in ("", "Bearer "):
            self.send_json(401, {"error": "no key"})
            return
        text = "[fake] " + last
        if self.path.endswith("/chat/completions"):
            if not req.get("stream"):
                msg = {"role": "assistant", "content": text}
                self.send_json(200, {"choices": [{"message": msg}]})
                return
            if "CALLTOOL" in last:
                args = json.dumps({"path": "tool.txt", "content": "from a native tool call"})
                self.sse([{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "write_file", "arguments": args[:20]}}]}}]},
                          {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": args[20:]}}]}}]},
                          b"[DONE]"])
                return
            events = [{"choices": [{"delta": {"reasoning_content": "hmm"}}]}]
            for i in range(0, len(text), 5):
                events.append({"choices": [{"delta": {"content": text[i:i + 5]}}]})
            events.append(b"not json")
            events.append({"choices": []})
            events.append(b"[DONE]")
            self.sse(events)
        else:
            if not req.get("stream"):
                self.send_json(200, {"content": [{"type": "text", "text": text}]})
                return
            events = [{"type": "message_start"},
                      {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hmm"}}]
            for i in range(0, len(text), 5):
                events.append({"type": "content_block_delta", "delta": {"type": "text_delta", "text": text[i:i + 5]}})
            events.append({"type": "message_stop"})
            self.sse(events)


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
print(server.server_address[1], flush=True)
server.serve_forever()
