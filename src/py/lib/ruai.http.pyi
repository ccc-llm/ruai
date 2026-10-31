# crate: pyrt/http
"""py0 package `ruai.http`: HTTP(S) requests. The response status may be any
HTTP status (4xx and 5xx are not raised); the body is read line by line
(`readline`, "" at the end: server-sent events) or at once (`read`). A
connection failure raises OSError. CPython stand-in: py0/python/ruai/http.py."""

from rust import Result

class Response:
    @property
    def status(self) -> int: ...
    @mut
    def readline(self) -> Result[str, OSError]: ...
    @mut
    def read(self) -> Result[str, OSError]: ...

def request(method: str, url: str, headers: dict[str, str], body: str, timeout: float) -> Result[Response, OSError | ValueError]: ...
