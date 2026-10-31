"""CPython stand-in for the py0 package `ruai.http` (src/pyrt/lib/ruai_http.rs)."""

import urllib.error as _error
import urllib.request as _request


class Response:
    def __init__(self, status, fp):
        self._status = status
        self._fp = fp

    @property
    def status(self):
        return self._status

    def readline(self):
        return self._fp.readline().decode("utf-8", "replace")

    def read(self):
        return self._fp.read().decode("utf-8", "replace")


def request(method, url, headers, body, timeout):
    data = body.encode() if body else None
    req = _request.Request(url, data=data, headers=dict(headers), method=method)
    try:
        r = _request.urlopen(req, timeout=timeout)
        return Response(r.status, r)
    except _error.HTTPError as e:
        return Response(e.code, e)
