"""CPython stand-in for the py0 package `ruai.json` (src/pyrt/lib/ruai_json.rs)."""

import copy as _copy
import json as _json


def _wrap(v):
    j = Json.__new__(Json)
    j._v = v
    return j


class Json:
    def __init__(self) -> None:
        self._v = None

    @staticmethod
    def loads(s):
        return _wrap(_json.loads(s))

    @staticmethod
    def null():
        return _wrap(None)

    @staticmethod
    def of_bool(b):
        return _wrap(bool(b))

    @staticmethod
    def of_int(i):
        return _wrap(int(i))

    @staticmethod
    def of_float(x):
        return _wrap(float(x))

    @staticmethod
    def of_str(s):
        return _wrap(str(s))

    @staticmethod
    def of_strs(xs):
        return _wrap([str(x) for x in xs])

    @staticmethod
    def new_list():
        return _wrap([])

    @staticmethod
    def new_dict():
        return _wrap({})

    def kind(self):
        v = self._v
        if v is None:
            return "null"
        return {bool: "bool", int: "int", float: "float", str: "str", list: "list", dict: "dict"}[type(v)]

    def is_null(self):
        return self._v is None

    def truthy(self):
        return bool(self._v)

    def str_or(self, default):
        return self._v if isinstance(self._v, str) else default

    def int_or(self, default):
        return self._v if type(self._v) is int else default

    def float_or(self, default):
        return float(self._v) if type(self._v) in (int, float) else default

    def bool_or(self, default):
        return self._v if isinstance(self._v, bool) else default

    def text(self):
        return self._v if isinstance(self._v, str) else _json.dumps(self._v, ensure_ascii=False)

    def get(self, key):
        return _wrap(_copy.deepcopy(self._v.get(key)) if isinstance(self._v, dict) else None)

    def get_str(self, key, default):
        return self.get(key).str_or(default)

    def has(self, key):
        return isinstance(self._v, dict) and key in self._v

    def keys(self):
        return list(self._v) if isinstance(self._v, dict) else []

    def at(self, i):
        v = self._v
        if isinstance(v, list) and -len(v) <= i < len(v):
            return _wrap(_copy.deepcopy(v[i]))
        return _wrap(None)

    def items(self):
        v = self._v
        if isinstance(v, dict):
            v = list(v.values())
        return [_wrap(_copy.deepcopy(x)) for x in v] if isinstance(v, list) else []

    def strs(self):
        return [x for x in self._v if isinstance(x, str)] if isinstance(self._v, list) else []

    def size(self):
        return len(self._v) if isinstance(self._v, (list, dict, str)) else 0

    def set(self, key, v):
        if not isinstance(self._v, dict):
            self._v = {}
        self._v[key] = _copy.deepcopy(v._v)

    def remove(self, key):
        if isinstance(self._v, dict):
            self._v.pop(key, None)

    def append(self, v):
        if not isinstance(self._v, list):
            self._v = []
        self._v.append(_copy.deepcopy(v._v))

    def copy(self):
        return _wrap(_copy.deepcopy(self._v))

    def dumps(self, indent=None, ensure_ascii=True):
        return _json.dumps(self._v, indent=indent, ensure_ascii=ensure_ascii)

    def __str__(self):
        return _json.dumps(self._v, ensure_ascii=False)

    def __repr__(self):
        return "Json(" + _json.dumps(self._v, ensure_ascii=False) + ")"
