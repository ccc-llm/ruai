"""Draft a py0 stub for a Python module (`ruai py stub --python MODULE`).

Run by the interpreter that has MODULE: python pystubgen.py MODULE. Prints the
stub. Functions, classes (constructor, methods, properties) and their
signatures come from `inspect`; annotations become py0 types where they map
(int, float, str, bool, None, list / dict / tuple / X | None of those, classes
of the module); anything else is `Object`, a Python object of no particular
type, to edit by hand. Defaults become `...` (Python's own default applies).
"""
import importlib
import inspect
import keyword
import sys
import types
import typing

MODULE = sys.argv[1]
mod = importlib.import_module(MODULE)
classes: dict[str, type] = {}
uses_object = False


def class_name(c: type) -> str:
    return c.__name__


def ty(a: object, pos: bool) -> str:
    """The py0 type of annotation `a` (`pos`: a parameter rather than a result)."""
    global uses_object
    if a is inspect.Parameter.empty or a is inspect.Signature.empty:
        uses_object = True
        return "Object"
    if a is None or a is type(None):
        return "None"
    if isinstance(a, str):
        a = {"int": int, "float": float, "str": str, "bool": bool, "None": None}.get(a, a)
        if isinstance(a, str) or a is None:
            uses_object = True
            return "Object" if a is not None else "None"
    simple = {int: "int", float: "float", str: "str", bool: "bool"}
    if a in simple:
        return simple[a]
    origin = typing.get_origin(a)
    args = typing.get_args(a)
    if origin in (list, typing.List) and len(args) == 1:
        return f"list[{ty(args[0], pos)}]"
    if origin in (dict, typing.Dict) and len(args) == 2 and args[0] is str:
        return f"dict[str, {ty(args[1], pos)}]"
    if origin in (tuple, typing.Tuple) and args and Ellipsis not in args:
        return "tuple[" + ", ".join(ty(x, pos) for x in args) + "]"
    if origin in (typing.Union, types.UnionType):
        rest = [x for x in args if x is not type(None)]
        if len(rest) == 1 and len(args) == 2:
            return f"{ty(rest[0], pos)} | None"
        if pos and all(x in simple for x in rest) and len(rest) == len(args):
            return " | ".join(simple[x] for x in rest)
    if inspect.isclass(a) and getattr(a, "__module__", "").split(".")[0] == MODULE.split(".")[0]:
        classes.setdefault(class_name(a), a)
        return class_name(a)
    uses_object = True
    return "Object"


def name_ok(n: str) -> bool:
    return n.isidentifier() and not keyword.iskeyword(n)


def params(sig: inspect.Signature, method: bool) -> str:
    out = ["self"] if method else []
    kinds = [p.kind for p in sig.parameters.values()]
    star_done = False
    for i, p in enumerate(sig.parameters.values()):
        if method and i == 0 and p.name in ("self", "cls"):
            continue
        if p.kind is p.VAR_KEYWORD:
            continue
        if p.kind is p.VAR_POSITIONAL:
            out.append(f"*{p.name}: {ty(p.annotation, True)}")
            star_done = True
            continue
        if p.kind is p.KEYWORD_ONLY and not star_done:
            out.append("*")
            star_done = True
        a = p.annotation
        # no annotation: the type of the default value
        if a is p.empty and type(p.default) in (int, float, str, bool):
            a = type(p.default)
        s = f"{p.name}: {ty(a, True)}"
        if p.default is not p.empty:
            s += " = ..."
        out.append(s)
        if p.kind is p.POSITIONAL_ONLY and (i + 1 == len(kinds) or kinds[i + 1] is not p.POSITIONAL_ONLY):
            out.append("/")
    return ", ".join(out)


def signature(f: object) -> inspect.Signature | None:
    try:
        return inspect.signature(f)
    except (ValueError, TypeError):
        return None


def function(name: str, f: object, indent: str, method: bool) -> list[str]:
    sig = signature(f)
    if sig is None:
        return [f"{indent}# {name}: no signature"]
    lines = []
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        lines.append(f"{indent}# **kwargs left out")
    lines.append(f"{indent}def {name}({params(sig, method)}) -> {ty(sig.return_annotation, False)}: ...")
    return lines


def public(n: str) -> bool:
    return not n.startswith("_") and name_ok(n)


body: list[str] = []
for n, v in sorted(vars(mod).items()):
    if not public(n):
        continue
    if inspect.isclass(v):
        if getattr(v, "__module__", "").split(".")[0] == MODULE.split(".")[0] and not issubclass(v, BaseException):
            classes.setdefault(n, v)
    elif callable(v) and (inspect.isfunction(v) or inspect.isbuiltin(v) or inspect.isroutine(v)):
        body += function(n, v, "", False)
        body.append("")

class_text: list[str] = []
done: set[str] = set()
while len(done) < len(classes):
    for n, c in list(classes.items()):
        if n in done:
            continue
        done.add(n)
        class_text.append(f"class {n}:")
        init = c.__dict__.get("__init__")
        if init is not None or c.__init__ is not object.__init__:
            sig = signature(c)
            if sig is not None:
                class_text.append(f"    def __init__({params(sig, True)}) -> None: ...")
        class_text.append("    def __str__(self) -> str: ...")
        class_text.append("    def __repr__(self) -> str: ...")
        for m, f in sorted(vars(c).items()):
            if not public(m):
                continue
            if isinstance(f, property):
                r = "Object"
                if f.fget is not None:
                    s = signature(f.fget)
                    r = ty(s.return_annotation, False) if s is not None else "Object"
                class_text.append("    @property")
                class_text.append(f"    def {m}(self) -> {r}: ...")
            elif isinstance(f, staticmethod):
                class_text.append("    @staticmethod")
                class_text += function(m, f.__func__, "    ", False)
            elif inspect.isfunction(f) or inspect.ismethoddescriptor(f):
                class_text += function(m, f, "    ", True)
        class_text.append("")

head = [
    f"# python: {MODULE}",
    f'"""py0 stub of `{MODULE}` (the real module, through CPython), drafted by',
    f"`ruai py stub --python {MODULE}`: check and tighten the types. `Object` is any",
    'Python object; `= ...` leaves an argument to Python\'s default."""',
    "",
]
if uses_object:
    head += ["", "class Object:", "    def __str__(self) -> str: ...", "    def __repr__(self) -> str: ...", ""]
print("\n".join(head + [""] + class_text + body).rstrip() + "\n")
