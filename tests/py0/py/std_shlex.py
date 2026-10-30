import shlex
from shlex import quote

cases: list[str] = [
    "a b c", "a 'b c' d", 'a "b \\" c" d', "a\\ b", "'' x", '""', "a'b'c",
    'x "a\\\\b" y', '"\\$HOME" \'$x\'', "  lead trail  ", "tab\there", "mix'ed q'\"uo\"tes",
    "a\\\nb", "", "   ", "\\'", "é 'ü x'", 'say "\\n"', "a''b", "'a'\"b\"c",
]
for c in cases:
    print(repr(c), shlex.split(c))
for bad in ["'unclosed", '"also', "trail\\", "ok 'no end"]:
    try:
        print(shlex.split(bad))
    except ValueError as e:
        print("ValueError:", e)
for w in ["", "plain", "has space", "it's", "a$b", "tab\t", "é", "-v", "x=1,y", "@%+=:,./-_", "a\"b"]:
    print(repr(w), quote(w))
print(shlex.join(["a", "b c", "it's", ""]))
print(shlex.split(shlex.join(["round", "trip's", "  x  ", ""])))
