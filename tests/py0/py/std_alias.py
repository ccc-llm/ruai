import fnmatch as fm
import html as h
import shlex

def matching(names: list[str], pat: str) -> list[str]:
    return [n for n in names if fm.fnmatch(n, pat)]

print(matching(["a.py", "b.rs", "c.py"], "*.py"), fm.filter(["x1", "y2"], "?1"))
print(h.escape("<b>"), h.unescape("&lt;b&gt;"))
words: list[str] = shlex.split("one 'two three'")
print(len(words), words[-1])
