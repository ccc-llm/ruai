import fnmatch

patterns: list[str] = [
    "*", "*.py", "?.txt", "[abc]*", "[!abc]*", "[a-c]x", "[z-a]", "[!]", "[]]", "[!]]x",
    "[", "a[", "*[0-9]", "[a-]x", "[-a]x", "[!-a]x", "a*b*c", "**", "[[]", "\\*",
    "[\\]x", "[a-c-e]x", "[!a-c-e]x", "?*?", "[^a]x", "[a-z-0]x", "*x*",
]
names: list[str] = [
    "", "a", "abc", "x.py", "a.txt", "]x", "-x", "bx", "zx", "a-b-c", "ab", "[", "a[",
    "a\\", "\\x", "dx", "ex", "\n", "axbyc", "7", "^x", "*", "0x",
]
for p in patterns:
    row = "".join("1" if fnmatch.fnmatchcase(n, p) else "0" for n in names)
    print(f"{p!r:12} {row}")
print(fnmatch.fnmatch("Readme.MD", "*.MD"), fnmatch.fnmatch("readme.md", "*.MD"))
print(fnmatch.filter(names, "*x"))
