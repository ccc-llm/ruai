# error: `from PACKAGE import x as y` is not supported
from shlex import split as sp

print(sp("a b"))
