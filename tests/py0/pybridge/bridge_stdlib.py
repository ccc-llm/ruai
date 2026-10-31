"""The Python bridge: CPython modules through `# python:` stubs (py0lib/): functions, keyword
and positional arguments, Python's defaults, objects with methods / properties / operators,
static methods, iterators as lists, and Python exceptions."""
import difflib
import statistics
import textwrap
from datetime import date
from fractions import Fraction

TEXT = "The quick brown fox jumps over the lazy dog " * 3
for line in textwrap.wrap(TEXT, 30):
    print(repr(line))
print(textwrap.fill(TEXT, width=40, initial_indent="> "))
print(textwrap.shorten(TEXT, 25, placeholder="..."), textwrap.dedent("   a\n   b"))
print(statistics.mean([1, 2, 3, 4]), statistics.median([3.5, 1.0, 2.0]), round(statistics.stdev([2.0, 4.0, 4.0, 5.0]), 6), statistics.mode(["a", "b", "a"]))
h = Fraction(1, 2)
t = Fraction(1, 3)
s = h + t
print(s, repr(s), s.numerator, s.denominator, float(s), s * 6, h < t, Fraction(355, 113).limit_denominator(10), Fraction(7))
print(list(difflib.unified_diff(["a", "b", "c"], ["a", "x", "c"], "old", "new", lineterm="")))
print(difflib.get_close_matches("appel", ["ape", "apple", "peach", "puppy"]))
d = date(2026, 11, 1)
print(d, d.isoformat(), d.weekday(), d.year, date.fromisoformat("2024-02-29"), repr(d))
for bad in [0, 1, 2]:
    try:
        if bad == 0:
            textwrap.wrap(TEXT, -1)
        elif bad == 1:
            Fraction(1, 0)
        else:
            date(2026, 13, 1)
    except ValueError as e:
        print("ValueError:", e)
    except ZeroDivisionError as e:
        print("ZeroDivisionError:", e)
