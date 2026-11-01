# try / except / finally, and Python's integer and float semantics.
def parse(s: str) -> int:
    try:
        return int(s)
    except ValueError as e:
        print("bad input:", e)
        return -1
    finally:
        print("parsed", repr(s))


print(parse("42"), parse("4x2"))
xs = [1, 2, 3]
try:
    print(xs[10])
except IndexError as e:
    print("IndexError:", e)
print(10 // 3, -7 % 3, 2 ** 62, 1 / 3)
