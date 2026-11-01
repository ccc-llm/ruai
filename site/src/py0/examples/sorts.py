# Sorting algorithms, checked against sorted().
class Rng:
    """A linear congruential generator (the constants of C's rand)."""

    def __init__(self, seed: int) -> None:
        self.state = seed

    def next(self) -> int:
        self.state = (self.state * 1103515245 + 12345) % 2147483648
        return self.state >> 8

    def below(self, n: int) -> int:
        return self.next() % n


def quicksort(xs: list[int]) -> list[int]:
    if len(xs) <= 1:
        return xs
    p = xs[len(xs) // 2]
    return quicksort([x for x in xs if x < p]) + [x for x in xs if x == p] + quicksort([x for x in xs if x > p])


def merge_sort(xs: list[int]) -> list[int]:
    if len(xs) <= 1:
        return xs
    m = len(xs) // 2
    a = merge_sort(xs[:m])
    b = merge_sort(xs[m:])
    out: list[int] = []
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    return out + a[i:] + b[j:]


def insertion_sort(xs: list[int]) -> list[int]:
    ys = xs.copy()
    for i in range(1, len(ys)):
        k = ys[i]
        j = i - 1
        while j >= 0 and ys[j] > k:
            ys[j + 1] = ys[j]
            j -= 1
        ys[j + 1] = k
    return ys


rng = Rng(1)
data = [rng.below(1000) for _ in range(2000)]
expected = sorted(data)
print("     quicksort:", quicksort(data) == expected)
print("    merge sort:", merge_sort(data) == expected)
print("insertion sort:", insertion_sort(data) == expected)
print("first ten:", sorted(data)[:10])
