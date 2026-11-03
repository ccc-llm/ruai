# The N-queens puzzle by backtracking: count all solutions, draw the first.
def solve(n: int) -> list[list[int]]:
    found: list[list[int]] = []
    cols: list[int] = []

    def place(row: int) -> None:
        if row == n:
            found.append(cols.copy())
            return
        for c in range(n):
            if all(c != q and abs(c - q) != row - r for r, q in enumerate(cols)):
                cols.append(c)
                place(row + 1)
                cols.pop()

    place(0)
    return found


for n in range(4, 9):
    print(f"{n} queens: {len(solve(n))} solutions")
first = solve(8)[0]
for c in first:
    print(" ".join("Q" if i == c else "." for i in range(8)))
