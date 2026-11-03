# Conway's Game of Life on a small torus: a glider crossing the board.
W = 20
H = 10


def step(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    counts: dict[tuple[int, int], int] = {}
    for x, y in cells:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    k = ((x + dx) % W, (y + dy) % H)
                    counts[k] = counts.get(k, 0) + 1
    return {k for k, n in counts.items() if n == 3 or (n == 2 and k in cells)}


def show(cells: set[tuple[int, int]]) -> str:
    return "\n".join("".join("█" if (x, y) in cells else "·" for x in range(W)) for y in range(H))


cells = {(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)}
for gen in range(0, 9, 4):
    print(f"generation {gen}")
    print(show(cells))
    for _ in range(4):
        cells = step(cells)
